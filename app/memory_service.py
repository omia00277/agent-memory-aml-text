import logging
import math
import re
from datetime import datetime, timezone
from typing import List, Optional, Union

from app.config import settings
from app.consolidator import _split_sentences, consolidate
from app.database import MemoryUnit, RawChunk, SessionLocal
from app.qdrant_store import qdrant_store
from app.schemas import (
    AddRequest,
    AddResponse,
    MemoryItem,
    SearchRequest,
    SearchResponse,
)
from app.siliconflow_client import sf_client

logger = logging.getLogger(__name__)

# 内容归一化。对dict字典且text字段做处理
def _normalize_content(content: Union[str, List[dict]]) -> str:
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("text")
        )
    return content


def _serialize_messages(messages: List) -> str:
    parts = []
    for msg in messages:
        content = _normalize_content(msg.content)
        parts.append(f"{msg.role}: {content}")
    return "\n".join(parts)

# 提取最早时间戳
def _extract_source_ts(messages: List) -> Optional[int]:
    timestamps = [m.timestamp for m in messages if m.timestamp is not None]
    return min(timestamps) if timestamps else None


def _ts_to_iso(ts: Optional[int]) -> Optional[str]:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _tokens(text: str) -> List[str]:
    """Lightweight CJK/ASCII tokenization for keyword overlap scoring."""
    text = text.lower()
    ascii_tokens = re.findall(r"[a-z0-9_]+", text)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    bigrams = []
    for run in cjk_runs:
        if len(run) == 1:
            bigrams.append(run)
        else:
            bigrams.extend(run[i : i + 2] for i in range(len(run) - 1))
    return ascii_tokens + bigrams


def _ngrams(text: str, n: int) -> List[str]:
    """Extract character n-grams for phrase matching."""
    text = text.lower()
    # For CJK, use character n-grams; for ASCII, use word n-grams.
    ascii_words = re.findall(r"[a-z0-9_]+", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    result = []
    if len(ascii_words) >= n:
        result.extend(
            " ".join(ascii_words[i : i + n])
            for i in range(len(ascii_words) - n + 1)
        )
    if len(cjk_chars) >= n:
        result.extend(
            "".join(cjk_chars[i : i + n])
            for i in range(len(cjk_chars) - n + 1)
        )
    return result


def _keyword_score(query: str, document: str) -> float:
    """Token overlap + phrase n-gram matching."""
    q_tokens = set(_tokens(query))
    if not q_tokens:
        return 0.0
    d_tokens = _tokens(document)
    overlap = q_tokens.intersection(d_tokens)
    if not overlap:
        return 0.0
    token_score = len(overlap) / len(q_tokens)

    # Phrase n-gram matching
    phrase_total = 0
    phrase_hits = 0
    for n in range(2, settings.keyword_ngram_max + 1):
        q_ng = _ngrams(query, n)
        if not q_ng:
            continue
        d_ng_set = set(_ngrams(document, n))
        phrase_total += len(q_ng)
        phrase_hits += sum(1 for ng in q_ng if ng in d_ng_set)
    phrase_score = phrase_hits / phrase_total if phrase_total > 0 else 0.0

    weight = settings.keyword_phrase_weight
    return (1 - weight) * token_score + weight * phrase_score


def _normalize_dense_scores(candidates: List[dict]) -> List[dict]:
    scores = [max(c.get("score", 0.0), 0.0) for c in candidates]
    if not scores:
        return candidates
    max_score = max(scores) or 1.0
    for c, s in zip(candidates, scores):
        c["dense_norm"] = s / max_score
    return candidates


def _recency_score(source_ts: Optional[int], latest_ts: Optional[int]) -> float:
    if source_ts is None or latest_ts is None:
        return 0.5
    age_ms = max(0, latest_ts - source_ts)
    # 30-day half-life in milliseconds; newer facts score closer to 1.
    half_life_ms = 30 * 24 * 3600 * 1000
    return math.pow(0.5, age_ms / half_life_ms)


_FILLER_PHRASES = {
    "好的", "嗯", "啊", "哦", "行", "可以", "知道了", "明白了", "我记住了",
    "谢谢", "不客气", "没问题", "再见", "拜拜", "是的", "对", "没错",
    "了解了", "清楚了", "收到", "好呢", "好滴", "好哒",
}


def _is_meaningful_raw_sentence(sentence: str) -> bool:
    """Filter out conversational fillers and overly short raw sentences."""
    s = sentence.strip()
    if len(s) < 5:
        return False
    # Drop if it consists only of filler phrases or punctuation
    cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", s)
    if cleaned in _FILLER_PHRASES:
        return False
    # Drop if it starts with a common filler followed by punctuation
    lower = s.lower()
    for filler in _FILLER_PHRASES:
        if lower.startswith(filler) and len(lower) <= len(filler) + 3:
            return False
    return True


def _build_query_text(query: Union[str, List[dict]], options: Optional[List[str]]) -> str:
    q = _normalize_content(query)
    if options:
        q = q + "\n" + "\n".join(options)
    return q


def add_memory(request: AddRequest) -> AddResponse:
    """Synchronous add: persist chunk and consolidated units, then make them searchable."""
    db = SessionLocal()
    try:
        existing = db.query(RawChunk).filter(RawChunk.request_id == request.request_id).first()
        if existing:
            logger.info(f"Duplicate Add request_id={request.request_id}; returning cached success.")
            return AddResponse(
                success=True,
                request_id=request.request_id,
                user_id=request.user_id,
                session_id=request.session_id,
            )

        text = _serialize_messages(request.messages)
        source_ts = _extract_source_ts(request.messages)
        created_at = datetime.now(timezone.utc)

        raw = RawChunk(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            content=text,
            messages=[m.model_dump() for m in request.messages],
            source_ts=source_ts,
            created_at=created_at,
        )
        db.add(raw)
        db.commit()
        db.refresh(raw)

        consolidated = consolidate(text)
        if not consolidated:
            consolidated = [text]

        # Add raw sentences as fallback units to preserve original details
        # that LLM consolidation may drop or over-generalize.
        raw_sentences = []
        if settings.enable_raw_sentence_fallback:
            for s in _split_sentences(text):
                # Skip conversational fillers and overly short sentences
                if not _is_meaningful_raw_sentence(s):
                    continue
                # Keep raw sentence if it is not already subsumed by a consolidated fact
                if not any(s in fact or fact in s for fact in consolidated):
                    raw_sentences.append(s)

        # Preserve order: facts first, raw fallbacks after; deduplicate exact strings
        seen = set()
        unit_contents = []
        unit_types = []
        for content, unit_type in [(c, "fact") for c in consolidated] + [
            (c, "raw") for c in raw_sentences
        ]:
            key = content.strip()
            if key and key not in seen:
                seen.add(key)
                unit_contents.append(content)
                unit_types.append(unit_type)

        vectors = sf_client.embed(unit_contents)

        for idx, content in enumerate(unit_contents):
            unit = MemoryUnit(
                user_id=request.user_id,
                session_id=request.session_id,
                source_request_id=request.request_id,
                content=content,
                unit_type=unit_types[idx],
                source_ts=source_ts,
                created_at=created_at,
            )
            db.add(unit)
            db.flush()
            qdrant_store.upsert(
                user_id=request.user_id,
                session_id=request.session_id,
                content=content,
                vector=vectors[idx],
                point_id=unit.id,
                created_at=created_at,
                unit_type=unit_types[idx],
                source_ts=source_ts,
                source_request_id=request.request_id,
            )

        db.commit()

        return AddResponse(
            success=True,
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )
    except Exception as e:
        db.rollback()
        logger.exception("Add memory failed")
        raise
    finally:
        db.close()


def search_memory(request: SearchRequest) -> SearchResponse:
    """Search within user scope, hybrid-score, rerank, and return top_k memories."""
    query_text = _build_query_text(request.query, request.options)

    query_vectors = sf_client.embed([query_text])
    query_vector = query_vectors[0]

    recall = min(request.top_k * settings.dense_recall_multiplier, 1000)
    candidates = qdrant_store.search(
        query_vector=query_vector,
        user_id=request.user_id,
        top_k=request.top_k,
        limit=recall,
    )

    if not candidates:
        return SearchResponse(data=[])

    latest_ts = max(
        (c.get("source_ts") for c in candidates if c.get("source_ts") is not None),
        default=None,
    )
    candidates = _normalize_dense_scores(candidates)

    plain_query = _normalize_content(request.query)
    for c in candidates:
        kw = _keyword_score(plain_query, c.get("content", ""))
        rec = _recency_score(c.get("source_ts"), latest_ts)
        c["hybrid_score"] = (
            settings.dense_weight * c.get("dense_norm", 0.0)
            + settings.keyword_weight * kw
            + settings.recency_weight * rec
        )

    candidates.sort(key=lambda c: c["hybrid_score"], reverse=True)
    recall_candidates = candidates[: recall]

    documents = [c["content"] for c in recall_candidates]
    rerank_results = sf_client.rerank(
        query=query_text,
        documents=documents,
        top_n=request.top_k,
    )

    if rerank_results:
        rerank_scores = {r["index"]: r["relevance_score"] for r in rerank_results}
        ordered = []
        for r in rerank_results:
            idx = r["index"]
            if 0 <= idx < len(recall_candidates):
                c = recall_candidates[idx]
                c["score"] = r["relevance_score"]
                ordered.append(c)
        returned_idx = {r["index"] for r in rerank_results}
        remaining = [
            c for i, c in enumerate(recall_candidates) if i not in returned_idx
        ]
        remaining.sort(key=lambda c: c["hybrid_score"], reverse=True)
        ordered.extend(remaining)
    else:
        ordered = recall_candidates
        for c in ordered:
            c["score"] = c["hybrid_score"]

    final = ordered[: request.top_k]

    data = [
        MemoryItem(
            id=c["id"],
            content=c["content"],
            score=c.get("score"),
            created_at=_ts_to_iso(c.get("source_ts")) or c.get("created_at"),
        )
        for c in final
    ]

    return SearchResponse(data=data)
