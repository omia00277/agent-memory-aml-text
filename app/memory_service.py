import json
import logging
import math
import re
from datetime import datetime, timezone
from typing import List, Optional, Union

from app.config import settings
from app.consolidator import _split_sentences, consolidate, fact_text
from app.database import Episode, MemoryUnit, RawChunk, SessionLocal
from app.qdrant_store import qdrant_store
from app.schemas import AddRequest, AddResponse, MemoryItem, SearchRequest, SearchResponse
from app.siliconflow_client import sf_client

logger = logging.getLogger(__name__)


def _normalize_content(content: Union[str, List[dict]]) -> str:
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("text")
        )
    return str(content)


def _serialize_messages(messages: List) -> str:
    return "\n".join(f"{msg.role}: {_normalize_content(msg.content)}" for msg in messages)


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
    text = text.lower()
    ascii_tokens = re.findall(r"[a-z0-9_]+", text)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    bigrams = []
    for run in cjk_runs:
        bigrams.extend(run if len(run) == 1 else (run[i : i + 2] for i in range(len(run) - 1)))
    return ascii_tokens + bigrams


def _ngrams(text: str, n: int) -> List[str]:
    text = text.lower()
    ascii_words = re.findall(r"[a-z0-9_]+", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    result = []
    if len(ascii_words) >= n:
        result.extend(" ".join(ascii_words[i : i + n]) for i in range(len(ascii_words) - n + 1))
    if len(cjk_chars) >= n:
        result.extend("".join(cjk_chars[i : i + n]) for i in range(len(cjk_chars) - n + 1))
    return result


def _keyword_score(query: str, document: str) -> float:
    q_tokens = set(_tokens(query))
    if not q_tokens:
        return 0.0
    overlap = q_tokens.intersection(_tokens(document))
    if not overlap:
        return 0.0
    token_score = len(overlap) / len(q_tokens)
    total = hits = 0
    for n in range(2, settings.keyword_ngram_max + 1):
        q_ng = _ngrams(query, n)
        d_ng = set(_ngrams(document, n))
        total += len(q_ng)
        hits += sum(1 for ng in q_ng if ng in d_ng)
    phrase_score = hits / total if total else 0.0
    return (1 - settings.keyword_phrase_weight) * token_score + settings.keyword_phrase_weight * phrase_score


def _normalize_dense_scores(candidates: List[dict]) -> List[dict]:
    max_score = max((max(c.get("score", 0.0), 0.0) for c in candidates), default=1.0) or 1.0
    for candidate in candidates:
        candidate["dense_norm"] = max(candidate.get("score", 0.0), 0.0) / max_score
    return candidates


def _recency_score(source_ts: Optional[int], latest_ts: Optional[int]) -> float:
    if source_ts is None or latest_ts is None:
        return 0.5
    return math.pow(0.5, max(0, latest_ts - source_ts) / (30 * 24 * 3600 * 1000))


_FILLER_PHRASES = {"好的", "嗯", "啊", "哦", "行", "可以", "知道了", "明白了", "谢谢", "不客气", "没问题", "再见", "拜拜", "是的", "对", "没错", "收到"}


def _is_meaningful_raw_sentence(sentence: str) -> bool:
    s = sentence.strip()
    if len(s) < 5:
        return False
    cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", s)
    return cleaned not in _FILLER_PHRASES


def _contains_temporal_info(text: str) -> bool:
    patterns = [r"\d{4}年", r"\d{1,2}月", r"\d{1,2}日", r"\b\d{4}-\d{2}-\d{2}\b", r"\b(?:last|next|this|every|each)\s+\w+\b", r"\b(?:first|then|later|earlier|finally|next|previously)\b"]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _text_similarity(a: str, b: str) -> float:
    aa, bb = set(_tokens(a)), set(_tokens(b))
    return len(aa & bb) / len(aa | bb) if aa and bb else 0.0


def _is_near_duplicate(a: str, b: str, threshold: float) -> bool:
    a, b = a.strip().lower(), b.strip().lower()
    return not a or not b or a == b or a in b or b in a or _text_similarity(a, b) >= threshold


def _deduplicate_results(candidates: List[dict]) -> List[dict]:
    selected = []
    for candidate in candidates:
        if any(_is_near_duplicate(candidate.get("content", ""), x.get("content", ""), settings.search_dedup_threshold) for x in selected):
            continue
        selected.append(candidate)
    return selected


def _normalize_fact(item, default_role: Optional[str] = None) -> dict:
    if isinstance(item, str):
        return {"text": item.strip(), "evidence": item.strip(), "role": default_role}
    if not isinstance(item, dict):
        return {}
    text = str(item.get("text") or item.get("content") or item.get("fact") or "").strip()
    if not text:
        return {}
    result = {"text": text, "evidence": str(item.get("evidence") or text).strip()}
    for key in ("source_id", "role", "subject", "question", "action"):
        if item.get(key) is not None:
            result[key] = str(item[key]).strip()
    if not result.get("role"):
        result["role"] = default_role
    return result


def _event_decision(fact: dict, candidates: List[Episode]) -> Optional[str]:
    if not candidates or not settings.enable_episode_building:
        return None
    schema = {
        "type": "object",
        "properties": {"same_episode_id": {"type": ["string", "null"]}},
        "required": ["same_episode_id"],
        "additionalProperties": False,
    }
    descriptions = "\n".join(f"{e.id}: {e.title} | {e.summary}" for e in candidates)
    prompt = f"新事实：{fact['text']}\n主题：{fact.get('subject','')}\n问题：{fact.get('question','')}\n动作：{fact.get('action','')}\n候选事件：\n{descriptions}\n只有在明确属于同一具体事件时返回候选 id，否则返回 null。"
    try:
        result = sf_client.chat_structured(
            [{"role": "system", "content": "你是事件归属判定器。保守判断，不要仅凭主题相似合并不同发生时点的事件。"}, {"role": "user", "content": prompt}],
            "episode_decision",
            schema,
            max_tokens=256,
        )
        chosen = result.get("same_episode_id")
        return chosen if chosen in {e.id for e in candidates} else None
    except Exception:
        fact_text_lower = fact["text"].lower()
        best = max(candidates, key=lambda e: _text_similarity(fact_text_lower, f"{e.title} {e.summary}"))
        return best.id if _text_similarity(fact_text_lower, f"{best.title} {best.summary}") >= 0.55 else None


def _assign_episode(db, fact: dict, vector: List[float], user_id: str, session_id: str, source_id: str, source_ts: Optional[int]) -> Optional[Episode]:
    if not settings.enable_episode_building:
        return None
    nearest = qdrant_store.search(
        query_vector=vector,
        user_id=user_id,
        top_k=settings.episode_candidate_limit,
        limit=settings.episode_candidate_limit,
        facts_only=True,
    )
    episode_ids = list(dict.fromkeys(c.get("episode_id") for c in nearest if c.get("episode_id")))
    candidates = []
    if episode_ids:
        candidates = db.query(Episode).filter(Episode.user_id == user_id, Episode.id.in_(episode_ids)).all()
    if not candidates:
        recent = db.query(Episode).filter(Episode.user_id == user_id).order_by(Episode.updated_at.desc()).limit(settings.episode_candidate_limit).all()
        candidates = [e for e in recent if _keyword_score(fact["text"], f"{e.title} {e.summary}") > 0]
    if not candidates:
        return _create_episode(db, fact, user_id, session_id, source_id)
    # Fast rule-based decision to avoid excessive LLM calls
    scored = []
    fact_lower = fact["text"].lower()
    for ep in candidates:
        sim = _text_similarity(fact_lower, f"{ep.title} {ep.summary}".lower())
        scored.append((sim, ep))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_sim, best_ep = scored[0]
    if best_sim >= 0.6:
        return _update_episode(best_ep, fact["text"], source_id)
    if best_sim <= 0.25 and len(candidates) >= 2:
        return _create_episode(db, fact, user_id, session_id, source_id)
    # Ambiguous: ask LLM
    episode_id = _event_decision(fact, candidates)
    if episode_id:
        episode = db.query(Episode).filter(Episode.id == episode_id, Episode.user_id == user_id).first()
        if episode:
            return _update_episode(episode, fact["text"], source_id)
    return _create_episode(db, fact, user_id, session_id, source_id)


def _create_episode(db, fact: dict, user_id: str, session_id: str, source_id: str) -> Episode:
    now = datetime.now(timezone.utc)
    episode = Episode(
        user_id=user_id,
        session_id=session_id,
        title=(fact.get("subject") or fact["text"])[:500],
        source_id=source_id,
        summary=fact["text"][:4000],
        created_at=now,
        updated_at=now,
    )
    db.add(episode)
    db.flush()
    return episode


def _update_episode(episode: Episode, fact_text: str, source_id: str) -> Episode:
    now = datetime.now(timezone.utc)
    episode.updated_at = now
    if fact_text not in episode.summary:
        episode.summary = f"{episode.summary}；{fact_text}"[:4000]
    if source_id and not episode.source_id:
        episode.source_id = source_id
    return episode


def _episode_context(db, episode_id: Optional[str], user_id: str) -> str:
    if not episode_id:
        return ""
    facts = db.query(MemoryUnit).filter(MemoryUnit.episode_id == episode_id, MemoryUnit.user_id == user_id, MemoryUnit.unit_type == "fact").order_by(MemoryUnit.created_at.asc()).limit(settings.episode_max_facts).all()
    return "\n".join(f"- {f.content}" for f in facts)


def add_memory(request: AddRequest) -> AddResponse:
    db = SessionLocal()
    try:
        existing = db.query(RawChunk).filter(RawChunk.request_id == request.request_id).first()
        if existing:
            return AddResponse(success=True, request_id=request.request_id, user_id=request.user_id, session_id=request.session_id)
        text = _serialize_messages(request.messages)
        source_ts = _extract_source_ts(request.messages)
        created_at = datetime.now(timezone.utc)
        raw = RawChunk(request_id=request.request_id, user_id=request.user_id, session_id=request.session_id, content=text, messages=[m.model_dump() for m in request.messages], source_ts=source_ts, created_at=created_at)
        db.add(raw)
        db.flush()
        consolidated = [_normalize_fact(item) for item in consolidate(text)]
        consolidated = [f for f in consolidated if f.get("text")]
        if not consolidated:
            consolidated = [{"text": text, "evidence": text}]

        raw_sentences = []
        if settings.enable_raw_sentence_fallback:
            for sentence in _split_sentences(text):
                sentence_text = sentence.get("text", sentence) if isinstance(sentence, dict) else sentence
                if _is_meaningful_raw_sentence(sentence_text) and not any(_is_near_duplicate(sentence_text, fact["text"], settings.search_dedup_threshold) for fact in consolidated):
                    raw_sentences.append({"text": sentence_text, "evidence": sentence_text, "role": sentence.get("role") if isinstance(sentence, dict) else None})
            raw_sentences = raw_sentences[: settings.raw_fallback_per_chunk]

        facts = []
        seen = set()
        for fact, unit_type in [(f, "fact") for f in consolidated] + [(f, "raw") for f in raw_sentences]:
            key = fact["text"].strip()
            if key and key not in seen:
                seen.add(key)
                facts.append((fact, unit_type))
        fact_indexes = [idx for idx, (_, unit_type) in enumerate(facts) if unit_type == "fact"]
        fact_vectors = sf_client.embed([facts[idx][0]["text"] for idx in fact_indexes])
        vector_by_index = dict(zip(fact_indexes, fact_vectors))
        for idx, (fact, unit_type) in enumerate(facts):
            episode = _assign_episode(db, fact, vector_by_index[idx], request.user_id, request.session_id, request.request_id, source_ts) if unit_type == "fact" else None
            unit = MemoryUnit(
                user_id=request.user_id,
                session_id=request.session_id,
                source_request_id=request.request_id,
                content=fact["text"],
                unit_type=unit_type,
                evidence=fact.get("evidence"),
                source_id=fact.get("source_id") or request.request_id,
                role=fact.get("role"),
                subject=fact.get("subject"),
                question=fact.get("question"),
                action=fact.get("action"),
                fact_type=fact.get("fact_type"),
                event_title=fact.get("event_title"),
                event_subject=fact.get("event_subject"),
                event_problem=fact.get("event_problem"),
                event_time=fact.get("event_time"),
                outcome=fact.get("outcome"),
                fact_metadata=fact,
                episode_id=episode.id if episode else None,
                source_ts=source_ts,
                created_at=created_at,
            )
            db.add(unit)
            db.flush()
            if unit_type == "fact":
                qdrant_store.upsert(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    content=unit.content,
                    vector=vector_by_index[idx],
                    point_id=unit.id,
                    created_at=created_at,
                    unit_type="fact",
                    source_ts=source_ts,
                    source_request_id=request.request_id,
                    episode_id=unit.episode_id,
                )
        db.commit()
        return AddResponse(success=True, request_id=request.request_id, user_id=request.user_id, session_id=request.session_id)
    except Exception:
        db.rollback()
        logger.exception("Add memory failed")
        raise
    finally:
        db.close()


def search_memory(request: SearchRequest) -> SearchResponse:
    query_text = _normalize_content(request.query)
    if request.options:
        query_text += "\n" + "\n".join(request.options)
    query_vector = sf_client.embed([query_text])[0]
    recall = min(max(request.top_k, 1) * max(settings.dense_recall_multiplier, settings.episode_fact_recall_multiplier), 1000)
    candidates = qdrant_store.search(query_vector=query_vector, user_id=request.user_id, top_k=request.top_k, limit=recall, facts_only=True)
    if not candidates:
        return SearchResponse(data=[])
    latest_ts = max((c.get("source_ts") for c in candidates if c.get("source_ts") is not None), default=None)
    _normalize_dense_scores(candidates)
    plain_query = _normalize_content(request.query)
    for candidate in candidates:
        candidate["hybrid_score"] = settings.dense_weight * candidate.get("dense_norm", 0.0) + settings.keyword_weight * _keyword_score(plain_query, candidate.get("content", "")) + settings.recency_weight * _recency_score(candidate.get("source_ts"), latest_ts)
    candidates.sort(key=lambda c: c["hybrid_score"], reverse=True)
    rerank_candidates = candidates[:recall]
    rerank_results = sf_client.rerank(query=query_text, documents=[c["content"] for c in rerank_candidates], top_n=min(request.top_k * 2, len(rerank_candidates)))
    ordered = []
    if rerank_results:
        used = set()
        for result in rerank_results:
            idx = result.get("index", -1)
            if 0 <= idx < len(rerank_candidates):
                item = rerank_candidates[idx]
                item["score"] = result.get("relevance_score", item["hybrid_score"])
                ordered.append(item)
                used.add(idx)
        ordered.extend(c for i, c in enumerate(rerank_candidates) if i not in used)
    else:
        ordered = rerank_candidates
        for item in ordered:
            item["score"] = item["hybrid_score"]

    db = SessionLocal()
    try:
        unit_ids = [c["id"] for c in ordered]
        units = {u.id: u for u in db.query(MemoryUnit).filter(MemoryUnit.id.in_(unit_ids), MemoryUnit.user_id == request.user_id).all()}
        groups = {}
        for item in ordered:
            unit = units.get(item["id"])
            event_id = unit.episode_id if unit and unit.episode_id else item.get("episode_id") or f"unit:{item['id']}"
            group = groups.setdefault(event_id, {"best": item, "items": [], "score": item.get("score", 0.0)})
            group["items"].append(item)
            group["score"] = max(group["score"], item.get("score", 0.0)) + min(0.03, 0.01 * len(group["items"]))
        ranked_groups = sorted(groups.values(), key=lambda g: g["score"], reverse=True)
        data = []
        for group in ranked_groups:
            best = group["best"]
            unit = units.get(best["id"])
            context = _episode_context(db, unit.episode_id if unit else None, request.user_id)
            content = best["content"]
            if context:
                content = f"{content}\n[同一事件相关事实]\n{context}"
            data.append(MemoryItem(id=best["id"], content=content, score=group["score"], created_at=_ts_to_iso(best.get("source_ts")) or best.get("created_at")))
            if len(data) >= request.top_k:
                break
        return SearchResponse(data=data)
    finally:
        db.close()
