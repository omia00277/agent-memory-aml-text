import json
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


# Patterns to detect temporal / sequential information in raw sentences.
_TEMPORAL_PATTERNS = [
    r"\d{4}年", r"\d{1,2}月", r"\d{1,2}日",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b",
    r"\b(?:last|next|this|every|each)\s+(?:week|month|year|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|morning|afternoon|evening|night|day|weekend)\b",
    r"\b\d+\s+(?:minutes?|hours?|days?|weeks?|months?|years?)\s+(?:ago|later|after|before)\b",
    r"\b(?:first|then|after\s+that|before|later|earlier|finally|next|previously)\b",
    r"\b(?:when|while|during|until|since)\b",
]


def _contains_temporal_info(text: str) -> bool:
    """Check whether a sentence contains explicit temporal or sequential information."""
    return any(re.search(p, text, flags=re.IGNORECASE) for p in _TEMPORAL_PATTERNS)


def _score_raw_sentence(sentence: str, consolidated: List[str]) -> float:
    """Score a raw sentence by informativeness; higher is better."""
    s = sentence.strip()
    if not s:
        return 0.0

    # 1. Length score: prefer 15~80 chars, penalize too short or too long
    length = len(s)
    length_score = max(0.0, 1.0 - abs(length - 45) / 80)

    # 2. Entity density: more named/semantic entities is better
    cjk_entities = len(re.findall(r"[\u4e00-\u9fff]{2,}", s))
    ascii_entities = len(re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", s))
    entity_score = min(1.0, (cjk_entities + ascii_entities) / 5)

    # 3. Novelty vs consolidated facts: lower overlap with facts means more new info
    if consolidated:
        max_overlap = max(_text_similarity(s, fact) for fact in consolidated)
        novelty_score = 1.0 - max_overlap
    else:
        novelty_score = 1.0

    # 4. Content-word ratio: more nouns/verbs/adjectives vs stopwords
    content_chars = len(re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]{2,}", s))
    content_ratio = content_chars / max(length, 1)
    content_score = min(1.0, content_ratio * 2.5)

    # 5. Temporal boost: prefer sentences that carry time/sequence info
    temporal_boost = settings.temporal_priority_boost if _contains_temporal_info(s) else 0.0

    score = 0.25 * length_score + 0.30 * entity_score + 0.25 * novelty_score + 0.20 * content_score + temporal_boost
    return score


def _build_query_text(query: Union[str, List[dict]], options: Optional[List[str]]) -> str:
    q = _normalize_content(query)
    if options:
        q = q + "\n" + "\n".join(options)
    return q


def _text_similarity(a: str, b: str) -> float:
    """Compute token overlap ratio for near-duplicate detection."""
    a_tokens = set(_tokens(a))
    b_tokens = set(_tokens(b))
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(intersection) / len(union)


def _is_near_duplicate(a: str, b: str, threshold: float) -> bool:
    """Check if two memory contents are near-duplicates."""
    a_norm = a.strip().lower()
    b_norm = b.strip().lower()
    if not a_norm or not b_norm:
        return True
    # Exact or substring containment
    if a_norm == b_norm or a_norm in b_norm or b_norm in a_norm:
        return True
    # High token overlap
    return _text_similarity(a_norm, b_norm) >= threshold


def _deduplicate_results(
    candidates: List[dict], threshold: Optional[float] = None
) -> List[dict]:
    """Remove near-duplicate memory contents, keeping the higher-scored one."""
    threshold = threshold if threshold is not None else settings.search_dedup_threshold
    selected: List[dict] = []
    for c in candidates:
        content = c.get("content", "")
        if any(_is_near_duplicate(content, s.get("content", ""), threshold) for s in selected):
            continue
        selected.append(c)
    return selected


# ---------------------------------------------------------------------------
# Entity-Attribute-Value (EAV) extraction and conflict-resolution helpers
# ---------------------------------------------------------------------------

# Common attribute synonyms for normalization.
_ATTRIBUTE_SYNONYMS = {
    "居住地": ["住在哪里", "住在哪儿", "住址", "居住城市", "住的城市", "住的地方"],
    "手机号": ["手机号码", "电话", "联系方式", "联系电话"],
    "工作": ["职业", "工作职位", "职位", "做什么工作", "工作单位", "工作城市"],
    "邮箱": ["电子邮箱", "邮件地址", "email"],
    "姓名": ["名字", "叫什么"],
    "年龄": ["多大", "几岁"],
    "状态": ["情况", "现状"],
}


def _normalize_attribute(attr: str) -> str:
    attr = str(attr).lower().strip()
    for canon, syns in _ATTRIBUTE_SYNONYMS.items():
        if attr == canon.lower() or attr in [s.lower() for s in syns]:
            return canon
    return attr


def _is_same_attribute(attr1: str, attr2: str) -> bool:
    n1 = _normalize_attribute(attr1)
    n2 = _normalize_attribute(attr2)
    if n1 == n2:
        return True
    # Fallback: token overlap
    return _text_similarity(n1, n2) >= 0.80


def _is_same_value(v1: str, v2: str) -> bool:
    s1 = str(v1).lower().strip()
    s2 = str(v2).lower().strip()
    if not s1 or not s2:
        return False
    if s1 == s2 or s1 in s2 or s2 in s1:
        return True
    return _text_similarity(s1, s2) >= 0.85


def _rule_based_eav(fact: str) -> Optional[dict]:
    """Fallback EAV extraction using simple patterns when LLM is unavailable."""
    text = fact.strip()
    if not text:
        return None

    entity = "用户"
    attribute = "事实"
    value = text

    # Common patterns
    # 居住
    m = re.search(r"(?:住|居住|搬到|生活在|工作在)\s*(?:在|到)?\s*([^，。,.]+?)(?:\s*(?:附近|旁边|区|市|省|国|里|那里))?\s*[，。,.]", text)
    if m:
        attribute = "居住地"
        value = m.group(1).strip()
        return {"entity": entity, "attribute": attribute, "value": value}

    # 手机号
    m = re.search(r"(?:手机|电话|联系方式)\s*(?:号码|是|为)?\s*[:：]?\s*([\d\-]{7,})", text)
    if m:
        attribute = "手机号"
        value = m.group(1).strip()
        return {"entity": entity, "attribute": attribute, "value": value}

    # 工作/职业
    m = re.search(r"(?:工作|职业|职位|是一名|做)\s*(?:是|为|的)?\s*[:：]?\s*([^，。,.]+?)(?:\s*[，。,.])", text)
    if m:
        attribute = "工作"
        value = m.group(1).strip()
        return {"entity": entity, "attribute": attribute, "value": value}

    # 年龄
    m = re.search(r"(\d{1,3})\s*(?:岁|years?\s*old)", text)
    if m:
        attribute = "年龄"
        value = m.group(1).strip() + "岁"
        return {"entity": entity, "attribute": attribute, "value": value}

    # 名字
    m = re.search(r"(?:叫|姓名|名字是)\s*[:：]?\s*([^，。,.]+?)(?:\s*[，。,.])", text)
    if m:
        attribute = "姓名"
        value = m.group(1).strip()
        return {"entity": entity, "attribute": attribute, "value": value}

    # 毕业/学位
    m = re.search(r"(?:毕业|获得|取得)\s*(?:了|有)?\s*([^，。,.]*?(?:学位|学历|证书))", text)
    if m:
        attribute = "学历"
        value = m.group(1).strip()
        return {"entity": entity, "attribute": attribute, "value": value}

    return {"entity": entity, "attribute": attribute, "value": value}


_EAV_SYSTEM_PROMPT = """You are a memory structuring assistant. Given a list of factual statements, extract the entity, attribute, and value for each.

Rules:
1. Entity: the person, object, or concept being described. Use "用户" if the subject is the user.
2. Attribute: the property or characteristic being asserted (e.g., 居住地, 手机号, 工作).
3. Value: the specific content of that attribute.
4. If a statement is a general fact with no clear attribute, use attribute "事实" and value as the whole statement.
5. Normalize entity to be concise (max 4 words).

Output strictly as JSON:
{
  "extractions": [
    {"entity": "...", "attribute": "...", "value": "..."}
  ]
}"""


def _extract_eav(facts: List[str]) -> List[Optional[dict]]:
    """Extract entity/attribute/value for each fact. Falls back to rules if LLM fails or disabled."""
    if not facts:
        return []

    # Try LLM first if enabled
    if settings.enable_llm_eav_extraction:
        try:
            prompt = "Extract from these facts:\n" + "\n".join(f"- {f}" for f in facts)
            response = sf_client.chat(
                messages=[
                    {"role": "system", "content": _EAV_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            data = json.loads(response)
            extractions = data.get("extractions", [])
            if len(extractions) == len(facts):
                return extractions
        except Exception as e:
            logger.debug(f"LLM EAV extraction failed: {e}")

    # Fallback to rule-based extraction per fact
    return [_rule_based_eav(f) for f in facts]


def _find_overridden_units(db, user_id: str, extractions: List[Optional[dict]]) -> List[MemoryUnit]:
    """Find existing valid units that are overridden by the new extractions."""
    overridden = []
    for eav in extractions:
        if not eav:
            continue
        entity = eav.get("entity")
        attribute = eav.get("attribute")
        value = eav.get("value")
        if not entity or not attribute or value is None:
            continue

        candidates = (
            db.query(MemoryUnit)
            .filter(
                MemoryUnit.user_id == user_id,
                MemoryUnit.valid == 1,
                MemoryUnit.entity == entity,
            )
            .all()
        )
        for unit in candidates:
            if unit.attribute and _is_same_attribute(unit.attribute, attribute):
                if unit.value and not _is_same_value(unit.value, value):
                    overridden.append(unit)
    return overridden


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
            raw_candidates = []
            for s in _split_sentences(text):
                # Skip conversational fillers and overly short sentences
                if not _is_meaningful_raw_sentence(s):
                    continue
                # Skip raw sentences already subsumed by a consolidated fact
                if any(s in fact or fact in s for fact in consolidated):
                    continue
                # Skip near-duplicate of any consolidated fact
                if consolidated and any(
                    _is_near_duplicate(s, fact, settings.search_dedup_threshold)
                    for fact in consolidated
                ):
                    continue
                score = _score_raw_sentence(s, consolidated)
                if score >= settings.raw_fallback_min_score:
                    raw_candidates.append((score, s))

            # Sort by informativeness and keep only the top N per chunk
            raw_candidates.sort(key=lambda x: x[0], reverse=True)
            raw_sentences = [s for _, s in raw_candidates[: settings.raw_fallback_per_chunk]]

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

        # Extract entity-attribute-value for consolidated facts only
        eav_list = _extract_eav(unit_contents)

        # Find and mark overridden units before inserting new ones
        overridden_units = _find_overridden_units(db, request.user_id, eav_list)
        for old_unit in overridden_units:
            old_unit.valid = 0
            db.add(old_unit)
            # Update Qdrant payload to reflect invalid status
            # (will be updated when we upsert the new unit, but re-upsert old point for safety)
            # Note: Qdrant does not support partial payload update via upsert? It does: upsert replaces point.
            # We'll re-upsert the old point with valid=False.
        db.flush()

        vectors = sf_client.embed(unit_contents)

        for idx, content in enumerate(unit_contents):
            eav = eav_list[idx] if idx < len(eav_list) else None
            unit = MemoryUnit(
                user_id=request.user_id,
                session_id=request.session_id,
                source_request_id=request.request_id,
                content=content,
                unit_type=unit_types[idx],
                source_ts=source_ts,
                created_at=created_at,
                entity=eav.get("entity") if eav else None,
                attribute=eav.get("attribute") if eav else None,
                value=eav.get("value") if eav else None,
                valid=1,
            )
            db.add(unit)
            db.flush()

            # Link overridden units to this new unit
            for old_unit in overridden_units:
                if eav and old_unit.attribute and _is_same_attribute(old_unit.attribute, eav.get("attribute", "")):
                    old_unit.superseded_by = unit.id

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
                entity=eav.get("entity") if eav else None,
                attribute=eav.get("attribute") if eav else None,
                value=eav.get("value") if eav else None,
                valid=True,
            )

        # Re-upsert overridden points with valid=False so Search can filter them
        for old_unit in overridden_units:
            old_vec = qdrant_store.client.retrieve(
                collection_name=qdrant_store.collection,
                ids=[old_unit.id],
                with_vectors=True,
            )
            if old_vec:
                qdrant_store.upsert(
                    user_id=old_unit.user_id,
                    session_id=old_unit.session_id,
                    content=old_unit.content,
                    vector=old_vec[0].vector,
                    point_id=old_unit.id,
                    created_at=old_unit.created_at,
                    unit_type=old_unit.unit_type,
                    source_ts=old_unit.source_ts,
                    source_request_id=old_unit.source_request_id,
                    entity=old_unit.entity,
                    attribute=old_unit.attribute,
                    value=old_unit.value,
                    valid=False,
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
        base_score = (
            settings.dense_weight * c.get("dense_norm", 0.0)
            + settings.keyword_weight * kw
            + settings.recency_weight * rec
        )
        # Validity penalty: superseded facts are heavily penalized so current-state facts rank higher.
        valid = c.get("valid", True)
        validity_multiplier = 1.0 if valid else 0.3
        c["hybrid_score"] = base_score * validity_multiplier

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

    # Apply post-rerank validity penalty so superseded facts drop below current facts.
    for c in ordered:
        valid = c.get("valid", True)
        if not valid:
            c["score"] = c.get("score", 0.0) * 0.2

    # Re-sort after validity penalty
    ordered.sort(key=lambda c: c.get("score", 0.0), reverse=True)

    # Deduplicate near-duplicate contents before returning top_k.
    # ordered is already ranked by relevance, so we keep the first (best) occurrence.
    deduped = _deduplicate_results(ordered)
    final = deduped[: request.top_k]

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
