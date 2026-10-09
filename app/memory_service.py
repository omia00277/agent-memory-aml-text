import logging
import math
import re
from datetime import datetime, timezone
from typing import List, Optional, Union

from app.config import settings
from app.consolidator import _split_sentences, consolidate, extract_state_facts
from app.database import MemoryEdge, MemoryUnit, RawChunk, SessionLocal
from app.qdrant_store import qdrant_store
from app.relations import RelationType, llm_classify_relations, rule_classify_relation
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
# Structured fact helpers
# ---------------------------------------------------------------------------

_ENTITY_ATTR_PATTERNS = [
    # NOTE: this regex set is a *fallback* used only when the LLM did not return
    # structured fields. It deliberately matches only *state* phrasings (X lives
    # in Y). Event phrasings such as "moved to X" are excluded on purpose: a move
    # is an event, and turning it into "居住地=X" would let an event unit supersede
    # a real state unit (see _apply_supersession).
    # 居住地 - Chinese
    (r"(?:住|居住|生活在)\s*(?:在|于)?\s*([^，。,.；;！!？?]+?)(?:\s*(?:附近|旁边|区|市|省|国|里|那里))?\s*[，。,.；;！!？?]", "居住地"),
    # 居住地 - English
    (r"(?:live\s+in|lived\s+in|reside\s+in|live\s+at|located\s+in)\s+([^，。,.；;！!？?]+?)(?:\s*[，。,.；;！!？?]|$)", "居住地"),
    # 手机号 - Chinese
    (r"(?:手机|电话|联系方式)\s*(?:号码|是|为)?\s*[:：]?\s*([\d\-]{7,})", "手机号"),
    # 手机号 - English
    (r"(?:phone|mobile|cell)\s*(?:number|is)?\s*[:：]?\s*([\d\-\+\(\)\s]{7,})", "手机号"),
    # 工作 - Chinese
    (r"(?:工作|职业|职位|是一名)\s*(?:是|为|的)?\s*[:：]?\s*([^，。,.；;！!？?]+?)(?:\s*[，。,.；;！!？?])", "工作"),
    # 工作 - English
    (r"(?:work\s+as|job\s+is|works\s+as|am\s+a|is\s+a)\s+([^，。,.；;！!？?]+?)(?:\s*[，。,.；;！!？?]|$)", "工作"),
    # 年龄 - Chinese
    (r"(\d{1,3})\s*(?:岁|years?\s*old)", "年龄"),
    # 年龄 - English
    (r"(?:age|i am|i'm)\s+(\d{1,3})\s*(?:years?\s*old)?", "年龄"),
    # 名字 - Chinese
    (r"(?:叫|姓名|名字是)\s*[:：]?\s*([^，。,.；;！!？?]+?)(?:\s*[，。,.；;！!？?])", "姓名"),
    # 名字 - English
    (r"(?:my\s+name\s+is|i\s+am|i'm)\s+([^，。,.；;！!？?]+?)(?:\s*[，。,.；;！!？?]|$)", "姓名"),
]


def _extract_entities_from_text(text: str) -> List[str]:
    """Extract simple entity mentions from raw text."""
    import re
    entities = {"用户"}
    # Named places (simple heuristic)
    for m in re.finditer(r"[\u4e00-\u9fff]{2,}(?:市|省|国|区|县|镇|村)", text):
        entities.add(m.group(0))
    for m in re.finditer(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text):
        entities.add(m.group(0))
    return list(entities)


def _infer_attribute_value(fact: dict) -> dict:
    """Infer attribute/value for facts that LLM didn't structure."""
    import re
    content = fact.get("content", "")
    if fact.get("attribute") and fact.get("value"):
        return fact
    for pattern, attr in _ENTITY_ATTR_PATTERNS:
        m = re.search(pattern, content)
        if m:
            fact = dict(fact)
            fact["attribute"] = attr
            fact["value"] = m.group(1).strip()
            break
    if not fact.get("entities"):
        fact["entities"] = _extract_entities_from_text(content)
    return fact


def _normalize_fact(fact: dict, source_ts: Optional[int]) -> dict:
    """Ensure a fact dict has all expected keys and normalized values."""
    normalized = {
        "content": str(fact.get("content", "")).strip(),
        "fact_type": fact.get("fact_type") or "general_fact",
        "attribute": fact.get("attribute") or None,
        "value": fact.get("value") or None,
        "entities": fact.get("entities") or _extract_entities_from_text(fact.get("content", "")),
        "source_ts": source_ts,
    }
    if isinstance(normalized["entities"], str):
        normalized["entities"] = [normalized["entities"]]
    return normalized


# ---------------------------------------------------------------------------
# Cross-message association helpers
# ---------------------------------------------------------------------------

def _retrieve_related_units(
    db, new_fact: dict, user_id: str, vector: List[float]
) -> List[MemoryUnit]:
    """Retrieve candidate related memory units using multiple strategies."""
    attribute = new_fact.get("attribute")
    entities = new_fact.get("entities") or []

    # 1. Structured match: same attribute or overlapping entities
    structured_query = db.query(MemoryUnit).filter(MemoryUnit.user_id == user_id)
    conditions = []
    if attribute:
        conditions.append(MemoryUnit.attribute == attribute)
    if entities:
        # SQLite JSON array matching is limited; use content LIKE for entities
        for e in entities:
            conditions.append(MemoryUnit.content.ilike(f"%{e}%"))
    if conditions:
        from sqlalchemy import or_
        structured_query = structured_query.filter(or_(*conditions))
    structured_query = structured_query.order_by(MemoryUnit.source_ts.desc()).limit(settings.structured_recall_limit)
    structured = structured_query.all()

    # 2. Dense semantic retrieval
    dense = qdrant_store.search(
        query_vector=vector,
        user_id=user_id,
        top_k=settings.dense_recall_top_k,
    )

    # 3. Entity keyword retrieval
    keyword = []
    if entities:
        keyword = qdrant_store.search_by_entities(
            user_id=user_id,
            entities=entities,
            query_vector=vector,
            top_k=settings.keyword_recall_top_k,
        )

    # Merge and deduplicate by unit id
    seen_ids = set()
    merged: List[MemoryUnit] = []

    # Structured matches are highest priority
    for unit in structured:
        if unit.id not in seen_ids:
            seen_ids.add(unit.id)
            merged.append(unit)

    # Then dense and keyword by original order (already ranked)
    for cand in dense + keyword:
        uid = cand.get("id")
        if uid and uid not in seen_ids:
            unit = db.query(MemoryUnit).filter(MemoryUnit.id == uid).first()
            if unit:
                seen_ids.add(uid)
                merged.append(unit)

    return structured, merged


def _select_candidates_for_classification(
    structured: List[MemoryUnit],
    merged: List[MemoryUnit],
) -> List[MemoryUnit]:
    """Select final candidates for relation classification.

    Structured matches are always included; remaining slots are filled from the
    merged ordered list up to related_candidates_max.
    """
    structured_ids = {u.id for u in structured}
    must = [u for u in merged if u.id in structured_ids]
    others = [u for u in merged if u.id not in structured_ids]
    max_total = settings.related_candidates_max
    remaining_slots = max(0, max_total - len(must))
    return must + others[:remaining_slots]


def _create_edges(db, new_unit: MemoryUnit, relations: List[dict]):
    """Persist classified relations as memory edges."""
    for rel in relations:
        edge = MemoryEdge(
            source_unit_id=new_unit.id,
            target_unit_id=rel["target_unit_id"],
            relation_type=rel["relation_type"],
            confidence=rel.get("confidence", 70),
        )
        db.add(edge)


def _value_skeleton(content: str, value: Optional[str]) -> str:
    """Blank out the attribute value so two updates of the same slot compare equal.

    "用户居住在北京" (value 北京) and "用户居住在上海" (value 上海) both become
    "用户居住在§". If the value is only a paraphrase and not literally present in
    the content, the content is returned unchanged, which keeps the guard strict.
    """
    if not value:
        return content or ""
    text = content or ""
    v = str(value).strip()
    if v and v in text:
        return text.replace(v, "§")
    return text


def _apply_supersession(db, new_unit: MemoryUnit) -> List[str]:
    """Mark older same-attribute units as superseded and mirror it into Qdrant.

    Supersession is decided purely from the structured fields of the units, never
    from the LLM relation direction: an *event* unit such as "用户上个月搬到上海"
    must not be allowed to supersede the *state* unit it gives rise to. A unit can
    only supersede another when it is itself a state fact (attribute + value), the
    older unit holds a different value for the same attribute, and the two facts
    are structurally the same apart from that value.

    The old units are kept (append-only audit trail) but flagged so that Search
    can softly demote them. Returns the list of superseded unit ids.
    """
    attribute = new_unit.attribute
    value = new_unit.value
    if not attribute or value is None or new_unit.unit_type != "fact":
        return []

    candidates = (
        db.query(MemoryUnit)
        .filter(MemoryUnit.user_id == new_unit.user_id)
        .filter(MemoryUnit.attribute == attribute)
        .filter(MemoryUnit.id != new_unit.id)
        .filter(MemoryUnit.superseded_by.is_(None))
        .all()
    )

    superseded_ids: List[str] = []
    new_value_norm = str(value).strip().lower()
    new_skeleton = _value_skeleton(new_unit.content, new_unit.value)
    for old in candidates:
        if old.value is None:
            continue
        if str(old.value).strip().lower() == new_value_norm:
            continue  # same value: a duplicate, not an update
        # Structural guard: only true "same slot, new value" pairs qualify. This
        # rejects loose LLM attribute labels such as 情感价值 shared by two
        # unrelated objects (a necklace and a bowl). Raw sentences are exempt:
        # their attributes come from the precise regex fallback, and their wording
        # legitimately differs from the consolidated fact (including cross-language).
        if old.unit_type != "raw":
            old_skeleton = _value_skeleton(old.content, old.value)
            if (
                _text_similarity(old_skeleton, new_skeleton)
                < settings.supersession_min_skeleton_similarity
            ):
                continue
        # Never let an older fact overwrite a newer one (out-of-order Add).
        if (
            new_unit.source_ts is not None
            and old.source_ts is not None
            and new_unit.source_ts < old.source_ts
        ):
            continue
        old.superseded_by = new_unit.id
        db.add(old)
        superseded_ids.append(old.id)

    # Mirror the flag into Qdrant payload so Search sees it without a DB join.
    for unit_id in superseded_ids:
        try:
            qdrant_store.mark_superseded(unit_id)
        except Exception as e:  # pragma: no cover - payload update is best-effort
            logger.warning(f"Failed to mark {unit_id} superseded in Qdrant: {e}")
    return superseded_ids


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
            consolidated = [{"content": text}]

        # Normalize structured facts and infer missing attribute/value/entities.
        fact_dicts = [_normalize_fact(f, source_ts) for f in consolidated]

        # State tracking uses its own narrow LLM call rather than riding on the
        # consolidation prompt: mixing them made the model drop the new state fact
        # whenever the chunk also contained another state (reproducible at temp 0).
        state_facts = [
            _normalize_fact(f, source_ts) for f in extract_state_facts(text)
        ]
        for sf_fact in state_facts:
            sf_fact["fact_type"] = "personal_state"
        fact_dicts = fact_dicts + state_facts

        # Regex fallback fills in attribute/value for facts the LLM left unstructured.
        fact_dicts = [_infer_attribute_value(f) for f in fact_dicts]

        # Add raw sentences as fallback units to preserve original details
        # that LLM consolidation may drop or over-generalize.
        raw_sentence_dicts = []
        if settings.enable_raw_sentence_fallback:
            raw_candidates = []
            consolidated_contents = [f.get("content", "") for f in fact_dicts]
            for s in _split_sentences(text):
                # Skip conversational fillers and overly short sentences
                if not _is_meaningful_raw_sentence(s):
                    continue
                # Skip raw sentences already subsumed by a consolidated fact
                if any(s in fact or fact in s for fact in consolidated_contents):
                    continue
                # Skip near-duplicate of any consolidated fact
                if consolidated_contents and any(
                    _is_near_duplicate(s, fact, settings.search_dedup_threshold)
                    for fact in consolidated_contents
                ):
                    continue
                score = _score_raw_sentence(s, consolidated_contents)
                if score >= settings.raw_fallback_min_score:
                    raw_candidates.append((score, s))

            # Sort by informativeness and keep only the top N per chunk
            raw_candidates.sort(key=lambda x: x[0], reverse=True)
            # Raw sentences get the same attribute fallback as facts: a raw copy of
            # an outdated value ("I live in Beijing.") must be superseded too,
            # otherwise it stays unpenalised and can outrank the new state.
            raw_sentence_dicts = [
                _infer_attribute_value(
                    _normalize_fact({"content": s, "fact_type": "raw"}, source_ts)
                )
                for _, s in raw_candidates[: settings.raw_fallback_per_chunk]
            ]

        # Preserve order: facts first, raw fallbacks after; deduplicate exact strings
        seen = set()
        all_units: List[dict] = []
        for fact in fact_dicts + raw_sentence_dicts:
            key = fact["content"].strip()
            if key and key not in seen:
                seen.add(key)
                fact["unit_type"] = "fact" if fact in fact_dicts else "raw"
                all_units.append(fact)

        unit_contents = [u["content"] for u in all_units]
        vectors = sf_client.embed(unit_contents)

        created_units: List[MemoryUnit] = []
        for idx, fact in enumerate(all_units):
            unit = MemoryUnit(
                user_id=request.user_id,
                session_id=request.session_id,
                source_request_id=request.request_id,
                content=fact["content"],
                unit_type=fact["unit_type"],
                source_ts=fact.get("source_ts") or source_ts,
                created_at=created_at,
                fact_type=fact.get("fact_type"),
                attribute=fact.get("attribute"),
                value=fact.get("value"),
                entities=fact.get("entities"),
            )
            db.add(unit)
            db.flush()
            created_units.append(unit)
            qdrant_store.upsert(
                user_id=request.user_id,
                session_id=request.session_id,
                content=fact["content"],
                vector=vectors[idx],
                point_id=unit.id,
                created_at=created_at,
                unit_type=fact["unit_type"],
                source_ts=fact.get("source_ts") or source_ts,
                source_request_id=request.request_id,
                fact_type=fact.get("fact_type"),
                attribute=fact.get("attribute"),
                value=fact.get("value"),
                entities=fact.get("entities"),
            )

        # Cross-message association: link new units to existing memory.
        for idx, (unit, fact) in enumerate(zip(created_units, all_units)):
            if unit.unit_type == "raw":
                # Raw sentences are less reliable for structured relations;
                # they stay searchable but do not drive the association graph.
                continue

            structured, merged = _retrieve_related_units(
                db, fact, request.user_id, vectors[idx]
            )
            candidates = _select_candidates_for_classification(structured, merged)

            # Rule-based classification first; ambiguous cases go to the LLM.
            relations = []
            ambiguous = []
            for cand in candidates:
                if cand.id == unit.id:
                    continue
                rule_result = rule_classify_relation(fact, cand)
                if rule_result:
                    rel_type, conf = rule_result
                    relations.append({
                        "target_unit_id": cand.id,
                        "relation_type": rel_type,
                        "confidence": conf,
                    })
                else:
                    ambiguous.append(cand)

            if ambiguous and settings.enable_relation_classification:
                relations.extend(llm_classify_relations(unit.content, ambiguous))

            _create_edges(db, unit, relations)

            # Direction A: supersession is derived from the structured fields, not
            # from the LLM relation direction. Old facts are kept but softly demoted,
            # while the new fact is a real, first-class memory unit Search can retrieve.
            _apply_supersession(db, unit)

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
        # Direction A: superseded facts are kept for audit but softly demoted so
        # the newer, still-valid fact wins without any query-side attribute mapping.
        if c.get("superseded"):
            base_score *= settings.superseded_score_penalty
        c["hybrid_score"] = base_score

    candidates.sort(key=lambda c: c["hybrid_score"], reverse=True)
    recall_candidates = candidates[: recall]

    documents = [c["content"] for c in recall_candidates]
    rerank_results = sf_client.rerank(
        query=query_text,
        documents=documents,
        top_n=request.top_k,
    )

    if rerank_results:
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

    # Apply the demotion after reranking too, since the cross-encoder does not
    # know about supersession and could otherwise lift a stale fact back up.
    for c in ordered:
        if c.get("superseded"):
            c["score"] = c.get("score", 0.0) * settings.superseded_score_penalty
    ordered.sort(key=lambda c: c.get("score", 0.0), reverse=True)

    # Deduplicate near-duplicate contents before returning top_k.
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
