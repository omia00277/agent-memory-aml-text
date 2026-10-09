import json
import logging
from typing import List, Optional

from app.config import settings
from app.siliconflow_client import sf_client

logger = logging.getLogger(__name__)


class RelationType:
    EQUIVALENT = "equivalent"
    UPDATES = "updates"
    CONTRADICTS = "contradicts"
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    ELABORATES = "elaborates"
    TEMPORAL_BEFORE = "temporal_before"
    TEMPORAL_AFTER = "temporal_after"
    RELATED = "related"


_CAUSAL_WORDS = {
    "because", "since", "as", "therefore", "so", "thus", "hence", "due to",
    "因为", "所以", "由于", "导致", "使得", "因此",
}

_UPDATE_WORDS = {
    "moved", "moved to", "relocated", "changed", "updated", "switched",
    "changed my", "updated my", "new", "now", "currently",
    "搬到", "搬到", "换了", "改成", "更新", "现在", "目前",
}


def _tokens(text: str) -> List[str]:
    import re
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


def _text_similarity(a: str, b: str) -> float:
    a_tokens = set(_tokens(a))
    b_tokens = set(_tokens(b))
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(intersection) / len(union)


def _value_contains(new_value: str, old_value: str) -> bool:
    """Check whether one value semantically contains the other (elaboration)."""
    if not new_value or not old_value:
        return False
    nv = str(new_value).lower().strip()
    ov = str(old_value).lower().strip()
    return nv in ov or ov in nv or _text_similarity(nv, ov) >= 0.85


def _same_attribute(attr1: Optional[str], attr2: Optional[str]) -> bool:
    if not attr1 or not attr2:
        return False
    return attr1.lower().strip() == attr2.lower().strip()


def _same_entities(entities1: Optional[List[str]], entities2: Optional[List[str]]) -> bool:
    if not entities1 or not entities2:
        return False
    e1 = {e.lower() for e in entities1}
    e2 = {e.lower() for e in entities2}
    return bool(e1 & e2)


def _has_update_words(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _UPDATE_WORDS)


def _has_causal_words(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in _CAUSAL_WORDS)


def rule_classify_relation(
    new_fact: dict,
    old_unit,
    dense_score: float = 0.0,
) -> Optional[tuple]:
    """Rule-based relation classifier. Returns (relation_type, confidence) or None."""
    new_content = str(new_fact.get("content", "")).strip()
    old_content = str(old_unit.content or "").strip()
    new_attr = new_fact.get("attribute") or ""
    old_attr = getattr(old_unit, "attribute", None) or ""
    new_value = new_fact.get("value") or ""
    old_value = getattr(old_unit, "value", None) or ""
    new_entities = new_fact.get("entities") or []
    old_entities = getattr(old_unit, "entities", None) or []
    new_ts = new_fact.get("source_ts")
    old_ts = getattr(old_unit, "source_ts", None)

    # 1. Equivalent: high textual similarity or identical value
    if _text_similarity(new_content, old_content) >= settings.equivalent_similarity_threshold:
        return (RelationType.EQUIVALENT, 95)
    if new_value and old_value and _text_similarity(str(new_value), str(old_value)) >= 0.90:
        return (RelationType.EQUIVALENT, 90)

    # 2. Updates: same attribute, different value, new fact is newer
    if _same_attribute(new_attr, old_attr) and new_value and old_value:
        if new_value != old_value:
            # Only mark as updates if new fact is chronologically newer
            if new_ts is not None and old_ts is not None:
                if new_ts >= old_ts:
                    return (RelationType.UPDATES, 90)
                # else: old is newer; old will update new when it was added (or already did)
            # No timestamps but update words present in new fact
            elif _has_update_words(new_content):
                return (RelationType.UPDATES, 75)

    # 3. Elaborates: same attribute and value containment
    if _same_attribute(new_attr, old_attr) and new_value and old_value:
        if _value_contains(new_value, old_value):
            return (RelationType.ELABORATES, 80)

    # 4. Causal: explicit causal words and shared entities
    if _has_causal_words(new_content) and _same_entities(new_entities, old_entities):
        if _has_causal_words(old_content):
            # Ambiguous; skip rule-based
            return None
        return (RelationType.CAUSES, 65)

    # 5. Strong entity overlap + moderate semantic similarity => related
    if _same_entities(new_entities, old_entities) and dense_score >= 0.55:
        return (RelationType.RELATED, 60)

    return None


_RELATION_PROMPT = """You are analyzing relationships between a new memory fact and existing memory facts.

For each pair, classify the relationship as one of:
- equivalent: same information, possibly paraphrased
- updates: new fact updates/replaces old fact (same attribute, new value)
- contradicts: directly conflicts with old fact
- causes: new fact explains or causes old fact
- caused_by: new fact is a result of old fact
- elaborates: new fact adds detail to old fact
- temporal_before: new fact happened before old fact
- temporal_after: new fact happened after old fact
- related: related but not any of the above
- unrelated: no clear relationship

Output strictly as JSON:
{
  "relations": [
    {"old": "...", "relation": "...", "confidence": 0.85, "reason": "..."}
  ]
}"""


def _format_pairs(new_content: str, old_contents: List[str]) -> str:
    lines = [f"New fact: {new_content}", "Existing facts:"]
    for idx, old in enumerate(old_contents, 1):
        lines.append(f"{idx}. {old}")
    return "\n".join(lines)


def llm_classify_relations(
    new_content: str,
    candidates: List,
    batch_size: int = 10,
) -> List[dict]:
    """Classify relationships between a new fact and candidate old facts using LLM.

    Returns a list of dicts with keys: target_unit_id, relation_type, confidence.
    """
    if not settings.enable_relation_classification or not candidates:
        return []

    results = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        old_contents = [c.content for c in batch]
        prompt = _format_pairs(new_content, old_contents)
        try:
            raw = sf_client.chat(
                [
                    {"role": "system", "content": _RELATION_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            data = json.loads(raw)
            relations = data.get("relations", [])
            for rel, cand in zip(relations, batch):
                relation_type = rel.get("relation", "unrelated")
                confidence = int(float(rel.get("confidence", 0)) * 100)
                if relation_type == "unrelated":
                    continue
                if confidence < settings.relation_confidence_threshold:
                    continue
                results.append({
                    "target_unit_id": cand.id,
                    "relation_type": relation_type,
                    "confidence": confidence,
                })
        except Exception as e:
            logger.warning(f"LLM relation classification failed: {e}")
            continue
    return results
