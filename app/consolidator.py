import json
import logging
import re
from typing import List

from app.config import settings
from app.siliconflow_client import sf_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是长期记忆归档助手。把输入的多轮对话压缩为原子化事实。"
    "每条事实必须包含 text 和 evidence；text 必须自包含，evidence 必须来自输入原文。"
    "尽量提取 source_id、role、subject、question、action。role 只能是 user、assistant、system 或 null。"
)

FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence": {"type": "string"},
                    "source_id": {"type": ["string", "null"]},
                    "role": {"type": ["string", "null"], "enum": ["user", "assistant", "system", None]},
                    "subject": {"type": ["string", "null"]},
                    "question": {"type": ["string", "null"]},
                    "action": {"type": ["string", "null"]},
                },
                "required": ["text", "evidence", "source_id", "role", "subject", "question", "action"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}


def _split_sentences(text: str) -> List[dict]:
    units = []
    for line_no, part in enumerate(re.split(r"\n+", text.strip())):
        part = part.strip()
        if not part:
            continue
        match = re.match(r"^(user|assistant|system)\s*:\s*(.*)$", part, flags=re.IGNORECASE)
        role = match.group(1).lower() if match else None
        body = match.group(2).strip() if match else part
        for sentence in re.split(r"[。！？!?；;]+", body):
            sentence = sentence.strip()
            if sentence and len(sentence) >= 2:
                units.append({"text": sentence, "evidence": sentence, "source_id": f"line-{line_no}", "role": role})
    return units


def _parse_llm_output(raw) -> List[dict]:
    if isinstance(raw, dict):
        data = raw.get("facts", [])
    else:
        if not raw:
            return []
        text = str(raw).strip()
        fence = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, flags=re.DOTALL)
        if fence:
            text = fence.group(1)
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            start_candidates = [p for p in (text.find("["), text.find("{")) if p >= 0]
            if not start_candidates:
                return []
            try:
                decoded = json.loads(text[min(start_candidates) : max(text.rfind("]"), text.rfind("}")) + 1])
            except json.JSONDecodeError:
                return []
        data = decoded.get("facts", []) if isinstance(decoded, dict) else decoded
    if not isinstance(data, list):
        return []
    result = []
    for item in data:
        if isinstance(item, str) and item.strip():
            result.append({"text": item.strip(), "evidence": item.strip()})
        elif isinstance(item, dict) and str(item.get("text") or item.get("fact") or "").strip():
            legacy_text = item.get("text") or item.get("fact")
            fact = {"text": str(legacy_text).strip(), "evidence": str(item.get("evidence") or legacy_text).strip()}
            for key in ("source_id", "role", "subject", "question", "action"):
                if item.get(key) is not None and str(item[key]).strip():
                    fact[key] = str(item[key]).strip()
            result.append(fact)
    return result


def consolidate(messages_text: str) -> List[dict]:
    if not settings.enable_llm_consolidation:
        return _split_sentences(messages_text)
    try:
        output = sf_client.chat_structured(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请提取以下对话的原子事实：\n\n{messages_text}"},
            ],
            "atomic_facts",
            FACT_SCHEMA,
            max_tokens=settings.llm_consolidation_max_tokens,
        )
        units = _parse_llm_output(output)
        if units:
            return units
        logger.warning("Structured fact extraction returned no facts; using fallback.")
    except Exception as e:
        logger.warning(f"Structured fact extraction failed ({e}); using fallback.")
    return _split_sentences(messages_text)


def fact_text(fact: dict) -> str:
    return str(fact.get("text", "")).strip()
