import json
import logging
import re
from typing import List

from app.config import settings
from app.siliconflow_client import sf_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是面向 Agent 长期记忆与 Episode Memory 构建的结构化事实提取器。\n\n"
    "你的目标不是总结对话，而是从多轮消息中忠实提取值得长期保存的事实，"
    "并保留它们所属的事件、过程状态与原始来源。\n\n"
    "事件识别规则：\n"
    "1. 首先识别输入中涉及的具体事件。\n"
    "2. 提取事件主体、要解决的问题或目标，以及明确出现的时间线索。\n"
    "3. 只有时间、地点、参与者或目标高度重合的消息才属于同一事件。\n"
    "4. 不要把主题相似但发生时间不同、目标不同或结果不同的事件混为一谈。\n\n"
    "事实提取规则：\n"
    "1. 将信息拆分为最小完整语义事实，而不是机械拆分为最短句子。\n"
    "2. 必须保留动作与其明确结果之间的绑定，特别是：失败尝试、解决原因、方案更换、"
    "否定信息、数值、时间和关键中间步骤。\n"
    "3. 每条 fact 的 text 必须脱离原始对话也能理解。\n"
    "4. 明确区分以下类型，并在 fact_type 中标注：\n"
    "   - observed: 用户实际执行或已经确认发生的行为/结果\n"
    "   - planned: 用户明确表达但尚未执行的计划/意图\n"
    "   - suggested: 助手提出的建议、方案或选项\n"
    "   - rejected: 用户明确否定、放弃或失败的内容\n"
    "   - confirmed_info: 用户明确确认的信息（不含动作）\n"
    "   - question: 用户提出的问题\n"
    "5. 不能因为助手建议了某件事，就推断用户执行了它。\n"
    "6. 不能因为先发生 A 再发生 B，就推断 A 导致 B。\n"
    "7. 单纯的疑问句不能自动转化为用户已经执行的行动或已经确认的事实。\n"
    "8. 不要提取寒暄、感谢、感叹、重复确认等无信息内容（如 'Thanks', 'Wow', 'Yeah'）。\n"
    "9. 优先使用原始消息中的具体措辞作为 evidence。\n"
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
                    "fact_type": {
                        "type": "string",
                        "enum": ["observed", "planned", "suggested", "rejected", "confirmed_info", "question"],
                    },
                    "source_id": {"type": ["string", "null"]},
                    "role": {"type": ["string", "null"], "enum": ["user", "assistant", "system", None]},
                    "event_title": {"type": ["string", "null"]},
                    "event_subject": {"type": ["string", "null"]},
                    "event_problem": {"type": ["string", "null"]},
                    "event_time": {"type": ["string", "null"]},
                    "subject": {"type": ["string", "null"]},
                    "action": {"type": ["string", "null"]},
                    "outcome": {"type": ["string", "null"]},
                },
                "required": [
                    "text",
                    "evidence",
                    "fact_type",
                    "source_id",
                    "role",
                    "event_title",
                    "event_subject",
                    "event_problem",
                    "event_time",
                    "subject",
                    "action",
                    "outcome",
                ],
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
            for key in (
                "source_id",
                "role",
                "subject",
                "action",
                "fact_type",
                "event_title",
                "event_subject",
                "event_problem",
                "event_time",
                "outcome",
            ):
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
                {"role": "user", "content": f"请从以下多轮对话中提取结构化原子事实：\n\n{messages_text}"},
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
