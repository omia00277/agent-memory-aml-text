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
    "1、记忆价值判断\n"
    "提取每条事实之前，先判断它是否提供了可以在未来被检索、验证或用于理解历史事件的信息。\n"
    "优先提取：\n"
    "- 人物、组织、项目、对象的具体信息和关系。\n"
    "- 用户的明确偏好、需求、目标和约束。\n"
    "- 具体发生过的事件、行动、经历和结果。\n"
    "- 已做出的决策、未执行的计划和状态变更。\n"
    "- 失败尝试、错误原因、问题诊断和解决过程。\n"
    "- 时间、地点、数值、版本、名称及关键中间步骤。\n"
    "- 对具体事件或方案有明确指向的评价、意见和选择。\n\n"
    "2、噪声过滤规则（严格执行）\n"
    "以下内容不应单独提取为原子事实：\n"
    "- 问候与告别，例如 Hello、Good morning、Bye。\n"
    "- 感谢与礼貌表达，例如 Thanks, Caroline、Thank you。\n"
    "- 无具体信息的感叹，例如 Wow, Caroline、That's awesome。\n"
    "- 祝福与鼓励，例如 Good luck、Congratulations。\n"
    "- 无实际信息增量的附和，例如 Sounds good、Cool、Nice。\n"
    "- 通用过渡语、寒暄、客套话和重复的社交回应。\n"
    "- 助手重复用户已经陈述的事实，且没有引入新信息的内容。\n\n"
    "不要因为一句话出现了人名、明确的说话者或完整的语法结构，就认为它值得记忆。\n"
    "不要把普通的感谢、夸奖、祝福转写成\"某人感谢了某人\"\"某人认为事情很好\"等低价值记忆。\n\n"
    "3、事件识别规则\n"
    "首先识别输入中涉及的具体事件，提取事件主体、要解决的问题或目标，以及明确出现的时间线索。\n\n"
    "4、事实提取规则\n"
    "将信息拆分为最小完整语义事实，而不是机械拆分为最短句子。\n"
    "必须保留动作与其明确结果之间的绑定，特别是失败尝试、解决原因、方案更换、"
    "否定信息、数值、时间和关键中间步骤。\n"
    "每条 Fact 的 text 必须脱离原始对话也能理解。\n"
    "明确区分用户实际执行的行为、助手提出的建议、尚未执行的计划、假设、已确认的信息及真实结果。\n"
    "不能因为助手建议了某件事，就推断用户执行了它。不能因为先发生 A 再发生 B，就推断 A 导致 B。\n"
    "单纯的疑问句不能自动转化为用户已经执行的行动或已经确认的事实。\n\n"
    "5、输出格式要求\n"
    "你的唯一输出必须是一个合法的 JSON 对象，其顶层结构为 {\"facts\": [...]}，"
    "并严格符合给定的 JSON Schema：每条 fact 的必填字段都必须给出（没有对应信息时填 null）。\n"
    "以纯 JSON 文本输出：不要输出 Markdown 代码围栏（```json ... ```），"
    "不要输出任何前言、注释、说明或后语，不要输出 schema 之外的其他字段。\n"
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
        # LLM 正常返回但判定本块没有值得记忆的事实：尊重其过滤结果，
        # 不做句子切分兜底（否则问候/寒暄等噪声会被重新拆成"事实"）。
        logger.warning("Structured fact extraction returned no facts; skipping sentence-split fallback.")
        return []
    except Exception as e:
        # LLM 调用/内容异常（网络、超时、401、JSON 解析失败等）：
        # 句子切分兜底仍然可用，保证写入不因模型故障而完全缺失。
        logger.warning(f"Structured fact extraction failed ({e}); using sentence-split fallback.")
        return _split_sentences(messages_text)


def fact_text(fact: dict) -> str:
    return str(fact.get("text", "")).strip()
