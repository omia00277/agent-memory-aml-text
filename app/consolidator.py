import json
import logging
import re
from typing import List, Optional

from app.config import settings
from app.siliconflow_client import sf_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是长期记忆归档助手。把输入的多轮对话压缩为若干条原子化、自包含、"
    "无歧义的事实记忆。\n\n"
    "要求：\n"
    "1. 保留关键实体、具体事件、时间、地点、职业、偏好、关系等原文细节，"
    "不要过度概括。例如 '去公园野餐' 不能简化为 '在公园度过愉快时光'。\n"
    "2. 对同一话题，既要有概括性事实，也要有具体细节事实。\n"
    "3. 【粒度优先】宁可多输出几条，也不要把多条独立信息合并成一条。\n"
    "4. 每条记忆必须能脱离上下文被独立理解。\n"
    "5. 只输出 JSON 数组，数组元素为对象，形如 {\"content\": \"...\"}；"
    "可选字段 entities 列出涉及的实体词。不要输出解释、序号或 markdown。\n\n"
    "示例：\n"
    "输入：user: 我上周六和朋友去中央公园野餐，带了草莓。\n"
    "输出：[{\"content\": \"用户上周六和朋友去中央公园野餐\"}, "
    "{\"content\": \"用户野餐时带了草莓\"}]"
)

# State extraction is deliberately a separate, narrow LLM task. Putting it inside
# SYSTEM_PROMPT made the model trade fact granularity for structured fields, and
# worse, it silently skipped the state fact when the chunk also contained another
# state (e.g. a move plus a phone number) — reproducible at temperature 0.
STATE_PROMPT = (
    "你只做一件事：从对话中抽取「单值属性的最新取值」。\n\n"
    "单值属性＝同一时刻只能有一个取值的东西，例如：居住地、手机号、职业、年龄、姓名、"
    "学历、婚姻状况、体重、邮箱、公司。\n"
    "【不要】抽取观点、感受、情绪、经历、活动、兴趣、爱好、艺术作品、人际关系——"
    "它们可以同时存在多个，不是单值属性。\n"
    "如果用户提到变更（搬家、换工作、改手机号等），请推断出变更后的最新取值。\n\n"
    "只输出 JSON 数组，元素形如 "
    "{\"attribute\": \"居住地\", \"value\": \"上海\", \"content\": \"用户居住在上海\"}。\n"
    "没有任何单值属性时输出 []。不要输出解释或 markdown。\n\n"
    "示例 1：\n"
    "输入：user: 我上个月搬到上海了。\n"
    "输出：[{\"attribute\": \"居住地\", \"value\": \"上海\", \"content\": \"用户居住在上海\"}]\n\n"
    "示例 2：\n"
    "输入：user: 我上周去公园野餐，很开心。\n"
    "输出：[]"
)


def _split_sentences(text: str) -> List[str]:
    """Deterministic fallback: split text into clean, self-contained lines."""
    parts = re.split(r"\n+", text.strip())
    units = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Strip role prefixes such as "user: " / "assistant: ".
        part = re.sub(r"^(user|assistant|system)\s*:\s*", "", part, flags=re.IGNORECASE)
        for sentence in re.split(r"[。！？!?；;]+", part):
            sentence = sentence.strip()
            if sentence and len(sentence) >= 2:
                units.append(sentence)
    return units


def _parse_llm_output(raw: str) -> List[dict]:
    """Parse LLM output into a list of structured fact dictionaries.

    Supports:
    - New structured format: [{"content": ..., "fact_type": ..., ...}]
    - Legacy string format: ["fact1", "fact2"]
    - Legacy dict format: [{"fact": "..."}]
    """
    if not raw:
        return []
    text = raw.strip()
    # Accept either a bare JSON array or fenced JSON.
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end <= start:
            return []
        text = text[start : end + 1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []

    result = []
    for item in data:
        if isinstance(item, str) and item.strip():
            cleaned = re.sub(
                r"^(user|assistant|system)\s*:\s*", "", item.strip(), flags=re.IGNORECASE
            )
            if cleaned:
                result.append({"content": cleaned})
        elif isinstance(item, dict):
            if item.get("fact"):
                result.append({"content": str(item["fact"]).strip()})
            elif item.get("content"):
                fact = {"content": str(item["content"]).strip()}
                for key in ("fact_type", "attribute", "value", "entities"):
                    if key in item:
                        fact[key] = item[key]
                if isinstance(fact.get("entities"), str):
                    fact["entities"] = [fact["entities"]]
                result.append(fact)
    return result


def consolidate(messages_text: str) -> List[dict]:
    """Turn a serialized message block into atomic structured memory units.

    Uses the configured LLM when enabled; falls back to sentence splitting
    so Add never loses searchability even if the model is unavailable.

    Returns a list of dicts with at least a "content" key, optionally including
    structured fields such as fact_type, attribute, value, and entities.
    """
    if not settings.enable_llm_consolidation:
        return [{"content": s} for s in _split_sentences(messages_text)]

    prompt = (
        "请将以下对话内容压缩为事实记忆：\n\n"
        f"{messages_text}\n\n"
        "只输出 JSON 数组。"
    )
    try:
        raw = sf_client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        units = _parse_llm_output(raw)
        if units:
            return units
        logger.warning("Consolidation LLM returned no parseable facts; using fallback.")
    except Exception as e:
        logger.warning(f"Consolidation LLM failed ({e}); using sentence fallback.")
    return [{"content": s} for s in _split_sentences(messages_text)]


def extract_state_facts(messages_text: str) -> List[dict]:
    """Extract current single-valued property values as their own memory units.

    This is a narrow, separate LLM task (see STATE_PROMPT) so that state tracking
    does not compete with fact consolidation for the model's attention.

    Returns a list of dicts with content/attribute/value/fact_type. On any failure
    it returns an empty list; the caller keeps its regex fallback for such cases.
    """
    if not settings.enable_llm_state_extraction:
        return []

    prompt = f"对话内容：\n\n{messages_text}"
    try:
        raw = sf_client.chat(
            [
                {"role": "system", "content": STATE_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        states = _parse_state_output(raw)
        return states
    except Exception as e:
        logger.warning(f"State extraction failed ({e}); continuing without it.")
        return []


def _parse_state_output(raw: str) -> List[dict]:
    """Parse the narrow state-extraction output into fact dicts."""
    if not raw:
        return []
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []

    results = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        attribute = item.get("attribute")
        value = item.get("value")
        content = item.get("content")
        if not attribute or value is None or not content:
            continue
        results.append({
            "content": str(content).strip(),
            "fact_type": "personal_state",
            "attribute": str(attribute).strip(),
            "value": str(value).strip(),
        })
    return results
