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
    "3. 时间、日期、顺序类信息必须保留：\n"
    "   - 具体时间：上周六、2023年5月、Saturday、last week、two days ago\n"
    "   - 事件顺序：first、then、after that、before、later\n"
    "   - 含时间的事实要单独成条，不要把时间省略\n"
    "4. 每条记忆必须能脱离上下文被独立理解。\n"
    "5. 只输出 JSON 数组，数组元素为字符串。不要输出解释、序号或 markdown。\n\n"
    "示例：\n"
    "输入：user: 我上周六和朋友去中央公园野餐，带了草莓。\n"
    "输出：[\"用户上周六和朋友去中央公园野餐\", \"用户野餐时带了草莓\"]"
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


def _parse_llm_output(raw: str) -> List[str]:
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
            # LLM occasionally echoes dialogue formatting ("user: ..."); strip it.
            cleaned = re.sub(
                r"^(user|assistant|system)\s*:\s*", "", item.strip(), flags=re.IGNORECASE
            )
            if cleaned:
                result.append(cleaned)
        elif isinstance(item, dict) and item.get("fact"):
            result.append(str(item["fact"]).strip())
    return result


def consolidate(messages_text: str) -> List[str]:
    """Turn a serialized message block into atomic memory units.

    Uses the configured LLM when enabled; falls back to sentence splitting
    so Add never loses searchability even if the model is unavailable.
    """
    if not settings.enable_llm_consolidation:
        return _split_sentences(messages_text)

    prompt = (
        "请将以下对话内容压缩为事实记忆：\n\n"
        f"{messages_text}\n\n"
        "只输出 JSON 字符串数组。"
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
    return _split_sentences(messages_text)
