"""LoCoMo-Refined dataset loading and AML-style chunking for local evaluation."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

SESSION_DT_FORMAT = "%I:%M %p on %d %B, %Y"


def parse_session_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), SESSION_DT_FORMAT)
    except (ValueError, AttributeError, TypeError):
        return None


def load_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_conversations(data_dir: Path) -> List[dict]:
    return list(load_jsonl(data_dir / "conversations.jsonl"))


def load_questions(data_dir: Path) -> List[dict]:
    return list(load_jsonl(data_dir / "questions.jsonl"))


def build_flat_messages(
    conversation: dict,
    include_captions: bool = False,
) -> List[dict]:
    """Flatten sessions into an ordered message list with millisecond timestamps."""
    out: List[dict] = []
    speaker_a = conversation.get("speaker_a")
    for session in sorted(conversation["sessions"], key=lambda s: s["session_index"]):
        dt = parse_session_dt(session.get("date_time"))
        base_ts = int(dt.timestamp() * 1000) if dt else None
        for msg in sorted(session["messages"], key=lambda m: m["message_index"]):
            text = (msg.get("text") or "").strip()
            caption = (msg.get("blip_caption") or "").strip()
            if include_captions and caption:
                text = f"{text}\n[image: {caption}]".strip()
            if not text:
                continue
            ts = None
            if base_ts is not None:
                ts = base_ts + int(msg["message_index"]) * 1000
            role = msg.get("role")
            if role not in ("user", "assistant"):
                role = "user" if msg.get("speaker") == speaker_a else "assistant"
            out.append(
                {
                    "dia_id": msg["dia_id"],
                    "role": role,
                    "content": text,
                    "ts_ms": ts,
                }
            )
    return out


def chunk_messages(
    messages: List[dict],
    max_messages: int = 20,
    max_words: int = 2000,
) -> List[List[dict]]:
    """Replicate the AML deterministic chunking: 20 messages or 2,000 words."""
    chunks: List[List[dict]] = []
    current: List[dict] = []
    current_words = 0
    for msg in messages:
        words = len(msg["content"].split())
        if current and (
            len(current) + 1 > max_messages or current_words + words > max_words
        ):
            chunks.append(current)
            current = []
            current_words = 0
        current.append(msg)
        current_words += words
    if current:
        chunks.append(current)
    return chunks


def group_questions_by_sample(questions: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for q in questions:
        grouped.setdefault(q["sample_id"], []).append(q)
    return grouped
