"""Public dataset export helpers for LoCoMo benchmark releases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_PUBLIC_EXPORT_DIR = Path("evals/locomo/public")


def build_public_qa_id(sample_id: str, qa_index: int) -> str:
    normalized = str(sample_id or "").strip() or "sample"
    return f"{normalized}#q{qa_index:04d}"


def load_json_or_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"{exc.msg} (while parsing {input_path} line {line_number})",
                    exc.doc,
                    exc.pos,
                ) from exc
            if isinstance(payload, dict):
                records.append(payload)
        return records

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError(f"expected JSON array or JSONL records: {input_path}")


def write_json_or_jsonl_records(*, records: Sequence[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".jsonl":
        lines = [json.dumps(record, ensure_ascii=False) for record in records]
        payload = "\n".join(lines)
        if lines:
            payload += "\n"
        output_path.write_text(payload, encoding="utf-8")
        return output_path

    output_path.write_text(
        json.dumps(list(records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def default_public_export_dir_for_dataset(dataset_path: str | Path) -> Path:
    input_path = Path(dataset_path)
    return input_path.parent / f"{input_path.stem}_public"


def export_public_dataset(
    *,
    dataset_path: str | Path,
    output_dir: str | Path | None = None,
    max_conversations: int | None = None,
) -> dict[str, Path]:
    input_path = Path(dataset_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected top-level dataset array: {input_path}")

    resolved_output_dir = (
        Path(output_dir) if output_dir is not None else default_public_export_dir_for_dataset(input_path)
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    conversations: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    category_counter: Counter[str] = Counter()
    multimodal_counter: Counter[str] = Counter()

    for conversation_idx, record in enumerate(payload):
        if max_conversations is not None and conversation_idx >= max_conversations:
            break
        if not isinstance(record, dict):
            continue
        conversation_record, question_records = _convert_conversation(
            conversation_idx=conversation_idx,
            record=record,
        )
        conversations.append(conversation_record)
        questions.extend(question_records)
        for item in question_records:
            category_counter[str(item.get("category") or "")] += 1
            multimodal_counter["true" if bool(item.get("is_multi_modality")) else "false"] += 1

    conversation_path = resolved_output_dir / "conversations.jsonl"
    questions_path = resolved_output_dir / "questions.jsonl"
    submission_template_path = resolved_output_dir / "submission_template.jsonl"
    manifest_path = resolved_output_dir / "manifest.json"

    write_json_or_jsonl_records(records=conversations, path=conversation_path)
    write_json_or_jsonl_records(records=questions, path=questions_path)
    write_json_or_jsonl_records(
        records=[{"qa_id": item["qa_id"], "predicted_answer": ""} for item in questions],
        path=submission_template_path,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_file_name": input_path.name,
                "conversation_count": len(conversations),
                "question_count": len(questions),
                "category_counts": dict(sorted(category_counter.items())),
                "is_multi_modality_counts": {
                    "false": int(multimodal_counter.get("false", 0)),
                    "true": int(multimodal_counter.get("true", 0)),
                },
                "files": {
                    "conversations": conversation_path.name,
                    "questions": questions_path.name,
                    "submission_template": submission_template_path.name,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": resolved_output_dir,
        "conversations": conversation_path,
        "questions": questions_path,
        "submission_template": submission_template_path,
        "manifest": manifest_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a public LoCoMo dataset package")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-conversations", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_public_dataset(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_conversations=args.max_conversations,
    )
    print(json.dumps({key: str(value) for key, value in result.items()}, ensure_ascii=False, indent=2))


def _convert_conversation(
    *,
    conversation_idx: int,
    record: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    conversation = record.get("conversation") or {}
    sample_id = str(record.get("sample_id") or f"conversation-{conversation_idx:04d}")
    speaker_a = str(conversation.get("speaker_a") or "")
    speaker_b = str(conversation.get("speaker_b") or "")

    sessions, message_lookup = _build_sessions(
        conversation=conversation,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
    )
    question_records = _build_question_records(
        conversation_idx=conversation_idx,
        sample_id=sample_id,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        qa_items=record.get("qa") or [],
        message_lookup=message_lookup,
    )

    message_count = sum(len(session["messages"]) for session in sessions)
    multimodal_message_count = sum(
        1 for session in sessions for message in session["messages"] if bool(message.get("has_multimodal_context"))
    )
    image_count = sum(len(message.get("images") or []) for session in sessions for message in session["messages"])

    return {
        "sample_id": sample_id,
        "conversation_idx": conversation_idx,
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "session_count": len(sessions),
        "message_count": message_count,
        "multimodal_message_count": multimodal_message_count,
        "image_count": image_count,
        "sessions": sessions,
        "conversation_history_text": _render_conversation_history_text(sessions=sessions, include_multimodal=False),
        "conversation_history_multimodal_text": _render_conversation_history_text(
            sessions=sessions,
            include_multimodal=True,
        ),
    }, question_records


def _build_sessions(
    *,
    conversation: dict[str, Any],
    speaker_a: str,
    speaker_b: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    sessions: list[dict[str, Any]] = []
    message_lookup: dict[str, dict[str, Any]] = {}
    session_index = 1
    while True:
        session_key = f"session_{session_index}"
        if session_key not in conversation:
            break

        date_time = str(conversation.get(f"session_{session_index}_date_time") or "")
        raw_messages = conversation.get(session_key) or []
        messages: list[dict[str, Any]] = []
        for position, raw_message in enumerate(raw_messages, start=1):
            if not isinstance(raw_message, dict):
                continue
            text = str(raw_message.get("text") or "").strip()
            if not text:
                continue
            dia_id = str(raw_message.get("dia_id") or "").strip()
            message_index = _parse_message_index(dia_id) or position
            images = _normalize_image_list(raw_message.get("img_url"))
            message = {
                "session_index": session_index,
                "session_date_time": date_time,
                "message_index": message_index,
                "dia_id": dia_id,
                "speaker": str(raw_message.get("speaker") or ""),
                "role": _normalize_role(
                    raw_message.get("speaker"),
                    speaker_a=speaker_a,
                    speaker_b=speaker_b,
                ),
                "text": text,
                "images": images,
                "blip_caption": str(raw_message.get("blip_caption") or "").strip(),
                "query": str(raw_message.get("query") or "").strip(),
            }
            message["has_multimodal_context"] = bool(message["images"] or message["blip_caption"] or message["query"])
            messages.append(message)
            if dia_id:
                message_lookup[dia_id] = message

        if not messages:
            break
        sessions.append(
            {
                "session_index": session_index,
                "date_time": date_time,
                "messages": messages,
            }
        )
        session_index += 1

    return sessions, message_lookup


def _build_question_records(
    *,
    conversation_idx: int,
    sample_id: str,
    speaker_a: str,
    speaker_b: str,
    qa_items: Any,
    message_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(qa_items, list):
        return []

    question_records: list[dict[str, Any]] = []
    for qa_index, raw_item in enumerate(qa_items):
        if not isinstance(raw_item, dict):
            continue
        question = str(raw_item.get("question") or "").strip()
        answer = _normalize_string_list(raw_item.get("answer"))
        evidence = _normalize_string_list(raw_item.get("evidence"))
        question_records.append(
            {
                "qa_id": build_public_qa_id(sample_id, qa_index),
                "sample_id": sample_id,
                "conversation_idx": conversation_idx,
                "qa_index": qa_index,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "question": question,
                "answer": answer,
                "category": str(raw_item.get("category") or ""),
                "is_multi_modality": bool(raw_item.get("is_multi_modality")),
                "evidence": evidence,
                "evidence_messages": [
                    _simplify_message_for_evidence(message_lookup[dia_id])
                    for dia_id in evidence
                    if dia_id in message_lookup
                ],
            }
        )
    return question_records


def _render_conversation_history_text(
    *,
    sessions: Sequence[dict[str, Any]],
    include_multimodal: bool,
) -> str:
    lines: list[str] = []
    for session in sessions:
        session_index = int(session.get("session_index", 0) or 0)
        date_time = str(session.get("date_time") or "")
        lines.append(f"--- Session {session_index} started at {date_time} ---")
        for message in session.get("messages") or []:
            if not isinstance(message, dict):
                continue
            speaker = str(message.get("speaker") or "")
            text = str(message.get("text") or "")
            lines.append(f"{speaker}: {text}")
            if not include_multimodal:
                continue
            images = message.get("images") or []
            if images:
                lines.append(f"  [images] {', '.join(str(item) for item in images)}")
            blip_caption = str(message.get("blip_caption") or "")
            if blip_caption:
                lines.append(f"  [caption] {blip_caption}")
            query = str(message.get("query") or "")
            if query:
                lines.append(f"  [query] {query}")
    return "\n".join(lines)


def _simplify_message_for_evidence(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "dia_id": str(message.get("dia_id") or ""),
        "session_index": int(message.get("session_index", 0) or 0),
        "message_index": int(message.get("message_index", 0) or 0),
        "speaker": str(message.get("speaker") or ""),
        "text": str(message.get("text") or ""),
        "images": [str(item) for item in message.get("images") or []],
        "blip_caption": str(message.get("blip_caption") or ""),
        "query": str(message.get("query") or ""),
        "has_multimodal_context": bool(message.get("has_multimodal_context")),
    }


def _normalize_string_list(raw_value: Any, *, fallback: Any = None) -> list[str]:
    values: list[str] = []
    if isinstance(raw_value, list):
        values.extend(str(item).strip() for item in raw_value if str(item).strip())
    elif raw_value not in (None, ""):
        text = str(raw_value).strip()
        if text:
            values.append(text)
    if not values and fallback not in (None, ""):
        if isinstance(fallback, list):
            values.extend(str(item).strip() for item in fallback if str(item).strip())
        else:
            fallback_text = str(fallback).strip()
            if fallback_text:
                values.append(fallback_text)
    return values


def _normalize_image_list(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if raw_value in (None, ""):
        return []
    text = str(raw_value).strip()
    return [text] if text else []


def _parse_message_index(dia_id: str) -> int | None:
    value = str(dia_id or "").strip()
    if ":" not in value:
        return None
    _, _, suffix = value.partition(":")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _normalize_role(raw_speaker: Any, *, speaker_a: str, speaker_b: str) -> str:
    speaker = str(raw_speaker or "").strip()
    if speaker == speaker_a:
        return "user"
    if speaker == speaker_b:
        return "assistant"
    return "user"


if __name__ == "__main__":
    main()
