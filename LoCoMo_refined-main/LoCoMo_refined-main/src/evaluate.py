"""CLI and helpers for scoring public LoCoMo predictions."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bleu_f1 import compute_bleu1, compute_token_f1
from export import load_json_or_jsonl_records, write_json_or_jsonl_records

JudgeFn = Callable[..., Awaitable[dict[str, Any]]]
DEFAULT_LLM_JUDGE = "refined"
DEFAULT_PUBLIC_METRICS = ("llm", "f1", "bleu")
SUPPORTED_PUBLIC_METRICS = frozenset({"llm", "f1", "bleu", "all"})


@dataclass
class LLMRuntime:
    evaluate: Callable[..., Awaitable[dict[str, Any]]]
    shutdown: Callable[..., Awaitable[None]]
    config: dict[str, str | None]


@dataclass
class _ProgressReporter:
    total: int
    label: str = "Scoring"
    stream: Any = None
    update_interval_s: float = 0.1

    def __post_init__(self) -> None:
        if self.stream is None:
            self.stream = sys.stderr
        self.completed = 0
        self.started_at = time.monotonic()
        self.last_render_at = 0.0
        self.last_rendered_completed = -1
        self._is_tty = bool(getattr(self.stream, "isatty", lambda: False)())

    def start(self) -> None:
        if self.total <= 0:
            return
        if self._is_tty:
            self._render(force=True)
            return
        print(f"{self.label}: 0/{self.total}", file=self.stream, flush=True)

    def advance(self) -> None:
        if self.total <= 0:
            return
        self.completed += 1
        if self._is_tty:
            self._render()
            return
        if self.completed == self.total or self.completed == 1 or self.completed % 100 == 0:
            print(f"{self.label}: {self.completed}/{self.total}", file=self.stream, flush=True)

    def finish(self) -> None:
        if self.total <= 0:
            return
        self.completed = max(self.completed, self.total)
        if self._is_tty:
            if self.last_rendered_completed != self.completed:
                self._render(force=True)
            print(file=self.stream, flush=True)
            return
        if self.completed not in {0, 1} and self.completed % 100 != 0:
            print(f"{self.label}: {self.completed}/{self.total}", file=self.stream, flush=True)

    def _render(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_render_at < self.update_interval_s and self.completed < self.total:
            return
        self.last_render_at = now
        self.last_rendered_completed = self.completed
        width = 24
        ratio = 1.0 if self.total <= 0 else min(max(self.completed / self.total, 0.0), 1.0)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = now - self.started_at
        message = f"\r{self.label}: {self.completed}/{self.total} [{bar}] {ratio * 100:5.1f}% elapsed {elapsed:5.1f}s"
        print(message, file=self.stream, end="", flush=True)


async def evaluate_public_predictions(
    *,
    questions_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path | None = None,
    metrics: Sequence[str] | None = None,
    llm_judge: str = DEFAULT_LLM_JUDGE,
    judge_fn: JudgeFn | None = None,
    concurrency: int = 4,
    evaluator_model: str | None = None,
    evaluator_base_url: str | None = None,
    evaluator_api_key: str | None = None,
    strict: bool = True,
    progress: bool = False,
) -> list[dict[str, Any]]:
    question_records = load_json_or_jsonl_records(questions_path)
    predictions_by_id = _load_predictions(
        predictions_path=predictions_path,
        question_records=question_records,
        strict=strict,
    )
    merged_records = _merge_question_predictions(
        question_records=question_records,
        predictions_by_id=predictions_by_id,
        llm_judge=llm_judge,
        strict=strict,
    )

    enabled_metrics = _normalize_metrics(metrics or DEFAULT_PUBLIC_METRICS)
    llm_runtime: LLMRuntime | None = None
    if judge_fn is None and "llm" in enabled_metrics:
        llm_runtime = _configure_llm_runtime(
            llm_judge=llm_judge,
            evaluator_model=evaluator_model,
            evaluator_base_url=evaluator_base_url,
            evaluator_api_key=evaluator_api_key,
        )

    semaphore = asyncio.Semaphore(max(1, concurrency))
    progress_reporter = _ProgressReporter(total=len(merged_records)) if progress else None
    tasks: list[asyncio.Task[tuple[int, dict[str, Any]]]] = []
    scored_pairs: list[tuple[int, dict[str, Any]]] = []

    async def _score_one(index: int, record: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            return index, await _evaluate_public_record(
                record=record,
                enabled_metrics=enabled_metrics,
                judge_fn=judge_fn,
                llm_judge=llm_judge,
                llm_runtime=llm_runtime,
            )

    try:
        tasks = [asyncio.create_task(_score_one(index, record)) for index, record in enumerate(merged_records)]
        if progress_reporter is not None and tasks:
            progress_reporter.start()
            for task in tasks:
                task.add_done_callback(lambda _: progress_reporter.advance())
        scored_pairs = await asyncio.gather(*tasks) if tasks else []
    finally:
        if progress_reporter is not None and tasks:
            progress_reporter.finish()
        if llm_runtime is not None:
            await llm_runtime.shutdown(llm_judge)

    evaluated = [result for _, result in sorted(scored_pairs, key=lambda item: item[0])]
    if output_path is not None:
        write_json_or_jsonl_records(records=evaluated, path=output_path)
    return evaluated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate public LoCoMo predictions")
    parser.add_argument("--questions-path", required=True)
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_PUBLIC_METRICS))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--llm-judge", default=DEFAULT_LLM_JUDGE)
    parser.add_argument("--evaluator-model", default=None)
    parser.add_argument("--evaluator-base-url", default=None)
    parser.add_argument("--evaluator-api-key", default=None)
    parser.add_argument("--allow-extra-predictions", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = args.output_path or _default_output_path(args.predictions_path)
    result = asyncio.run(
        evaluate_public_predictions(
            questions_path=args.questions_path,
            predictions_path=args.predictions_path,
            output_path=output_path,
            metrics=args.metrics,
            concurrency=args.concurrency,
            llm_judge=args.llm_judge,
            evaluator_model=args.evaluator_model,
            evaluator_base_url=args.evaluator_base_url,
            evaluator_api_key=args.evaluator_api_key,
            strict=not bool(args.allow_extra_predictions),
            progress=not bool(args.no_progress),
        )
    )
    print(
        json.dumps(
            {
                "record_count": len(result),
                "output_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _default_output_path(predictions_path: str | Path) -> Path:
    path = Path(predictions_path)
    if path.suffix == ".jsonl":
        return path.with_name(f"{path.stem}_scored.jsonl")
    suffix = path.suffix or ".json"
    return path.with_name(f"{path.stem}_scored{suffix}")


def _load_predictions(
    *,
    predictions_path: str | Path,
    question_records: Sequence[dict[str, Any]],
    strict: bool,
) -> dict[str, dict[str, Any]]:
    records = load_json_or_jsonl_records(predictions_path)
    known_qa_ids = {
        str(item.get("qa_id") or "").strip() for item in question_records if str(item.get("qa_id") or "").strip()
    }
    indexed_question_ids: dict[tuple[int, int], str] = {}
    for item in question_records:
        qa_id = str(item.get("qa_id") or "").strip()
        conversation_idx = _coerce_index_value(item.get("conversation_idx"))
        qa_index = _coerce_index_value(item.get("qa_index"))
        if qa_id and conversation_idx is not None and qa_index is not None:
            indexed_question_ids[(conversation_idx, qa_index)] = qa_id

    predictions_by_id: dict[str, dict[str, Any]] = {}
    unknown_qa_ids: list[str] = []
    for item in records:
        qa_id = _resolve_prediction_qa_id(
            item=item,
            indexed_question_ids=indexed_question_ids,
            strict=strict,
        )
        if qa_id is None:
            continue
        if qa_id not in known_qa_ids:
            unknown_qa_ids.append(qa_id)
            continue
        if qa_id in predictions_by_id:
            raise ValueError(f"duplicate prediction for qa_id: {qa_id}")
        predictions_by_id[qa_id] = {**item, "qa_id": qa_id}

    if strict and unknown_qa_ids:
        sample = ", ".join(sorted(set(unknown_qa_ids))[:5])
        raise ValueError(f"prediction file contains unknown qa_id values: {sample}")
    return predictions_by_id


def _merge_question_predictions(
    *,
    question_records: Sequence[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
    llm_judge: str,
    strict: bool,
) -> list[dict[str, Any]]:
    merged_records: list[dict[str, Any]] = []
    for question_record in question_records:
        qa_id = str(question_record.get("qa_id") or "").strip()
        prediction_record = predictions_by_id.get(qa_id)
        prediction_found = prediction_record is not None
        predicted_answer = ""
        if prediction_record is not None:
            predicted_answer = str(
                prediction_record.get("predicted_answer") or prediction_record.get("response") or ""
            ).strip()

        merged_records.append(
            {
                **question_record,
                "predicted_answer": predicted_answer,
                "prediction_found": prediction_found,
                "llm_judge": llm_judge,
                "success": prediction_found,
                "errors": [] if prediction_found else [_missing_prediction_error(qa_id)],
            }
        )
    return merged_records


def _missing_prediction_error(qa_id: str) -> dict[str, Any]:
    return {
        "stage": "prediction_lookup",
        "error": f"missing prediction for qa_id={qa_id}",
        "error_code": "MISSING_PREDICTION",
        "retriable": False,
    }


def _resolve_prediction_qa_id(
    *,
    item: dict[str, Any],
    indexed_question_ids: dict[tuple[int, int], str],
    strict: bool,
) -> str | None:
    qa_id = str(item.get("qa_id") or "").strip()
    if qa_id:
        return qa_id

    conversation_idx = _coerce_index_value(item.get("conversation_idx"))
    qa_index = _coerce_index_value(item.get("qa_index"))
    if conversation_idx is None or qa_index is None:
        raise ValueError("prediction record missing qa_id or conversation_idx + qa_index")
    resolved_qa_id = indexed_question_ids.get((conversation_idx, qa_index))
    if resolved_qa_id is not None or not strict:
        return resolved_qa_id
    raise ValueError(
        "prediction record references unknown conversation_idx + qa_index: "
        f"conversation_idx={conversation_idx}, qa_index={qa_index}"
    )


def _coerce_index_value(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalize_metrics(metrics: Sequence[str]) -> set[str]:
    normalized = {
        part.strip().lower()
        for metric in metrics
        for part in str(metric).split(",")
        if part.strip()
    }
    unknown = normalized - SUPPORTED_PUBLIC_METRICS
    if unknown:
        raise ValueError(f"unknown metrics requested: {', '.join(sorted(unknown))}")
    if "all" in normalized:
        normalized.discard("all")
        normalized.update({"llm", "f1", "bleu"})
    return normalized


def _configure_llm_runtime(
    *,
    llm_judge: str,
    evaluator_model: str | None,
    evaluator_base_url: str | None,
    evaluator_api_key: str | None,
) -> LLMRuntime:
    from llm_judge import (
        ashutdown_llm_judge,
        configure_llm_judge,
        evaluate_llm_judge,
        resolve_llm_judge_config,
    )

    resolved_config = resolve_llm_judge_config(
        name=llm_judge,
        model=evaluator_model,
        base_url=evaluator_base_url,
        api_key=evaluator_api_key,
    )
    configure_llm_judge(
        name=llm_judge,
        model=resolved_config["model"],
        base_url=resolved_config["base_url"],
        api_key=resolved_config["api_key"],
    )
    return LLMRuntime(evaluate=evaluate_llm_judge, shutdown=ashutdown_llm_judge, config=resolved_config)


async def _evaluate_public_record(
    *,
    record: dict[str, Any],
    enabled_metrics: set[str],
    judge_fn: JudgeFn | None,
    llm_judge: str,
    llm_runtime: LLMRuntime | None,
) -> dict[str, Any]:
    question = str(record.get("question") or "")
    original_text = str(record.get("predicted_answer") or record.get("response") or "")
    normalized_prediction = _normalize_prediction_text(original_text)
    answer_candidates = _normalize_answer_candidates(record.get("answer"), "")

    best_answer = ""
    metrics_summary: dict[str, Any] = {}
    if any(metric in enabled_metrics for metric in {"llm", "f1", "bleu"}):
        best_answer, lexical_scores = await _evaluate_candidates(
            question=question,
            prediction=normalized_prediction,
            candidates=answer_candidates,
            enabled_metrics=enabled_metrics,
            judge_fn=judge_fn,
            llm_judge=llm_judge,
            llm_runtime=llm_runtime,
        )
        metrics_summary.update(lexical_scores)

    if not best_answer:
        best_answer = answer_candidates[0] if answer_candidates else ""

    output = {
        **record,
        "matched_answer": best_answer,
        "response": normalized_prediction,
        **metrics_summary,
    }
    if normalized_prediction != original_text.strip():
        output["ori_response"] = original_text
    return output


async def _evaluate_candidates(
    *,
    question: str,
    prediction: str,
    candidates: Sequence[str],
    enabled_metrics: set[str],
    judge_fn: JudgeFn | None,
    llm_judge: str,
    llm_runtime: LLMRuntime | None,
) -> tuple[str, dict[str, float | str]]:
    exact_match = next((str(candidate) for candidate in candidates if prediction == str(candidate)), None)
    if exact_match is not None:
        summary: dict[str, float | str] = {}
        if "llm" in enabled_metrics:
            summary["llm_score"] = 1.0
            summary["llm_reason"] = "Predicted answer exactly matches the reference answer."
        if "f1" in enabled_metrics:
            summary["f1_score"] = 1.0
        if "bleu" in enabled_metrics:
            summary["bleu_score"] = 1.0
        return exact_match, summary

    if prediction == "" and any(str(candidate) != "" for candidate in candidates):
        summary = {}
        if "llm" in enabled_metrics:
            summary["llm_score"] = 0.0
            summary["llm_reason"] = "Predicted answer is empty while the reference answer is not."
        if "f1" in enabled_metrics:
            summary["f1_score"] = 0.0
        if "bleu" in enabled_metrics:
            summary["bleu_score"] = 0.0
        best_answer = next((str(candidate) for candidate in candidates if str(candidate) != ""), str(candidates[0]))
        return best_answer, summary

    if not candidates or all(candidate == "" for candidate in candidates):
        summary: dict[str, float | str] = {}
        if "llm" in enabled_metrics:
            summary["llm_score"] = 0.0
            summary["llm_reason"] = ""
        if "f1" in enabled_metrics:
            summary["f1_score"] = 0.0
        if "bleu" in enabled_metrics:
            summary["bleu_score"] = 0.0
        return "", summary

    llm_tasks: list[asyncio.Task[tuple[str, dict[str, Any]]]] = []
    candidate_metrics_map: dict[str, dict[str, float | str]] = {}
    for candidate in candidates:
        metrics: dict[str, float | str] = {}
        if "f1" in enabled_metrics:
            metrics["f1"] = compute_token_f1(prediction, candidate)
        if "bleu" in enabled_metrics:
            metrics["bleu"] = compute_bleu1(prediction, candidate)
        candidate_metrics_map[candidate] = metrics
        if "llm" in enabled_metrics:
            llm_tasks.append(
                asyncio.create_task(
                    _evaluate_llm_candidate(
                        question=question,
                        candidate=candidate,
                        prediction=prediction,
                        judge_fn=judge_fn,
                        llm_judge=llm_judge,
                        llm_runtime=llm_runtime,
                    )
                )
            )

    if llm_tasks:
        for task in asyncio.as_completed(llm_tasks):
            candidate, judge_result = await task
            candidate_metrics_map[candidate]["llm"] = float(judge_result.get("score", 0.0) or 0.0)
            candidate_metrics_map[candidate]["llm_reason"] = str(judge_result.get("reason", "") or "")

    def _sort_key(c: str) -> tuple[float, float, float]:
        m = candidate_metrics_map.get(c, {})
        llm_v = float(m.get("llm", 0.0) or 0.0) if "llm" in enabled_metrics else -1.0
        f1_v = float(m.get("f1", 0.0) or 0.0) if "f1" in enabled_metrics else -1.0
        bleu_v = float(m.get("bleu", 0.0) or 0.0) if "bleu" in enabled_metrics else -1.0
        return (llm_v, f1_v, bleu_v)

    best_answer = max(candidates, key=_sort_key)
    best_metrics = candidate_metrics_map.get(best_answer, {})

    summary: dict[str, float | str] = {}
    if "llm" in enabled_metrics:
        summary["llm_score"] = float(best_metrics.get("llm", 0.0) or 0.0)
        summary["llm_reason"] = str(best_metrics.get("llm_reason", "") or "")
    if "f1" in enabled_metrics:
        summary["f1_score"] = float(best_metrics.get("f1", 0.0) or 0.0)
    if "bleu" in enabled_metrics:
        summary["bleu_score"] = float(best_metrics.get("bleu", 0.0) or 0.0)
    return best_answer, summary


async def _evaluate_llm_candidate(
    *,
    question: str,
    candidate: str,
    prediction: str,
    judge_fn: JudgeFn | None,
    llm_judge: str,
    llm_runtime: LLMRuntime | None,
) -> tuple[str, dict[str, Any]]:
    if judge_fn is not None:
        result = await judge_fn(
            question=question,
            reference_answer=candidate,
            predicted_answer=prediction,
        )
        return candidate, result
    if llm_runtime is None:
        raise RuntimeError("llm runtime not configured")
    result = await llm_runtime.evaluate(
        question=question,
        reference_answer=candidate,
        predicted_answer=prediction,
        name=llm_judge,
    )
    return candidate, result


def _normalize_answer_candidates(raw_candidates: Any, fallback: Any) -> list[str]:
    candidates: list[str] = []
    if isinstance(raw_candidates, list):
        candidates.extend(str(candidate) for candidate in raw_candidates if candidate not in (None, ""))
    elif raw_candidates not in (None, ""):
        candidates.append(str(raw_candidates))
    if not candidates and fallback not in (None, ""):
        candidates.append(str(fallback))
    if not candidates:
        candidates.append("")
    return candidates


def _normalize_prediction_text(text: str) -> str:
    cleaned = str(text or "").strip()
    boxed_value = _extract_last_boxed_value(cleaned)
    if boxed_value is not None:
        return boxed_value
    lower = cleaned.lower()
    if "final answer:" in lower:
        idx = lower.index("final answer:")
        cleaned = cleaned[idx + len("final answer:") :].strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    return cleaned


def _extract_last_boxed_value(text: str) -> str | None:
    last_value: str | None = None
    for match in re.finditer(r"\\box(?:ed)?\{", text):
        opening_brace_index = match.end() - 1
        extracted = _extract_braced_content(text, opening_brace_index)
        if extracted is not None:
            last_value = extracted.strip()
    return last_value


def _extract_braced_content(text: str, opening_brace_index: int) -> str | None:
    if opening_brace_index < 0 or opening_brace_index >= len(text) or text[opening_brace_index] != "{":
        return None

    depth = 0
    content_start = opening_brace_index + 1
    for index in range(opening_brace_index, len(text)):
        ch = text[index]
        if ch == "{":
            depth += 1
            continue
        if ch != "}":
            continue
        depth -= 1
        if depth == 0:
            return text[content_start:index]
    return text[content_start:]


if __name__ == "__main__":
    main()
