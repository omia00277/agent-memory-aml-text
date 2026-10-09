"""Local LoCoMo-Refined retrieval evaluation against the real memory pipeline.

Usage (from project root):

    venv\\Scripts\\python.exe bench\\run_locomo_eval.py --limit 1

Uses real SiliconFlow calls (LLM consolidation, embedding, rerank). Storage is
isolated in bench.db / qdrant_bench so the dev server's data is untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path


def _configure_storage_from_argv() -> str:
    """Isolate benchmark storage, before the app (and Qdrant client) is imported.

    Without this every run reuses ./bench.db and ./qdrant_bench. Because Add is
    idempotent on request_id, later runs would silently skip every Add and only
    re-measure the search path against stale units.

    --tag NAME   store under ./bench_NAME.db and ./qdrant_bench_NAME
    --fresh      delete that storage first so every Add really runs
    """
    argv = sys.argv
    tag = ""
    fresh = "--fresh" in argv
    for i, arg in enumerate(argv):
        if arg == "--tag" and i + 1 < len(argv):
            tag = argv[i + 1]
        elif arg.startswith("--tag="):
            tag = arg.split("=", 1)[1]

    if tag:
        db_path = f"./bench_{tag}.db"
        qdrant_path = f"./qdrant_bench_{tag}"
    else:
        db_path = "./bench.db"
        qdrant_path = "./qdrant_bench"

    if fresh:
        for p in (db_path, qdrant_path):
            Path(p).unlink(missing_ok=True) if Path(p).is_file() else shutil.rmtree(p, ignore_errors=True)

    os.environ["QDRANT_PATH"] = qdrant_path
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ.setdefault("ENABLE_LLM_CONSOLIDATION", "true")
    return tag or "default"


STORAGE_TAG = _configure_storage_from_argv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.database import MemoryUnit, RawChunk, SessionLocal, init_db
from app.memory_service import add_memory, search_memory
from app.schemas import AddRequest, Message, SearchRequest

from bench.locomo_loader import (
    build_flat_messages,
    chunk_messages,
    group_questions_by_sample,
    load_conversations,
    load_questions,
)


def evaluate_question(items, gold_dia_ids, unit_map, chunk_map):
    gold = set(gold_dia_ids)
    covered = set()
    first_hit_rank = None
    for rank, item in enumerate(items, start=1):
        src = unit_map.get(item.id)
        dias = set(chunk_map.get(src, []))
        overlap = dias & gold
        if overlap:
            covered |= overlap
            if first_hit_rank is None:
                first_hit_rank = rank
    return {
        "hit": bool(covered),
        "coverage": len(covered) / len(gold) if gold else 0.0,
        "mrr": 1.0 / first_hit_rank if first_hit_rank else 0.0,
        "covered": sorted(covered),
    }


def normalize_evidence(raw):
    """Evidence entries may pack several dia_ids into one string: 'D8:6; D9:17'."""
    gold = set()
    for entry in raw or []:
        for part in str(entry).split(";"):
            part = part.strip()
            if part:
                gold.add(part)
    return gold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="LoCoMo_refined-main/LoCoMo_refined-main/data/public",
    )
    parser.add_argument("--limit", type=int, default=1, help="number of conversations")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--include-captions", action="store_true")
    parser.add_argument("--max-questions", type=int, default=0, help="0 = all")
    parser.add_argument("--output", default="bench_results.json")
    parser.add_argument(
        "--tag",
        default="",
        help="storage tag: uses ./bench_<tag>.db and ./qdrant_bench_<tag>",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete the tagged storage before running (forces every Add to execute)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    conversations = load_conversations(data_dir)[: args.limit]
    questions = load_questions(data_dir)
    questions_by_sample = group_questions_by_sample(questions)

    init_db()
    print(f"conversations={len(conversations)}  questions_total={len(questions)}")

    db = SessionLocal()
    results = []
    overall = defaultdict(float)
    per_category = defaultdict(lambda: defaultdict(float))
    totals = {"questions": 0, "chunks": 0, "skipped_empty": 0}

    try:
        for conv in conversations:
            sid = conv["sample_id"]
            print(f"\n=== conversation {sid} ===")
            msgs = build_flat_messages(conv, include_captions=args.include_captions)
            totals["skipped_empty"] += conv["message_count"] - len(msgs)
            chunks = chunk_messages(msgs)
            chunk_map = {}
            for i, chunk in enumerate(chunks):
                req_id = f"{sid}:chunk-{i:03d}"
                chunk_map[req_id] = [m["dia_id"] for m in chunk]
                add_req = AddRequest(
                    request_id=req_id,
                    messages=[
                        Message(role=m["role"], content=m["content"], timestamp=m["ts_ms"])
                        for m in chunk
                    ],
                    user_id=sid,
                    session_id=f"{sid}:sess",
                )
                t0 = time.time()
                add_memory(add_req)
                totals["chunks"] += 1
                print(
                    f"  add {i + 1}/{len(chunks)} ok in {time.time() - t0:.1f}s"
                    f"  (messages={len(chunk)})"
                )

            unit_map = {
                u.id: u.source_request_id
                for u in db.query(MemoryUnit).filter(MemoryUnit.user_id == sid).all()
            }
            chunks_stored = (
                db.query(RawChunk).filter(RawChunk.user_id == sid).count()
            )
            print(f"  memory units indexed: {len(unit_map)}")
            if chunks_stored < len(chunks):
                raise RuntimeError(
                    f"storage was not fresh for {sid}: {chunks_stored}/{len(chunks)} "
                    "chunks persisted, so some Adds were deduplicated. "
                    "Re-run with --fresh (or a new --tag) to measure Add-side changes."
                )

            qs = questions_by_sample.get(sid, [])
            if args.max_questions:
                qs = qs[: args.max_questions]
            for qi, q in enumerate(qs):
                gold = normalize_evidence(q.get("evidence", []))
                if not gold:
                    overall["no_evidence"] += 1
                    continue
                search_req = SearchRequest(
                    query=q["question"],
                    user_id=sid,
                    top_k=args.top_k,
                )
                t0 = time.time()
                resp = search_memory(search_req)
                metrics = evaluate_question(
                    resp.data, gold, unit_map, chunk_map
                )
                cat = str(q.get("category", "?"))
                overall["n"] += 1
                overall["hit"] += metrics["hit"]
                overall["coverage"] += metrics["coverage"]
                overall["mrr"] += metrics["mrr"]
                per_category[cat]["n"] += 1
                per_category[cat]["hit"] += metrics["hit"]
                per_category[cat]["coverage"] += metrics["coverage"]
                per_category[cat]["mrr"] += metrics["mrr"]
                results.append(
                    {
                        "qa_id": q["qa_id"],
                        "category": cat,
                        "question": q["question"],
                        "gold": q.get("answer"),
                        "evidence": q.get("evidence"),
                        **metrics,
                        "top5": [
                            {"content": it.content, "score": it.score}
                            for it in resp.data[:5]
                        ],
                    }
                )
                if (qi + 1) % 10 == 0 or qi + 1 == len(qs):
                    n = overall["n"]
                    print(
                        f"  search {qi + 1}/{len(qs)}  "
                        f"hit@k={overall['hit'] / n:.3f}  "
                        f"cov={overall['coverage'] / n:.3f}  "
                        f"mrr={overall['mrr'] / n:.3f}  "
                        f"(last {time.time() - t0:.1f}s)"
                    )
    finally:
        db.close()

    n = overall["n"] or 1
    print("\n=== summary ===")
    print(f"storage_tag={STORAGE_TAG}")
    print(f"questions={overall['n']}  chunks={totals['chunks']}  "
          f"skipped_empty_msgs={totals['skipped_empty']}")
    print(f"hit@{args.top_k}      = {overall['hit'] / n:.4f}")
    print(f"coverage@k = {overall['coverage'] / n:.4f}")
    print(f"mrr        = {overall['mrr'] / n:.4f}")
    for cat in sorted(per_category):
        c = per_category[cat]
        cn = c["n"] or 1
        print(
            f"  cat {cat}: n={c['n']}  hit={c['hit'] / cn:.3f}  "
            f"cov={c['coverage'] / cn:.3f}  mrr={c['mrr'] / cn:.3f}"
        )

    misses = [r for r in results if not r["hit"]][:3]
    if misses:
        print("\n=== first misses (debug) ===")
        for m in misses:
            print(f"[{m['qa_id']}] {m['question']}")
            print(f"  gold evidence: {m['evidence']}")
            for t in m["top5"]:
                print(f"    {t['score']:.3f}  {t['content'][:80]}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "summary": {
                    "questions": overall["n"],
                    "hit_at_k": overall["hit"] / n,
                    "coverage_at_k": overall["coverage"] / n,
                    "mrr": overall["mrr"] / n,
                    "per_category": {
                        c: {
                            "n": v["n"],
                            "hit": v["hit"] / (v["n"] or 1),
                            "coverage": v["coverage"] / (v["n"] or 1),
                            "mrr": v["mrr"] / (v["n"] or 1),
                        }
                        for c, v in sorted(per_category.items())
                    },
                },
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nresults written to {args.output}")


if __name__ == "__main__":
    main()
