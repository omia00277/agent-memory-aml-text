"""Multi-dataset retrieval self-test against the real memory pipeline.

Supports:
- LoCoMo-Refined (exact MRR via dia_ids)
- LongMemEval-S (approximate hit via answer substring matching)

Usage (from project root):
    python bench/run_multi_eval.py --datasets locomorefined longmemevals --sample-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("QDRANT_PATH", "./qdrant_bench_multi")
os.environ.setdefault("DATABASE_URL", "sqlite:///./bench_multi.db")
os.environ.setdefault("ENABLE_LLM_CONSOLIDATION", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.database import SessionLocal, init_db
from app.memory_service import add_memory, search_memory
from app.schemas import AddRequest, Message, SearchRequest

from bench.locomo_loader import (
    build_flat_messages,
    chunk_messages,
    group_questions_by_sample,
    load_conversations,
    load_questions,
)


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def answer_in_results(answer: str, items: list) -> bool:
    """Approximate check: is the answer (or a significant token) present in retrieved items."""
    answer_norm = normalize_text(answer)
    if not answer_norm:
        return False
    answer_tokens = [t for t in answer_norm.split() if len(t) > 2]
    for item in items:
        content_norm = normalize_text(item.content)
        # Direct substring
        if answer_norm in content_norm:
            return True
        # Majority of non-trivial tokens present
        if answer_tokens:
            hits = sum(1 for t in answer_tokens if t in content_norm)
            if hits >= max(1, len(answer_tokens) * 0.6):
                return True
    return False


def run_locomo_refined(sample_ratio: float, top_k: int):
    db = SessionLocal()
    data_dir = Path("LoCoMo_refined-main/LoCoMo_refined-main/data/public")
    if not data_dir.exists():
        print(f"[LoCoMo-Refined] data dir not found: {data_dir}")
        return None

    conversations = load_conversations(data_dir)
    questions = load_questions(data_dir)
    questions_by_sample = group_questions_by_sample(questions)

    # Sample conversations
    n_sample = max(1, int(len(conversations) * sample_ratio))
    conversations = conversations[:n_sample]

    overall = defaultdict(float)
    per_category = defaultdict(lambda: defaultdict(float))
    totals = {"questions": 0, "chunks": 0}

    for conv in conversations:
        sid = conv["sample_id"]
        print(f"\n[LoCoMo-Refined] conversation {sid}")
        msgs = build_flat_messages(conv)
        chunks = chunk_messages(msgs)
        chunk_map = {}

        for i, chunk in enumerate(chunks):
            req = AddRequest(
                request_id=f"{sid}-chunk-{i}",
                messages=[Message(role=m["role"], content=m["content"], timestamp=m.get("timestamp", 0)) for m in chunk],
                user_id=sid,
                session_id=sid,
            )
            try:
                add_memory(req)
                totals["chunks"] += 1
                print(f"  add {i + 1}/{len(chunks)} ok")
            except Exception as e:
                print(f"  add {i + 1}/{len(chunks)} failed: {e}")

        # Build chunk -> dia_ids map for evaluation
        # Need to reconstruct which dias were in each chunk
        for i, chunk in enumerate(chunks):
            dia_ids = set()
            for m in chunk:
                # messages from build_flat_messages have dia_id
                dia_id = m.get("dia_id")
                if dia_id:
                    dia_ids.add(dia_id)
            chunk_map[f"{sid}-chunk-{i}"] = dia_ids

        qs = questions_by_sample.get(sid, [])
        for qi, q in enumerate(qs):
            gold = set()
            for ev in q.get("evidence", []):
                for part in str(ev).split(";"):
                    part = part.strip()
                    if part:
                        gold.add(part)
            if not gold:
                continue

            try:
                t0 = time.time()
                result = search_memory(SearchRequest(query=q["question"], user_id=sid, top_k=top_k))
                latency = time.time() - t0
                items = result.data

                first_hit_rank = None
                covered = set()
                for rank, item in enumerate(items, start=1):
                    # item.id is uuid; map back to source chunk via DB
                    from app.database import MemoryUnit as DBMemoryUnit
                    unit = db.query(DBMemoryUnit).filter(DBMemoryUnit.id == item.id).first()
                    src = unit.source_request_id if unit else None
                    dias = chunk_map.get(src, set())
                    overlap = dias & gold
                    if overlap:
                        covered |= overlap
                        if first_hit_rank is None:
                            first_hit_rank = rank

                hit = bool(covered)
                coverage = len(covered) / len(gold) if gold else 0.0
                mrr = 1.0 / first_hit_rank if first_hit_rank else 0.0
                cat = str(q.get("category", "unknown"))

                overall["hit"] += hit
                overall["coverage"] += coverage
                overall["mrr"] += mrr
                overall["latency"] += latency
                overall["n"] += 1

                per_category[cat]["hit"] += hit
                per_category[cat]["coverage"] += coverage
                per_category[cat]["mrr"] += mrr
                per_category[cat]["n"] += 1

                totals["questions"] += 1
                if (qi + 1) % 10 == 0:
                    n = overall["n"]
                    print(
                        f"  search {qi + 1}/{len(qs)}  "
                        f"hit={overall['hit'] / n:.3f}  "
                        f"cov={overall['coverage'] / n:.3f}  "
                        f"mrr={overall['mrr'] / n:.3f}  "
                        f"lat={latency:.1f}s"
                    )
            except Exception as e:
                print(f"  search {qi + 1}/{len(qs)} failed: {e}")

    n = overall["n"] or 1
    summary = {
        "dataset": "locomo-refined",
        "questions": totals["questions"],
        "chunks": totals["chunks"],
        "hit_at_k": overall["hit"] / n,
        "coverage_at_k": overall["coverage"] / n,
        "mrr": overall["mrr"] / n,
        "avg_latency": overall["latency"] / n,
        "per_category": {
            cat: {
                "n": c["n"],
                "hit": c["hit"] / (c["n"] or 1),
                "coverage": c["coverage"] / (c["n"] or 1),
                "mrr": c["mrr"] / (c["n"] or 1),
            }
            for cat, c in per_category.items()
        },
    }
    db.close()
    return summary


def stream_lme_items(path: Path, sample_ratio: float):
    """Stream LongMemEval-S items and yield sampled ones without loading full file."""
    count = 0
    sampled = 0
    with open(path) as f:
        ch = f.read(1)
        while ch and ch != "[":
            ch = f.read(1)
        while True:
            ch = f.read(1)
            while ch and ch in " \n\r\t,":
                ch = f.read(1)
            if not ch or ch == "]":
                break
            if ch != "{":
                continue
            obj_text = ch
            depth = 1
            in_string = False
            escape = False
            while depth > 0:
                ch = f.read(1)
                if not ch:
                    break
                obj_text += ch
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
            count += 1
            if count % int(1 / sample_ratio) == 0:
                sampled += 1
                yield json.loads(obj_text)


def run_longmemeval_s(sample_ratio: float, top_k: int):
    path = Path("data_external/longmemeval_s_cleaned.json")
    if not path.exists():
        print(f"[LongMemEval-S] data file not found: {path}")
        return None

    total_adds = 0
    total_searches = 0
    add_success = 0
    search_success = 0
    hit_count = 0
    total_latency_add = 0.0
    total_latency_search = 0.0
    total_items = 0

    for item in stream_lme_items(path, sample_ratio):
        total_items += 1
        qid = item["question_id"]
        question = item["question"]
        answer = item["answer"]
        sessions = item.get("haystack_sessions", [])

        # Flatten all sessions into one message list
        all_messages = []
        base_ts = 1704067200000
        for si, session in enumerate(sessions):
            for mi, m in enumerate(session):
                all_messages.append(
                    Message(
                        role=m.get("role", "user"),
                        content=m.get("content", ""),
                        timestamp=base_ts + si * 3600000 + mi * 60000,
                    )
                )

        if not all_messages:
            continue

        # Add all messages for this question in one request
        try:
            t0 = time.time()
            add_req = AddRequest(
                request_id=f"lme-{qid}",
                messages=all_messages,
                user_id=f"lme-{qid}",
                session_id=f"lme-{qid}",
            )
            add_memory(add_req)
            add_success += 1
            total_latency_add += time.time() - t0
            total_adds += 1
        except Exception as e:
            print(f"[LongMemEval-S] {qid} add failed: {e}")
            total_adds += 1
            continue

        # Search
        try:
            t0 = time.time()
            result = search_memory(SearchRequest(query=question, user_id=f"lme-{qid}", top_k=top_k))
            latency = time.time() - t0
            total_latency_search += latency
            total_searches += 1
            search_success += 1

            if answer_in_results(answer, result.data):
                hit_count += 1

            if total_items % 5 == 0:
                print(
                    f"[LongMemEval-S] {total_items} processed, "
                    f"add_ok={add_success}/{total_adds}, "
                    f"search_ok={search_success}/{total_searches}, "
                    f"hit={hit_count}/{total_searches}, "
                    f"avg_lat_add={total_latency_add/total_adds:.1f}s, "
                    f"avg_lat_search={total_latency_search/total_searches:.1f}s"
                )
        except Exception as e:
            print(f"[LongMemEval-S] {qid} search failed: {e}")
            total_searches += 1

    summary = {
        "dataset": "longmemeval-s",
        "questions": total_items,
        "add_success_rate": add_success / (total_adds or 1),
        "search_success_rate": search_success / (total_searches or 1),
        "approx_hit_at_k": hit_count / (total_searches or 1),
        "avg_add_latency": total_latency_add / (total_adds or 1),
        "avg_search_latency": total_latency_search / (total_searches or 1),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["locomorefined", "longmemevals"],
        choices=["locomorefined", "longmemevals"],
    )
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", default="bench_multi_results.json")
    args = parser.parse_args()

    init_db()
    results = {}

    for dataset in args.datasets:
        print(f"\n{'='*40}\nRunning {dataset}\n{'='*40}")
        if dataset == "locomorefined":
            results[dataset] = run_locomo_refined(args.sample_ratio, args.top_k)
        elif dataset == "longmemevals":
            results[dataset] = run_longmemeval_s(args.sample_ratio, args.top_k)

    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(json.dumps(results, indent=2, ensure_ascii=False))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
