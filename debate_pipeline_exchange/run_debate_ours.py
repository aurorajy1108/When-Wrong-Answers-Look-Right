#!/usr/bin/env python3
"""Run the exchange-style debate pipeline on the dataset."""

import argparse
import json
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET = REPO_ROOT / "dataset" / "final_dataset.json"

sys.path.insert(0, str(REPO_ROOT))

from debate_pipeline_exchange.pipeline_debate_ours import run_debate


def load_dataset(path):
    dataset_path = Path(path)
    if not dataset_path.is_absolute():
        cwd_path = Path.cwd() / dataset_path
        script_relative_path = SCRIPT_DIR / dataset_path
        dataset_path = cwd_path if cwd_path.exists() else script_relative_path
    with open(dataset_path, "r") as f:
        return json.load(f)["dataset"]


def flatten_variants(dataset):
    flat = []
    for entry in dataset:
        for variant in entry["variants"]:
            flat.append({
                "problem_id": entry["problem_id"],
                "variant_id": variant["variant_id"],
                "problem": entry["problem"],
                "candidate_answer": variant["candidate_answer"],
                "label": variant["label"],
                "reasoning": variant.get("reasoning", ""),
            })
    return flat


def compute_metrics(results):
    tp = fp = tn = fn = 0
    for row in results:
        truth = row["true"]
        pred = row["pred"]
        if truth == "correct" and pred == "correct":
            tp += 1
        elif truth == "wrong" and pred == "wrong":
            tn += 1
        elif truth == "wrong" and pred == "correct":
            fp += 1
        elif truth == "correct" and pred == "wrong":
            fn += 1
    total = max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "accuracy": round((tp + tn) / total, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "total": tp + tn + fp + fn,
    }


def compute_problem_level_metrics(results):
    grouped = defaultdict(list)
    for row in results:
        grouped[row["problem_id"]].append(row)
    correct = sum(1 for rows in grouped.values() if all(r["true"] == r["pred"] for r in rows))
    total = len(grouped)
    return {
        "problem_accuracy": round(correct / max(1, total), 4),
        "problems_correct": correct,
        "problems_total": total,
    }


def write_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_summary(summary_path: Path, *, agents: int, rounds: int, results):
    metrics = compute_metrics(results)
    problem_level = compute_problem_level_metrics(results)
    write_json(summary_path, {
        "agents": agents,
        "rounds": rounds,
        "metrics": metrics,
        "problem_level": {
            "accuracy": problem_level["problem_accuracy"],
            "correct": problem_level["problems_correct"],
            "total": problem_level["problems_total"],
        },
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--problem-ids", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--variant-timeout", type=float, default=180.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dataset = load_dataset(args.dataset)
    ids = None
    if args.problem_ids:
        ids = set(int(x.strip()) for x in args.problem_ids.split(",") if x.strip())
        raw_dataset = [entry for entry in raw_dataset if entry["problem_id"] in ids]
    elif args.sample is not None:
        raw_dataset = raw_dataset[:args.sample]

    dataset = flatten_variants(raw_dataset)
    results = []
    details = []
    total = len(dataset)
    t0 = time.time()
    lock = threading.Lock()

    tag = "full" if not args.sample and not args.problem_ids else (
        f"sample{args.sample}" if args.sample is not None else f"ids{'_'.join(str(i) for i in sorted(ids))}"
    )
    details_path = output_dir / f"details_exchange_{tag}.json"
    summary_path = output_dir / f"results_exchange_{tag}.json"
    progress_path = output_dir / f"progress_exchange_{tag}.jsonl"

    def _flush():
        with lock:
            write_json(details_path, details)
            write_summary(summary_path, agents=args.agents, rounds=args.rounds, results=results)

    def _process_item(item):
        progress_events = []

        def _hook(event):
            event = dict(event)
            event["problem_id"] = item["problem_id"]
            event["variant_id"] = item["variant_id"]
            event["ts"] = time.time()
            progress_events.append(event)
            if event.get("event") == "round_start":
                print(f"\n[P{item['problem_id']} v{item['variant_id']}] round {event['round']} start", flush=True)
            elif event.get("event") == "agent_output":
                print(
                    f"\n[P{item['problem_id']} v{item['variant_id']}] r{event['round']} a{event['agent_id']} "
                    f"{event.get('stance')} {event.get('assessment_type')} {event.get('evidence_grade')}",
                    flush=True,
                )

        out = run_debate(
            problem=item["problem"],
            candidate_answer=item["candidate_answer"],
            candidate_reasoning=item.get("reasoning", ""),
            num_agents=args.agents,
            max_rounds=args.rounds,
            progress_hook=_hook,
        )
        result = {
            "true": item["label"],
            "pred": out["final_verdict"],
            "problem_id": item["problem_id"],
            "variant_id": item["variant_id"],
        }
        detail = {
            "problem_id": item["problem_id"],
            "variant_id": item["variant_id"],
            "candidate_answer": item["candidate_answer"],
            "true_label": item["label"],
            "prediction": out["final_verdict"],
            "consensus_type": out["consensus_type"],
            "vote_counts": out["vote_counts"],
            "first_round_counts": out["first_round_counts"],
            "final_round_counts": out["final_round_counts"],
            "first_round_consensus": out["first_round_consensus"],
            "final_round_consensus": out["final_round_consensus"],
            "stop_round": out["stop_round"],
            "stances": out["stances"],
            "rounds": out["rounds"],
            "round_vote_counts": out["round_vote_counts"],
            "exchanges": out["exchanges"],
            "progress_events": progress_events,
        }
        return result, detail

    def _timeout_record(item, message):
        result = {
            "true": item["label"],
            "pred": "wrong",
            "problem_id": item["problem_id"],
            "variant_id": item["variant_id"],
        }
        detail = {
            "problem_id": item["problem_id"],
            "variant_id": item["variant_id"],
            "candidate_answer": item["candidate_answer"],
            "true_label": item["label"],
            "prediction": "wrong",
            "consensus_type": "timeout_partial",
            "vote_counts": {"correct": 0, "wrong": 0},
            "first_round_counts": {"correct": 0, "wrong": 0},
            "final_round_counts": {"correct": 0, "wrong": 0},
            "first_round_consensus": "timeout",
            "final_round_consensus": "timeout",
            "stop_round": 0,
            "stances": [],
            "rounds": [],
            "round_vote_counts": [],
            "exchanges": [],
            "progress_events": [{
                "event": "timeout_or_error",
                "message": message,
                "problem_id": item["problem_id"],
                "variant_id": item["variant_id"],
                "ts": time.time(),
            }],
        }
        return result, detail

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(_process_item, item): item for item in dataset}
            for idx, future in enumerate(as_completed(future_map), 1):
                item = future_map[future]
                try:
                    result, detail = future.result(timeout=args.variant_timeout)
                except Exception as e:
                    result, detail = _timeout_record(item, f"{type(e).__name__}: {e}")
                results.append(result)
                details.append(detail)
                with open(progress_path, "a", encoding="utf-8") as pf:
                    pf.write(json.dumps({
                        "problem_id": item["problem_id"],
                        "variant_id": item["variant_id"],
                        "pred": result["pred"],
                        "true": item["label"],
                        "consensus_type": detail["consensus_type"],
                    }, ensure_ascii=False) + "\n")
                _flush()
                ok = "✓" if result["pred"] == item["label"] else "✗"
                print(f"[{idx}/{total}] P{item['problem_id']} v{item['variant_id']} -> {result['pred']} [{ok}]")
    else:
        for idx, item in enumerate(dataset, 1):
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_process_item, item)
                    result, detail = future.result(timeout=args.variant_timeout)
            except Exception as e:
                result, detail = _timeout_record(item, f"{type(e).__name__}: {e}")
            results.append(result)
            details.append(detail)
            with open(progress_path, "a", encoding="utf-8") as pf:
                pf.write(json.dumps({
                    "problem_id": item["problem_id"],
                    "variant_id": item["variant_id"],
                    "pred": result["pred"],
                    "true": item["label"],
                    "consensus_type": detail["consensus_type"],
                }, ensure_ascii=False) + "\n")
            _flush()
            ok = "✓" if result["pred"] == item["label"] else "✗"
            print(f"[{idx}/{total}] P{item['problem_id']} v{item['variant_id']} -> {result['pred']} [{ok}]")
    _flush()

    elapsed = time.time() - t0
    print(f"Total time: {elapsed/60:.1f} min")
    print(f"Details -> {details_path}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
