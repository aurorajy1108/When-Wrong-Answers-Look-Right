#!/usr/bin/env python3
"""
Experiment 5 — Homogeneous debate baseline.

Runs the debate pipeline where all 5 agents share IDENTICAL system prompts
(no persona differentiation). Tests whether heterogeneous agent roles are
responsible for the 59% structural gain over B4.

Expected: homogeneous debate ≈ B4 (call-matched majority vote).
If confirmed, validates that role heterogeneity drives the structural gain.

Usage (smoke test — 2 problems):
    cd /path/to/ECE399
    python3 debate_pipeline_exchange/run_debate_uniform.py \
        --problem-ids 26,51 \
        --output-dir debate_pipeline_exchange/smoke_homogeneous

Full run:
    python3 debate_pipeline_exchange/run_debate_uniform.py \
        --output-dir debate_pipeline_exchange/debate_uniform__single_run \
        --workers 10
"""

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

from debate_pipeline_exchange.pipeline_debate_uniform import run_debate_homogeneous


def load_dataset(path):
    with open(path) as f:
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
        t, p = row["true"], row["pred"]
        if t == "correct" and p == "correct": tp += 1
        elif t == "wrong"   and p == "wrong":   tn += 1
        elif t == "wrong"   and p == "correct": fp += 1
        elif t == "correct" and p == "wrong":   fn += 1
    total = max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 2*prec*rec / max(1e-8, prec+rec)
    return {"accuracy": round((tp+tn)/total,4), "precision": round(prec,4),
            "recall": round(rec,4), "f1": round(f1,4),
            "confusion": {"TP":tp,"TN":tn,"FP":fp,"FN":fn}}


def compute_problem_metrics(results):
    grouped = defaultdict(list)
    for r in results:
        grouped[r["problem_id"]].append(r)
    correct = sum(1 for rows in grouped.values() if all(r["true"]==r["pred"] for r in rows))
    return {"problems_correct": correct, "problems_total": len(grouped),
            "problem_accuracy": round(correct/max(1,len(grouped)),4)}


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--problem-ids", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--variant-timeout", type=float, default=180.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dataset = load_dataset(args.dataset)
    ids = None
    if args.problem_ids:
        ids = set(int(x.strip()) for x in args.problem_ids.split(",") if x.strip())
        raw_dataset = [e for e in raw_dataset if e["problem_id"] in ids]
    elif args.sample:
        raw_dataset = raw_dataset[:args.sample]

    dataset = flatten_variants(raw_dataset)
    tag = ("full" if not args.sample and not args.problem_ids
           else (f"sample{args.sample}" if args.sample
                 else f"ids{'_'.join(str(i) for i in sorted(ids))}"))

    details_path = output_dir / f"details_homogeneous_{tag}.json"
    results_path = output_dir / f"results_homogeneous_{tag}.json"

    results, details = [], []
    lock = threading.Lock()
    total = len(dataset)
    t0 = time.time()

    def _flush():
        with lock:
            write_json(details_path, details)
            m = compute_metrics(results)
            pm = compute_problem_metrics(results)
            write_json(results_path, {
                "ablation": "homogeneous_agents",
                "metrics": m,
                "problem_level": pm,
                "agents": 5,
                "rounds": 5,
            })

    def _process(item):
        out = run_debate_homogeneous(
            problem=item["problem"],
            candidate_answer=item["candidate_answer"],
            candidate_reasoning=item.get("reasoning", ""),
            num_agents=5,
            max_rounds=5,
        )
        result = {"true": item["label"], "pred": out["final_verdict"],
                  "problem_id": item["problem_id"], "variant_id": item["variant_id"]}
        detail = {
            "problem_id": item["problem_id"],
            "variant_id": item["variant_id"],
            "true_label": item["label"],
            "prediction": out["final_verdict"],
            "consensus_type": out["consensus_type"],
            "vote_counts": out["vote_counts"],
            "stop_round": out["stop_round"],
            "round_vote_counts": out["round_vote_counts"],
            "stances": out["stances"],
            "rounds": out["rounds"],
        }
        return result, detail

    def _timeout_record(item, msg):
        result = {"true": item["label"], "pred": "wrong",
                  "problem_id": item["problem_id"], "variant_id": item["variant_id"]}
        detail = {**result, "consensus_type": "error", "vote_counts": {},
                  "stop_round": 0, "round_vote_counts": [], "stances": [], "rounds": []}
        return result, detail

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fmap = {ex.submit(_process, item): item for item in dataset}
            for idx, future in enumerate(as_completed(fmap), 1):
                item = fmap[future]
                try:
                    r, d = future.result(timeout=args.variant_timeout)
                except Exception as e:
                    r, d = _timeout_record(item, str(e))
                results.append(r); details.append(d)
                _flush()
                ok = "✓" if r["pred"] == item["label"] else "✗"
                print(f"[{idx}/{total}] P{item['problem_id']} v{item['variant_id']} "
                      f"-> {r['pred']} [{ok}]")
    else:
        for idx, item in enumerate(dataset, 1):
            try:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    r, d = ex.submit(_process, item).result(timeout=args.variant_timeout)
            except Exception as e:
                r, d = _timeout_record(item, str(e))
            results.append(r); details.append(d)
            _flush()
            ok = "✓" if r["pred"] == item["label"] else "✗"
            print(f"[{idx}/{total}] P{item['problem_id']} v{item['variant_id']} "
                  f"-> {r['pred']} [{ok}]")

    _flush()
    m = compute_metrics(results)
    pm = compute_problem_metrics(results)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Precision: {m['precision']:.3f}  Recall: {m['recall']:.3f}  "
          f"FP: {m['confusion']['FP']}  FN: {m['confusion']['FN']}")
    print(f"Problem accuracy: {pm['problems_correct']}/{pm['problems_total']}")
    print(f"Details -> {details_path}")


if __name__ == "__main__":
    main()
