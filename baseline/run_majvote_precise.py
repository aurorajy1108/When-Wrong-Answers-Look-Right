#!/usr/bin/env python3
"""
B4-Generic: Call-matched majority vote with the precision-first prompt.

Each of the N=11 calls uses the SAME per-call prompt as the first round of
the homogeneous debate pipeline (_GENERIC_INSTRUCTIONS + _GENERIC_INITIAL_TEMPLATE).
Verdict: ≥ceil(N×0.6) support votes.  No multi-round exchange.

Purpose: isolate prompt quality from debate structure.

Comparison table (target):
  B4          (simple prompt   + vote N=11)  → FP=15.7   [existing]
  B4-Generic  (precision prompt + vote N=11)  → FP=?      [this run]
  Hom. Debate (precision prompt + debate ~11) → FP=5.0    [existing]
  Default     (het. prompts     + debate ~11) → FP=3.0    [existing]

If B4-Generic ≈ B4      → prompt quality NOT a confound.
If B4-Generic ≈ Hom     → debate structure adds little beyond prompt.
If B4-Generic is between → both contribute independently.

Usage (smoke test — 2 problems):
    cd /path/to/ECE399
    python3 baseline/run_majvote_precise.py \
        --problem-ids 26,51 \
        --output-dir baseline/majvote_precise__smoke

Full run:
    python3 baseline/run_majvote_precise.py \
        --output-dir baseline/majvote_precise__single_run \
        --workers 10
"""

import argparse
import json
import math
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

from debate_pipeline_stance_only.pipeline import call_llm, _force_stance, _strip_problem, _extract_json
from debate_pipeline_exchange.pipeline_debate_uniform import _GENERIC_INSTRUCTIONS, _GENERIC_INITIAL_TEMPLATE
from debate_pipeline_exchange.prompts_debate_ours import INITIAL_OUTPUT_SCHEMA


# ---------------------------------------------------------------------------
# B4-Generic judge
# ---------------------------------------------------------------------------

def b4_generic_judge(
    problem: str,
    candidate_answer: str,
    candidate_reasoning: str,
    n: int = 11,
    threshold_frac: float = 0.6,
) -> dict:
    """
    N independent calls with the precision-first prompt, majority vote.

    Each call uses a tag prefix [b4gen_n{n}_run={i}] so that the LLM cache
    stores N distinct entries per variant — identical to how call_matched_majority_judge
    separates its 11 calls with [cm_n11_run={i}] prefixes.
    """
    threshold = math.ceil(n * threshold_frac)

    # The per-call user prompt is identical to hom-debate round-1.
    # agent_id/agent_name are cosmetic but kept for prompt parity.
    base_prompt = _GENERIC_INITIAL_TEMPLATE.format(
        agent_id=1,
        agent_name="Mathematical Judge",
        problem=_strip_problem(problem),
        candidate_answer=candidate_answer,
        candidate_reasoning=candidate_reasoning or "(none)",
        output_schema=INITIAL_OUTPUT_SCHEMA,
    )

    votes, assessment_types, reasonings = [], [], []
    for run_idx in range(n):
        # Prefix distinguishes cache entries across the N calls.
        keyed = f"[b4gen_n{n}_run={run_idx}]\n{base_prompt}"
        raw = call_llm(
            keyed,
            tag=None,  # tag already embedded in prompt string for cache key
            instructions=_GENERIC_INSTRUCTIONS,
            effort="low",
        )
        parsed = _extract_json(raw) or {}
        stance = parsed.get("stance", "")
        if stance not in {"support", "oppose"}:
            stance = _force_stance(raw)
        atype = str(parsed.get("assessment_type", "")).strip()
        votes.append(stance)
        assessment_types.append(atype)
        reasonings.append(raw.strip()[:400])

    support_count = votes.count("support")
    # Ungated verdict: stance majority only
    if support_count >= threshold:
        prediction_ungated = "correct"
    else:
        prediction_ungated = "wrong"

    # Gated verdict: also require ≥threshold answer_supported assessments
    answer_supported_count = sum(1 for a in assessment_types if a == "answer_supported")
    if support_count >= threshold and answer_supported_count >= threshold:
        prediction_gated = "correct"
    else:
        prediction_gated = "wrong"

    return {
        "prediction": prediction_ungated,       # primary (ungated) verdict
        "prediction_gated": prediction_gated,   # secondary (gated) verdict
        "votes": votes,
        "assessment_types": assessment_types,
        "support_count": support_count,
        "answer_supported_count": answer_supported_count,
        "threshold": threshold,
        "n_calls": n,
        "reasonings": reasonings,
    }


# ---------------------------------------------------------------------------
# Dataset helpers (identical to other runners)
# ---------------------------------------------------------------------------

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


def compute_metrics(results, pred_key="pred"):
    tp = fp = tn = fn = 0
    for row in results:
        t, p = row["true"], row[pred_key]
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


def compute_problem_metrics(results, pred_key="pred"):
    grouped = defaultdict(list)
    for r in results:
        grouped[r["problem_id"]].append(r)
    correct = sum(
        1 for rows in grouped.values()
        if all(r["true"] == r[pred_key] for r in rows)
    )
    return {"problems_correct": correct, "problems_total": len(grouped),
            "problem_accuracy": round(correct/max(1,len(grouped)),4)}


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--problem-ids", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n-calls", type=int, default=11)
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

    details_path = output_dir / f"details_b4gen_{tag}.json"
    results_path = output_dir / f"results_b4gen_{tag}.json"

    results, details = [], []
    lock = threading.Lock()
    total = len(dataset)
    t0 = time.time()

    def _flush():
        with lock:
            write_json(details_path, details)
            m = compute_metrics(results, "pred")
            mg = compute_metrics(results, "pred_gated")
            pm = compute_problem_metrics(results, "pred")
            pmg = compute_problem_metrics(results, "pred_gated")
            write_json(results_path, {
                "ablation": "b4_generic_precision_prompt",
                "n_calls": args.n_calls,
                "threshold": math.ceil(args.n_calls * 0.6),
                "metrics_ungated": m,
                "metrics_gated": mg,
                "problem_level_ungated": pm,
                "problem_level_gated": pmg,
            })

    def _process(item):
        out = b4_generic_judge(
            problem=item["problem"],
            candidate_answer=item["candidate_answer"],
            candidate_reasoning=item.get("reasoning", ""),
            n=args.n_calls,
        )
        result = {
            "true": item["label"],
            "pred": out["prediction"],
            "pred_gated": out["prediction_gated"],
            "problem_id": item["problem_id"],
            "variant_id": item["variant_id"],
        }
        detail = {
            "problem_id": item["problem_id"],
            "variant_id": item["variant_id"],
            "true_label": item["label"],
            "prediction": out["prediction"],
            "prediction_gated": out["prediction_gated"],
            "votes": out["votes"],
            "assessment_types": out["assessment_types"],
            "support_count": out["support_count"],
            "answer_supported_count": out["answer_supported_count"],
            "threshold": out["threshold"],
        }
        return result, detail

    def _timeout_record(item, msg):
        result = {"true": item["label"], "pred": "wrong", "pred_gated": "wrong",
                  "problem_id": item["problem_id"], "variant_id": item["variant_id"]}
        detail = {**result, "votes": [], "assessment_types": [],
                  "support_count": 0, "answer_supported_count": 0, "threshold": 0}
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
                gok = "✓" if r["pred_gated"] == item["label"] else "✗"
                print(f"[{idx}/{total}] P{item['problem_id']} v{item['variant_id']} "
                      f"-> ungated:{r['pred']}[{ok}] gated:{r['pred_gated']}[{gok}]")
    else:
        # Sequential path: each variant has N serial calls; use a generous timeout.
        # ThreadPoolExecutor.shutdown(wait=True) blocks until the worker completes
        # even when future.result(timeout=T) raises, so the timeout only affects
        # what result is recorded, not how long we wait.  Use workers>=2 to get
        # genuine parallel execution with real timeouts.
        for idx, item in enumerate(dataset, 1):
            try:
                with ThreadPoolExecutor(max_workers=1) as ex:
                    r, d = ex.submit(_process, item).result(timeout=args.variant_timeout)
            except Exception as e:
                r, d = _timeout_record(item, str(e))
            results.append(r); details.append(d)
            _flush()
            ok = "✓" if r["pred"] == item["label"] else "✗"
            gok = "✓" if r["pred_gated"] == item["label"] else "✗"
            print(f"[{idx}/{total}] P{item['problem_id']} v{item['variant_id']} "
                  f"-> ungated:{r['pred']}[{ok}] gated:{r['pred_gated']}[{gok}]")

    _flush()
    m  = compute_metrics(results, "pred")
    mg = compute_metrics(results, "pred_gated")
    pm  = compute_problem_metrics(results, "pred")
    pmg = compute_problem_metrics(results, "pred_gated")
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min  ({total} variants, N={args.n_calls} calls each)")
    print(f"Ungated  — Prec:{m['precision']:.3f}  Rec:{m['recall']:.3f}  "
          f"FP:{m['confusion']['FP']}  ProbAcc:{pm['problem_accuracy']:.3f}")
    print(f"Gated    — Prec:{mg['precision']:.3f}  Rec:{mg['recall']:.3f}  "
          f"FP:{mg['confusion']['FP']}  ProbAcc:{pmg['problem_accuracy']:.3f}")
    print(f"Details -> {details_path}")


if __name__ == "__main__":
    main()
