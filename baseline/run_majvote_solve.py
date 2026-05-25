#!/usr/bin/env python3
"""
MajVote-Solve: N=11 independent solve calls with majority-vote answer selection.

Compute-matched control for MajVote-Verify (11 calls/variant via majority vote).
For each of 58 problems: call the model 11 times independently to solve from
scratch, extract the boxed answer from each, take the majority normalized answer.

Comparison target:
  Solve N=1    (single call)        → ProbAcc=39.1% (3-run avg)
  MajVote-Solve (this, N=11)        → ?
  MajVote-Verify (N=11 verify)      → ProbAcc=60.9%
  Debate (N≈37 total)               → ProbAcc=59.8%

Usage:
  cd /path/to/ECE399
  python3 baseline/run_majvote_solve.py --smoke --run-id 1   # 3-problem smoke test
  python3 baseline/run_majvote_solve.py --run-id 1 --workers 10
  python3 baseline/run_majvote_solve.py --run-id 2 --workers 10
  python3 baseline/run_majvote_solve.py --run-id 3 --workers 10
"""

import argparse
import atexit
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET = REPO_ROOT / "dataset" / "final_dataset.json"

# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------
def _load_dotenv(path=".env"):
    base_dir = str(SCRIPT_DIR)
    for candidate in [os.path.join(base_dir, path), os.path.join(base_dir, "..", path)]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip(); v = v.strip().strip('"').strip("'")
                    if k:
                        os.environ[k] = v
            return

_load_dotenv()

# ---------------------------------------------------------------------------
# LLM cache (per-run cache file, same SHA-256 scheme as run_solve_from_scratch_for_verify_vs_solve.py)
# ---------------------------------------------------------------------------
_cache: dict = {}
_cache_lock = threading.Lock()
_cache_dirty = 0
_cache_file: str = ""          # set in main()
_thread_local = threading.local()
_api_sem = None
_SAVE_EVERY = 5


def _get_client():
    from openai import OpenAI
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI()
    return _thread_local.client


def _get_sem(limit: int):
    global _api_sem
    if _api_sem is None:
        _api_sem = threading.BoundedSemaphore(max(1, limit))
    return _api_sem


def _load_cache(path: str):
    global _cache
    if os.path.exists(path):
        with open(path) as f:
            _cache = json.load(f)
        print(f"[cache] loaded {len(_cache)} entries from {path}")
    else:
        _cache = {}


def _flush_cache():
    global _cache_dirty
    with _cache_lock:
        if _cache_file:
            with open(_cache_file, "w") as f:
                json.dump(_cache, f, indent=2)
            _cache_dirty = 0


def _extract_text(resp) -> str:
    text = (getattr(resp, "output_text", None) or "").strip()
    if text:
        return text
    chunks = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text":
                v = (getattr(content, "text", None) or "").strip()
                if v:
                    chunks.append(v)
    return "\n".join(chunks).strip()


def call_llm(prompt: str, instructions: str, effort: str, cache_salt: str,
             workers: int = 10) -> str:
    global _cache_dirty
    key_input = "\x00".join([
        "model=gpt-5.2", f"effort={effort}", "max_tokens=16000",
        f"instructions={instructions}", f"prompt={prompt}", f"salt={cache_salt}",
    ])
    cache_key = hashlib.sha256(key_input.encode()).hexdigest()

    with _cache_lock:
        if cache_key in _cache:
            print(".", end="", flush=True)
            return _cache[cache_key]

    print("*", end="", flush=True)
    t0 = time.time()
    kwargs = dict(
        model="gpt-5.2",
        input=prompt,
        reasoning={"effort": effort},
        max_output_tokens=16000,
    )
    if instructions:
        kwargs["instructions"] = instructions
    with _get_sem(workers):
        resp = _get_client().responses.create(**kwargs)
    elapsed = time.time() - t0
    result = _extract_text(resp)
    print(f"({elapsed:.0f}s)", end="", flush=True)

    with _cache_lock:
        _cache[cache_key] = result
        _cache_dirty += 1
        should_save = _cache_dirty >= _SAVE_EVERY
    if should_save:
        _flush_cache()
    return result


# ---------------------------------------------------------------------------
# Answer extraction (identical to run_solve_from_scratch_for_verify_vs_solve.py)
# ---------------------------------------------------------------------------
_PROBLEM_SUFFIX_RE = re.compile(r'\s*After solving the above problem.*$', re.DOTALL)

def _strip_problem(problem: str) -> str:
    return _PROBLEM_SUFFIX_RE.sub('', problem).strip()


def _extract_boxed_balanced(text: str):
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def _clean_answer(s: str) -> str:
    s = s.strip()
    s = re.sub(r'\\(?:text|mathrm)\{[^{}]*\}', '', s)
    if "=" in s:
        s = s.split("=")[-1].strip()
    for wrap in [("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")]:
        if s.startswith(wrap[0]) and s.endswith(wrap[1]):
            s = s[len(wrap[0]):-len(wrap[1])].strip()
    return s.strip()


def extract_boxed(text: str):
    raw = _extract_boxed_balanced(text)
    if raw:
        cleaned = _clean_answer(raw)
        if re.search(r"\d", cleaned):
            return cleaned
        return None
    for pat in [r"Final Answer\s*[:\-]\s*(.+)", r"Answer\s*[:\-]\s*(.+)"]:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            cleaned = _clean_answer(matches[-1].group(1).strip())
            if re.search(r"\d", cleaned):
                return cleaned
    return None


def normalize_answer(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r'\\(?:text|mathrm)\{([^}]+)\}', r'\1', s)
    s = s.replace(',', '').replace(' ', '').lower()
    try:
        return str(int(s))
    except ValueError:
        pass
    try:
        return f"{float(s):.6g}"
    except ValueError:
        pass
    return s


# ---------------------------------------------------------------------------
# Per-problem MajVote-Solve
# ---------------------------------------------------------------------------
_SOLVE_INSTRUCTIONS = (
    "Solve the problem. "
    "Write your final answer as \\boxed{answer}."
)

N_CALLS = 11


def majvote_solve_problem(problem: str, problem_id: int, run_id: int,
                           workers: int = 10) -> dict:
    """
    Make N_CALLS independent solve calls, extract answers, take majority vote.

    Cache salt format: mvsol-n11-r{run_id}-pid{problem_id}-c{call_idx}
    This ensures:
      - Each of the 11 calls has a distinct cache entry.
      - Different run_ids get different entries (for multi-run variance).
    """
    clean = _strip_problem(problem)
    prompt = f"Problem:\n{clean}\n\nSolve the problem."

    raw_answers = []     # extracted string or None
    norm_answers = []    # normalized string or None

    for call_idx in range(N_CALLS):
        salt = f"mvsol-n{N_CALLS}-r{run_id}-pid{problem_id}-c{call_idx}"
        output = call_llm(prompt, instructions=_SOLVE_INSTRUCTIONS,
                          effort="low", cache_salt=salt, workers=workers)
        ans = extract_boxed(output)
        raw_answers.append(ans)
        norm_answers.append(normalize_answer(ans) if ans is not None else None)

    # Majority vote: most common non-None normalized answer.
    # Ties (no single plurality) → no_consensus → counted as wrong.
    valid_norms = [a for a in norm_answers if a is not None]
    if not valid_norms:
        majority_norm = None
        vote_counts = {}
        consensus = "all_invalid"
    else:
        cnt = Counter(valid_norms)
        top_count = cnt.most_common(1)[0][1]
        top_answers = [a for a, c in cnt.items() if c == top_count]
        if len(top_answers) == 1:
            majority_norm = top_answers[0]
            consensus = "majority"
        else:
            majority_norm = None        # genuine tie → wrong
            consensus = "tie"
        vote_counts = dict(cnt)

    return {
        "raw_answers": raw_answers,
        "norm_answers": norm_answers,
        "vote_counts": vote_counts,
        "majority_norm": majority_norm,
        "consensus": consensus,
        "n_valid": len(valid_norms),
        "n_calls": N_CALLS,
    }


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def load_problems(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return data["dataset"]


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _cache_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--run-id", type=int, required=True, help="Run index (1/2/3)")
    parser.add_argument("--smoke", action="store_true", help="Only run first 3 problems")
    parser.add_argument("--problem-ids", default=None, help="Comma-separated problem IDs")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    # Output directory
    tag = "smoke" if args.smoke else f"r{args.run_id}"
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / f"mvsol_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-run cache to avoid cross-run collisions (each run uses unique salts anyway,
    # but a shared cache also works — using per-run cache for cleanliness)
    _cache_file = str(output_dir / "mvsol_cache.json")
    _load_cache(_cache_file)
    atexit.register(_flush_cache)

    problems = load_problems(args.dataset)
    if args.smoke:
        problems = problems[:3]
    elif args.problem_ids:
        ids = set(int(x) for x in args.problem_ids.split(",") if x.strip())
        problems = [p for p in problems if p["problem_id"] in ids]

    total = len(problems)
    print(f"\n{'='*60}")
    print(f"  MajVote-Solve  |  run_id={args.run_id}  |  N={N_CALLS}  |  problems={total}")
    print(f"  effort=low  |  workers={args.workers}  |  {'SMOKE' if args.smoke else 'FULL'}")
    print(f"{'='*60}\n")

    details = []
    results_lock = threading.Lock()

    def _do_problem(prob):
        pid = prob["problem_id"]
        gt = str(prob.get("ground_truth", "")).strip()
        r = majvote_solve_problem(prob["problem"], pid, args.run_id, workers=args.workers)
        maj = r["majority_norm"]
        correct = (maj is not None and normalize_answer(gt) != "" and
                   maj == normalize_answer(gt))
        return {
            "problem_id": pid,
            "ground_truth": gt,
            "ground_truth_norm": normalize_answer(gt),
            "majority_answer": r["majority_norm"],
            "correct": correct,
            "consensus": r["consensus"],
            "n_valid": r["n_valid"],
            "n_calls": r["n_calls"],
            "vote_counts": r["vote_counts"],
            "raw_answers": r["raw_answers"],
        }

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fmap = {ex.submit(_do_problem, p): p["problem_id"] for p in problems}
        for fut in as_completed(fmap):
            pid = fmap[fut]
            d = fut.result()
            completed += 1
            status = "✓" if d["correct"] else ("INVALID" if d["majority_answer"] is None else "✗")
            print(f"\n  [{completed}/{total}] P{pid} "
                  f"maj={d['majority_answer']!r}  gt={d['ground_truth']!r}  "
                  f"valid={d['n_valid']}/{N_CALLS}  [{status}]")
            with results_lock:
                details.append(d)
                # Flush intermediate results
                details_sorted = sorted(details, key=lambda x: x["problem_id"])
                write_json(output_dir / f"details_mvsol_{tag}.json", details_sorted)

    details.sort(key=lambda x: x["problem_id"])

    # Metrics
    n_total = len(details)
    n_correct = sum(1 for d in details if d["correct"])
    n_invalid = sum(1 for d in details if d["majority_answer"] is None)
    prob_acc = round(n_correct / max(1, n_total), 4)

    metrics = {
        "run_id": args.run_id,
        "n_calls_per_problem": N_CALLS,
        "problems_total": n_total,
        "problems_correct": n_correct,
        "problems_invalid": n_invalid,
        "problem_level_accuracy": prob_acc,
    }
    write_json(output_dir / f"results_mvsol_{tag}.json", metrics)

    elapsed_note = ""
    print(f"\n{'='*60}")
    print(f"  MajVote-Solve run {args.run_id} complete")
    print(f"  Problems: {n_total}  Correct: {n_correct}  Invalid: {n_invalid}")
    print(f"  Problem-level accuracy: {prob_acc:.1%}")
    print(f"{'='*60}")
    return metrics


if __name__ == "__main__":
    main()
