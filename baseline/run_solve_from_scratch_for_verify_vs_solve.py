#!/usr/bin/env python3
"""
H1 Solving Experiment: measure model zero-shot solving accuracy.

For each of 58 problems, ask gpt-5.2 to solve from scratch (no candidate
answer, no reasoning trace provided). Compare the extracted \\boxed{} answer
against the dataset ground truth.

This yields a "solving accuracy" to compare against the verification
accuracy reported in the paper (B0: 64.4% problem-level), providing
direct evidence for H1 (verification >= solving).

Usage:
  python run_solve_from_scratch_for_verify_vs_solve.py --smoke
  python run_solve_from_scratch_for_verify_vs_solve.py --run-id 1
  python run_solve_from_scratch_for_verify_vs_solve.py --run-id 2
  python run_solve_from_scratch_for_verify_vs_solve.py --run-id 3
  python run_solve_from_scratch_for_verify_vs_solve.py --run-id 1 --workers 6
"""

import json
import re
import os
import sys
import hashlib
import argparse
import shutil
import time
import atexit
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datetime import datetime


# ---------------------------------------------------------------------------
# .env loader (same pattern as run_single_judge_and_majvote_verify.py)
# ---------------------------------------------------------------------------
def _load_dotenv(path=".env"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [os.path.join(base_dir, path), os.path.join(base_dir, "..", path)]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k:
                        os.environ[k] = v
            return

_load_dotenv()


# ---------------------------------------------------------------------------
# LLM call with SHA-256-keyed cache (mirrors run_single_judge_and_majvote_verify.py)
# ---------------------------------------------------------------------------
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h1_solve_cache.json")
_cache: dict = {}
_cache_lock = threading.Lock()
_cache_dirty = 0
_thread_local = threading.local()
_api_semaphore = None
_CACHE_SAVE_EVERY = 5


def _get_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI()
    return _thread_local.client


def _get_semaphore(limit: int):
    global _api_semaphore
    if _api_semaphore is None:
        _api_semaphore = threading.BoundedSemaphore(max(1, limit))
    return _api_semaphore


def _load_cache():
    global _cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            _cache = json.load(f)

def _save_cache():
    with _cache_lock:
        with open(CACHE_FILE, "w") as f:
            json.dump(_cache, f, indent=2)
        global _cache_dirty
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


def call_llm(prompt: str, instructions: str = "", effort: str = "low",
             max_tokens: int = 16000, cache_salt: str = "",
             workers: int = 6) -> str:
    global _cache_dirty
    key_input = "\x00".join([
        "model=gpt-5.2", f"effort={effort}", f"max_tokens={max_tokens}",
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
        max_output_tokens=max_tokens,
    )
    if instructions:
        kwargs["instructions"] = instructions
    with _get_semaphore(workers):
        resp = _get_client().responses.create(**kwargs)
    elapsed = time.time() - t0
    result = _extract_text(resp)
    print(f"({elapsed:.0f}s)", end="", flush=True)

    with _cache_lock:
        _cache[cache_key] = result
        _cache_dirty += 1
        should_save = _cache_dirty >= _CACHE_SAVE_EVERY

    if should_save:
        _save_cache()
    return result


_load_cache()
atexit.register(_save_cache)


# ---------------------------------------------------------------------------
# Answer extraction (mirrors run_single_judge_and_majvote_verify.py)
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
# Solving prompts
# ---------------------------------------------------------------------------
_SOLVE_INSTRUCTIONS = (
    "Solve the problem. "
    "Write your final answer as \\boxed{answer}."
)

_SOLVE_RETRY_INSTRUCTIONS = (
    "Reply with exactly the final answer and nothing else: \\boxed{answer}"
)


def solve_problem(problem: str, run_id: int, problem_id: int,
                  workers: int = 6) -> dict:
    """Ask the model to solve problem from scratch; return result dict."""
    clean_problem = _strip_problem(problem)
    prompt = (
        f"Problem:\n{clean_problem}\n\n"
        "Solve the problem."
    )
    salt = f"h1-solve-run{run_id}-pid{problem_id}"
    output = call_llm(prompt, instructions=_SOLVE_INSTRUCTIONS,
                      effort="low", max_tokens=10000, cache_salt=salt, workers=workers)
    solved = extract_boxed(output)

    if solved is None:
        retry_salt = f"h1-solve-run{run_id}-pid{problem_id}-retry"
        output_retry = call_llm(
            prompt, instructions=_SOLVE_RETRY_INSTRUCTIONS,
            effort="low", max_tokens=512, cache_salt=retry_salt, workers=workers,
        )
        solved = extract_boxed(output_retry)
        if solved is not None:
            output = output_retry

    return {"solved_answer": solved, "reasoning": output.strip()}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_problems(dataset_path: str) -> list:
    with open(dataset_path) as f:
        data = json.load(f)
    return data["dataset"]   # list of {problem_id, problem, ground_truth, variants}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(details: list) -> dict:
    valid = [d for d in details if d["solved_answer"] is not None]
    invalid = len(details) - len(valid)
    correct = sum(1 for d in valid if d["correct"])
    total_valid = len(valid)
    accuracy = correct / max(1, total_valid)
    return {
        "problems_total": len(details),
        "problems_valid": total_valid,
        "problems_invalid": invalid,
        "problems_correct": correct,
        "problem_level_accuracy": round(accuracy, 4),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------
def run_experiment(dataset_path: str, run_id: int, smoke: bool,
                   output_dir: str, workers: int) -> dict:
    problems = load_problems(dataset_path)

    if smoke:
        problems = problems[:3]
        tag = f"smoke_r{run_id}"
    else:
        tag = f"r{run_id}"

    print(f"\n{'='*60}")
    print(f"  H1 Solve Experiment  |  run_id={run_id}  |  problems={len(problems)}")
    print(f"  effort=low  |  workers={workers}  |  {'SMOKE TEST' if smoke else 'FULL RUN'}")
    print(f"{'='*60}\n")

    details = []

    def _do_problem(prob: dict) -> dict:
        pid = prob["problem_id"]
        gt = str(prob["ground_truth"]).strip()
        result = solve_problem(prob["problem"], run_id, pid, workers=workers)
        solved = result["solved_answer"]
        correct = (
            normalize_answer(solved) == normalize_answer(gt)
            if solved is not None else False
        )
        return {
            "problem_id": pid,
            "ground_truth": gt,
            "solved_answer": solved,
            "correct": correct,
            "reasoning": result["reasoning"],
        }

    if workers <= 1:
        for i, prob in enumerate(problems, 1):
            pid = prob["problem_id"]
            print(f"  [{i}/{len(problems)}] pid={pid} ", end="", flush=True)
            d = _do_problem(prob)
            status = "OK" if d["correct"] else ("INVALID" if d["solved_answer"] is None else "WRONG")
            print(f" -> solved={d['solved_answer']!r}  gt={d['ground_truth']!r}  [{status}]")
            details.append(d)
    else:
        futures_map = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for prob in problems:
                fut = ex.submit(_do_problem, prob)
                futures_map[fut] = prob["problem_id"]
            for i, fut in enumerate(as_completed(futures_map), 1):
                pid = futures_map[fut]
                d = fut.result()
                status = "OK" if d["correct"] else ("INVALID" if d["solved_answer"] is None else "WRONG")
                print(f"  [{i}/{len(problems)}] pid={pid} -> solved={d['solved_answer']!r}  gt={d['ground_truth']!r}  [{status}]")
                details.append(d)
        # sort by problem_id for deterministic output
        details.sort(key=lambda x: x["problem_id"])

    print()
    metrics = compute_metrics(details)

    print(f"  Problems: {metrics['problems_total']}  |  "
          f"Valid: {metrics['problems_valid']}  |  "
          f"Invalid: {metrics['problems_invalid']}")
    print(f"  Correct:  {metrics['problems_correct']}/{metrics['problems_valid']} "
          f"= {metrics['problem_level_accuracy']:.1%} problem-level solving accuracy")

    # save
    os.makedirs(output_dir, exist_ok=True)
    details_path = os.path.join(output_dir, f"h1_solve_details_{tag}.json")
    results_path = os.path.join(output_dir, f"h1_solve_results_{tag}.json")
    cache_dest   = os.path.join(output_dir, "h1_solve_cache.json")

    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "smoke": smoke, **metrics}, f, indent=2)
    _save_cache()
    if os.path.exists(CACHE_FILE):
        shutil.copy2(CACHE_FILE, cache_dest)

    print(f"\n  Details  -> {details_path}")
    print(f"  Results  -> {results_path}")
    print(f"  Cache    -> {cache_dest}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description="H1 solving experiment: measure zero-shot solving accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_solve_from_scratch_for_verify_vs_solve.py --smoke
  python run_solve_from_scratch_for_verify_vs_solve.py --run-id 1
  python run_solve_from_scratch_for_verify_vs_solve.py --run-id 1 --workers 6
        """,
    )
    parser.add_argument("--dataset", "-d",
        default="../dataset/final_dataset.json")
    parser.add_argument("--run-id", type=int, default=1,
        help="Run index (1, 2, or 3). Affects cache salt so each run is independent.")
    parser.add_argument("--smoke", action="store_true",
        help="Smoke test: run only the first 3 problems.")
    parser.add_argument("--output-dir", default=None,
        help="Directory for output files (default: h1_solve_r{N}_YYYYMMDD/ next to this script)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log", action="store_true",
        help="Tee stdout to h1_solve_run.log in output-dir")
    args = parser.parse_args()

    datestamp = datetime.now().strftime("%Y%m%d")
    if args.output_dir:
        output_dir = args.output_dir
    elif args.smoke:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"h1_smoke_r{args.run_id}_{datestamp}",
        )
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"h1_solve_r{args.run_id}_{datestamp}",
        )
    os.makedirs(output_dir, exist_ok=True)

    log_file = None
    if args.log:
        log_path = os.path.join(output_dir, "h1_solve_run.log")
        log_file = open(log_path, "w", encoding="utf-8")

        class _Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, data):
                for f in self.files:
                    f.write(data)
            def flush(self):
                for f in self.files:
                    f.flush()

        sys.stdout = _Tee(sys.__stdout__, log_file)

    print("=" * 60)
    print("  H1 Solve Experiment — gpt-5.2 zero-shot solving")
    print(f"  run_id={args.run_id}  smoke={args.smoke}  workers={args.workers}")
    print(f"  dataset: {args.dataset}")
    print(f"  output:  {output_dir}")
    print(f"  started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    t0 = time.time()
    run_experiment(
        dataset_path=args.dataset,
        run_id=args.run_id,
        smoke=args.smoke,
        output_dir=output_dir,
        workers=args.workers,
    )
    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")
    print(f"  Finished:   {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if log_file:
        sys.stdout = sys.__stdout__
        log_file.close()
