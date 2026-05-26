# When-Wrong-Answers-Look-Right

## Naming Convention

- `run_*.py`: code used to run an experiment
- `*_results_summary.json`: main aggregate metrics for that experiment
- `*_predictions_detailed.json`: per-problem / per-variant detailed outputs
- `*_llm_cache.json`: cached model calls for that experiment
- `*_progress_log.jsonl`: step-by-step progress log during a run
- `__run1`, `__run2`, `__run3`: paper runs averaged together
- `__single_run`: one ablation run used directly in the paper
- Folder names do not include dates anymore

## Main Dataset

- Dataset used by the paper:
  - [final_dataset.json](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/dataset/final_dataset.json)
 

## Experiments

### 1. Single-Judge

- Code:
  - [run_single_judge_and_majvote_verify.py](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/run_single_judge_and_majvote_verify.py)
- Results:
  - [single_judge__run1](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/single_judge__run1)
  - [single_judge__run2](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/single_judge__run2)
  - [single_judge__run3](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/single_judge__run3)
- Inside each run folder:
  - `single_judge_results_summary.json`
  - `single_judge_predictions_detailed.json`
  - `single_judge_llm_cache.json`

### 2. MajVote-Solve

- Code:
  - [run_majvote_solve.py](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/run_majvote_solve.py)
- Results:
  - [majvote_solve__run1](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/majvote_solve__run1)
  - [majvote_solve__run2](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/majvote_solve__run2)
  - [majvote_solve__run3](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/majvote_solve__run3)
- Inside each run folder:
  - `majvote_solve_results_summary.json`
  - `majvote_solve_predictions_detailed.json`
  - `majvote_solve_llm_cache.json`

### 3. MajVote-Verify

- Code:
  - [run_single_judge_and_majvote_verify.py](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/run_single_judge_and_majvote_verify.py)
- Results:
  - [majvote_verify__run1](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/majvote_verify__run1)
  - [majvote_verify__run2](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/majvote_verify__run2)
  - [majvote_verify__run3](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/majvote_verify__run3)
- Inside each run folder:
  - `majvote_verify_results_summary.json`
  - `majvote_verify_predictions_detailed.json`
  - `majvote_verify_llm_cache.json`

### 4. MajVote-Precise

- Code:
  - [run_majvote_precise.py](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/run_majvote_precise.py)
- Results:
  - [majvote_precise__single_run](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/majvote_precise__single_run)
- Inside:
  - `majvote_precise_results_summary.json`
  - `majvote_precise_predictions_detailed.json`

### 5. Debate-NoGate

- Code:
  - [run_debate_no_gate.py](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/debate_pipeline_exchange/run_debate_no_gate.py)
  - [pipeline_debate_ours.py](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/debate_pipeline_exchange/pipeline_debate_ours.py)
  - [prompts_debate_ours.py](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/debate_pipeline_exchange/prompts_debate_ours.py)
- Results:
  - [debate_no_gate__single_run](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/debate_pipeline_exchange/debate_no_gate__single_run)
- Inside:
  - `debate_no_gate_results_summary.json`
  - `debate_no_gate_predictions_detailed.json`

### 6. Debate-Uniform

- Code:
  - [run_debate_uniform.py](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/debate_pipeline_exchange/run_debate_uniform.py)
  - [pipeline_debate_uniform.py](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/debate_pipeline_exchange/pipeline_debate_uniform.py)
  - [prompts_debate_ours.py](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/debate_pipeline_exchange/prompts_debate_ours.py)
- Results:
  - [debate_uniform__single_run](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/debate_pipeline_exchange/debate_uniform__single_run)
- Inside:
  - `debate_uniform_results_summary.json`
  - `debate_uniform_predictions_detailed.json`

### 7. Debate (Ours)

- Code:
  - [run_debate_ours.py](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/debate_pipeline_exchange/run_debate_ours.py)
  - [pipeline_debate_ours.py](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/debate_pipeline_exchange/pipeline_debate_ours.py)
  - [prompts_debate_ours.py](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/debate_pipeline_exchange/prompts_debate_ours.py)
- Results:
  - [debate_ours__run1](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/debate_pipeline_exchange/debate_ours__run1)
  - [debate_ours__run2](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/debate_pipeline_exchange/debate_ours__run2)
  - [debate_ours__run3](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/debate_pipeline_exchange/debate_ours__run3)
- Inside each run folder:
  - `debate_ours_results_summary.json`
  - `debate_ours_predictions_detailed.json`
  - `debate_ours_progress_log.jsonl`

### 8. Solve-From-Scratch For The Verify-vs-Solve Figure

- Code:
  - [run_solve_from_scratch_for_verify_vs_solve.py](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/run_solve_from_scratch_for_verify_vs_solve.py)
- Results:
  - [solve_from_scratch__run1](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/solve_from_scratch__run1)
  - [solve_from_scratch__run2](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/solve_from_scratch__run2)
  - [solve_from_scratch__run3](/Users/aurorasun/Desktop/ECE399/ECE399/When-Wrong-Answers-Look-Right/baseline/solve_from_scratch__run3)
- Inside each run folder:
  - `solve_from_scratch_results_summary.json`
  - `solve_from_scratch_predictions_detailed.json`
  - `solve_from_scratch_llm_cache.json`
  - `solve_from_scratch_run.log`

## Quick Reading Guide

- If you want the main paper result, open:
  - `debate_pipeline_exchange/debate_ours__run1/`
  - `debate_pipeline_exchange/debate_ours__run2/`
  - `debate_pipeline_exchange/debate_ours__run3/`
- If you want the main baseline, open:
  - `baseline/single_judge__run1/`
  - `baseline/single_judge__run2/`
  - `baseline/single_judge__run3/`
- If you want the ablations, open:
  - `baseline/majvote_precise__single_run/`
  - `debate_pipeline_exchange/debate_no_gate__single_run/`
  - `debate_pipeline_exchange/debate_uniform__single_run/`

 Code is released under the MIT License. Dataset files are released for research use only.
