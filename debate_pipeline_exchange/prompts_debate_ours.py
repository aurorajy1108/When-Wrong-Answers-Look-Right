"""Prompt templates for the exchange-style debate pipeline."""

import copy
from debate_pipeline_stance_only.prompts import AGENT_PROFILES as _BASE_PROFILES


# Deep-copy base profiles and apply strict-mode overrides for the exchange pipeline.
# Goal: prefer false negatives over false positives.
# Agents must have *positive* concrete evidence to vote "support"; plausibility is not enough.
AGENT_PROFILES = copy.deepcopy(_BASE_PROFILES)

_STRICT_INITIAL = (
    "\n\nEVIDENCE QUALITY RULES for this pipeline: "
    "Prefer avoiding false positives (wrongly supporting a wrong answer) over false negatives. "
    "Require concrete evidence — not merely plausibility — to vote 'answer_supported'. "
    "If you land in 'reasoning_insufficient_but_answer_not_refuted', weigh the balance of evidence "
    "carefully: lean toward 'oppose' if you see specific warning signs, but do NOT automatically "
    "oppose just because the proof is incomplete. Use your judgment about whether the answer itself "
    "is likely correct."
)

_STRICT_ROUND = (
    " EVIDENCE QUALITY RULE: resist changing your stance without a concrete reason. "
    "Do not switch sides based on vague concern or an argument you cannot evaluate. "
    "If you are currently supporting, require a specific identified error — not just uncertainty — "
    "to switch to oppose. Focus on the unresolved disagreement flagged in the broadcast."
)

# Deliberation protocol injected into each agent's initial prompt, replacing the bare {output_schema}
# placeholder.  This forces the agent to work through a structured checklist before committing.
_INITIAL_DELIBERATION = """\
DELIBERATION PROTOCOL — complete every step in your rationale before writing the JSON:

Step A  Restate the claim.
  Write the candidate answer as a precise mathematical statement (one sentence).

Step B  Enumerate failure modes.
  List 2–4 specific failure modes that are common for this problem type
  (e.g., off-by-one error, missing boundary case, theorem misapplied, optimality vs feasibility gap,
  arithmetic inconsistency, sign error, equality case handled incorrectly).
  For each failure mode state: does it appear here, and on what evidence?

Step C  Perform one independent check.
  Execute at least one concrete, independent verification:
  a small-case test, an arithmetic spot-check, a bound argument, or a theorem applicability test.
  If you cannot complete any check, your evidence_grade must be 'weak'.

Step D  Steelman the opposition.
  State the single strongest reason the answer could be WRONG (even if you ultimately support it).
  Directly assess: is that reason FATAL (definitively proves the answer wrong), UNCERTAIN (raises
  doubt but is not conclusive), or REFUTED (you can disprove it)?
  Only let a steelman argument change your verdict if it is FATAL — not merely uncertain.

Step E  Commit to a verdict.
  Based on Steps A–D choose your verdict.
  - If your independent check (Step C) confirms the answer, vote 'correct' even if the submitted
    reasoning is imperfect.
  - If Steps A–D reveal a specific concrete flaw, vote 'wrong'.
  - If you are genuinely uncertain, let the overall balance of evidence decide — do not default
    to either side purely out of caution.

{output_schema}"""

for _p in AGENT_PROFILES:
    _p["initial_instructions"] = _p["initial_instructions"] + _STRICT_INITIAL
    _p["round_system_prompt"] = _p["round_system_prompt"] + _STRICT_ROUND
    # Inject deliberation protocol into the initial prompt template.
    _p["initial_prompt_template"] = _p["initial_prompt_template"].replace(
        "{output_schema}", _INITIAL_DELIBERATION
    )


INITIAL_OUTPUT_SCHEMA = """\
You MUST end with valid JSON only — no markdown, no extra text.

Output JSON:
{
  "verdict": "correct" | "wrong",
  "assessment_type": "answer_supported" | "answer_refuted" | "reasoning_insufficient_but_answer_not_refuted",
  "evidence_grade": "strong" | "medium" | "weak",
  "rationale": "...",
  "summary": "...",
  "swing_issue": "...",
  "key_checks": ["...", "..."],
  "confidence": 0.0,
  "outbound_broadcast": "...",
  "outbound_dm": [
    {"to": 1, "message": "..."}
  ]
}
"""


ROUND_OUTPUT_SCHEMA = """\
You MUST end with valid JSON only — no markdown, no extra text.

Output JSON:
{
  "stance": "support" | "oppose",
  "assessment_type": "answer_supported" | "answer_refuted" | "reasoning_insufficient_but_answer_not_refuted",
  "evidence_grade": "strong" | "medium" | "weak",
  "rationale": "...",
  "summary": "...",
  "swing_issue": "...",
  "key_checks": ["...", "..."],
  "confidence": 0.0,
  "changed_mind": true,
  "triggering_agent": 2,
  "change_reason": "...",
  "addressed_points": ["..."],
  "outbound_broadcast": "...",
  "outbound_dm": [
    {"to": 1, "message": "..."}
  ]
}
"""


ROUND_PROMPT = """\
You are agent {agent_id} ({agent_name}) in a multi-agent debate about whether a candidate answer is correct.

Problem:
{problem}

Candidate Answer:
{candidate_answer}

Candidate Reasoning (may be wrong):
{candidate_reasoning}

Your previous stance:
{previous_stance}

Your previous assessment type:
{previous_assessment_type}

Your previous summary:
{previous_summary}

Your previous key checks:
{previous_key_checks}

Scheduler broadcast for this round:
{broadcast_message}

Direct messages sent to you:
{direct_messages}

====  EVIDENCE QUALITY RULES  ====
This pipeline values precision: avoid endorsing a wrong answer.
- Plausibility alone does NOT justify 'support' — require a concrete check or argument.
- Do NOT switch your stance based solely on another agent's assertion you cannot verify yourself.
- If you are supporting, you need a specific reason to switch to oppose — vague uncertainty is
  not enough. Require a concrete identified flaw, not just an unresolved question.
- If you are opposing, a concrete verified argument from another agent CAN move you to support.
- 'reasoning_insufficient_but_answer_not_refuted': weigh the evidence; lean oppose if you see
  specific warning signs, but do not mandate oppose just because the proof is incomplete.
==================================

DELIBERATION PROTOCOL — work through all four steps in your rationale before writing the JSON:

Step 1 — Address the open disagreements.
The broadcast above identifies unresolved support vs. oppose disagreements.
You MUST directly respond to the strongest argument from the opposing side:
- If you support: explain exactly why the strongest oppose claim is wrong, incomplete, or irrelevant.
- If you oppose: explain exactly why the strongest support claim fails to establish correctness.
Do not skip this step. Restating your prior stance without engaging the opposing argument is not acceptable.

Step 2 — Check one new thing.
Perform or recall at least one concrete, independent verification you have not mentioned before:
an exact arithmetic step, a theorem applicability test, a small-case sanity check, a bound argument,
or an invariant check.
State what you checked and what the result was.
If you cannot produce such a check, your evidence_grade is 'weak' — but weak evidence for support
is still support if no concrete flaw has been identified.

Step 3 — Steelman the side you are NOT on.
Write the single best argument for the opposing stance — the hardest case against your current position.
Then classify it: FATAL (definitively proves answer wrong), UNCERTAIN (raises doubt but inconclusive),
or REFUTED (you can disprove it).
Only let this steelman argument change your stance if it is FATAL — an uncertain objection is not
sufficient reason to flip.

Step 4 — Commit.
State your final stance and assessment_type only after completing Steps 1–3.
- If you are changing stance: record the trigger in `change_reason` and the agent in `triggering_agent`.
- If you are keeping your stance: explain specifically why Steps 1–3 were not enough to move you.
- Do not silently switch sides.

Message behavior:
- `outbound_broadcast`: at most one concise, concrete claim you want every agent to address next round.
  Prioritise the single most important unresolved mathematical issue.
- `outbound_dm`: 0–2 targeted messages. Each should challenge or answer one specific mathematical claim
  made by a named agent. Be concrete and short.

Use assessment types carefully:
- answer_supported: you have concrete positive evidence the answer is correct
- answer_refuted: you have concrete evidence the answer is wrong
- reasoning_insufficient_but_answer_not_refuted: reasoning is weak, you did not disprove the answer —
  weigh whether specific warning signs exist; lean oppose if they do, lean support if they don't

Use `evidence_grade` carefully:
- strong: concrete exact check, validated counterexample, or trusted and applicable theorem argument
- medium: partially checked argument with exactly one clear unresolved gap
- weak: stance rests on plausibility, unverified intuition, or an unchecked claim

The `summary` should be 1–3 sentences naming the main reason for your current stance.
The `swing_issue` should be one short sentence naming the decisive open question.

{output_schema}
"""
