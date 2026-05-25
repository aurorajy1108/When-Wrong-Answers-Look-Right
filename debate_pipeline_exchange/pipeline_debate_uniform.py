"""
Experiment 5 — Homogeneous debate pipeline.

All 5 agents use IDENTICAL system prompts (no persona differentiation).
Same deliberation protocol (Steps A-E), same round protocol (Steps 1-4),
same verdict rules as the default pipeline.

Purpose: test whether heterogeneous agent roles are responsible for
structural gains, as predicted by DynaDebate and the Choi et al.
martingale analysis.

Expected: homogeneous debate ≈ B4 (call-matched majority vote).
If confirmed, this validates that the 59% structural gain in the
default pipeline is attributable to role heterogeneity.
"""

from __future__ import annotations

from debate_pipeline_stance_only.pipeline import (
    call_llm,
    _clean_key_checks,
    _consensus_from_counts,
    _counts_from_stances,
    _extract_json,
    _force_stance,
    _strip_problem,
    _verdict_from_stance,
)
from debate_pipeline_exchange.pipeline_debate_ours import (
    _MIN_ROUNDS_BEFORE_STOP,
    _SUPERMAJORITY_SUPPORT,
    _MIN_ANSWER_SUPPORTED,
    _coerce_confidence,
    _clean_message,
    _clean_grade,
    _grade_weight,
    _priority,
    _parse_outbound_dm,
    _exchange_record,
    _build_disagreement_digest,
    _build_scheduler_broadcast,
    _pick_auto_direct_messages,
)
from debate_pipeline_exchange.prompts_debate_ours import (
    INITIAL_OUTPUT_SCHEMA,
    ROUND_OUTPUT_SCHEMA,
    ROUND_PROMPT,
)
from debate_pipeline_exchange.prompts_debate_ours import _INITIAL_DELIBERATION


# ---------------------------------------------------------------------------
# Generic (homogeneous) agent definition — same for all 5 agents
# ---------------------------------------------------------------------------

_GENERIC_INSTRUCTIONS = (
    "You are a rigorous mathematical judge. "
    "Your task is to decide whether the candidate ANSWER to a competition mathematics problem "
    "is correct, given the problem statement, the candidate answer, and accompanying reasoning. "
    "Judge the answer directly — not the quality or completeness of the submitted proof. "
    "A flawed proof does not automatically make the answer wrong, but you need concrete evidence "
    "(a specific numerical check, a bound argument, or a theorem applicability test) to confirm "
    "or refute the answer. "
    "Avoid confirming an answer merely because it looks plausible. "
    "Prefer false negatives over false positives: require positive evidence to support, "
    "not merely the absence of a found flaw."
)

_GENERIC_INITIAL_TEMPLATE = """\
You are Agent {agent_id}: Mathematical Judge.

Problem:
{problem}

Candidate Answer:
{candidate_answer}

Candidate Reasoning (may be wrong, incomplete, or overcomplicated):
{candidate_reasoning}

Task:
Decide whether the CANDIDATE ANSWER is correct.
Judge the answer, not the quality of the reasoning.

""" + _INITIAL_DELIBERATION

_GENERIC_ROUND_SYSTEM_PROMPT = (
    "You are a rigorous mathematical judge participating in a multi-agent debate. "
    "Judge whether the candidate ANSWER is correct based on evidence accumulated in the debate. "
    "Change your stance only when a concrete, specific mathematical argument compels you to. "
    "Do not flip based on social pressure or vague concern. "
    "Focus on the unresolved disagreement flagged in the broadcast."
)

# 5 identical profiles — only agent_id differs (needed for DM routing)
HOMOGENEOUS_PROFILES = [
    {
        "agent_id": i + 1,
        "name": "Mathematical Judge",
        "initial_mode": "judge",
        "initial_instructions": _GENERIC_INSTRUCTIONS,
        "initial_prompt_template": _GENERIC_INITIAL_TEMPLATE,
        "round_system_prompt": _GENERIC_ROUND_SYSTEM_PROMPT,
    }
    for i in range(5)
]


def _profile_for_agent_hom(agent_id: int) -> dict:
    for p in HOMOGENEOUS_PROFILES:
        if p["agent_id"] == agent_id:
            return p
    raise ValueError(f"Unknown agent_id: {agent_id}")


# ---------------------------------------------------------------------------
# Initial deliberation (homogeneous version)
# ---------------------------------------------------------------------------

def _initial_stances_hom(problem: str, candidate_answer: str,
                          candidate_reasoning: str, num_agents: int):
    outputs = []
    for idx in range(num_agents):
        profile = HOMOGENEOUS_PROFILES[idx]
        prompt = profile["initial_prompt_template"].format(
            agent_id=profile["agent_id"],
            agent_name=profile["name"],
            problem=_strip_problem(problem),
            candidate_answer=candidate_answer,
            candidate_reasoning=candidate_reasoning or "(none)",
            output_schema=INITIAL_OUTPUT_SCHEMA,
        )
        raw = call_llm(
            prompt,
            tag=f"hom_init_agent_{idx + 1}_judge",
            instructions=profile["initial_instructions"],
            effort="low",
        )
        parsed = _extract_json(raw) or {}
        stance = _force_stance(raw)
        rationale = parsed.get("rationale", raw.strip())
        outputs.append(
            _exchange_record(
                agent_id=idx + 1,
                stance=stance,
                rationale=rationale,
                assessment_type=str(parsed.get("assessment_type", "")).strip(),
                summary=parsed.get("summary", rationale[:220]),
                key_checks=parsed.get("key_checks"),
                confidence=parsed.get("confidence"),
                evidence_grade=parsed.get("evidence_grade", "weak"),
                swing_issue=parsed.get("swing_issue", ""),
                raw=raw,
                source="hom_initial",
                outbound_broadcast=parsed.get("outbound_broadcast", ""),
                outbound_dm=_parse_outbound_dm(parsed.get("outbound_dm")),
            )
        )
    return outputs


# ---------------------------------------------------------------------------
# Exchange round (homogeneous version)
# ---------------------------------------------------------------------------

def _debate_round_hom(*, round_idx: int, previous_round: list[dict],
                       problem: str, candidate_answer: str, candidate_reasoning: str):
    broadcast = _build_scheduler_broadcast(previous_round, round_idx)
    direct_inbox = _pick_auto_direct_messages(previous_round)
    outputs = []
    for prior in previous_round:
        agent_id = prior["agent_id"]
        profile = _profile_for_agent_hom(agent_id)
        direct_messages = direct_inbox.get(agent_id) or ["(none)"]
        prompt = ROUND_PROMPT.format(
            agent_id=agent_id,
            agent_name=profile["name"],
            problem=_strip_problem(problem),
            candidate_answer=candidate_answer,
            candidate_reasoning=candidate_reasoning or "(none)",
            previous_stance=prior["stance"],
            previous_assessment_type=prior.get("assessment_type", "") or "(none)",
            previous_summary=prior.get("summary", "") or "(none)",
            previous_key_checks=", ".join(prior.get("key_checks") or []) or "(none)",
            broadcast_message=broadcast,
            direct_messages="\n".join(f"- {m}" for m in direct_messages),
            output_schema=ROUND_OUTPUT_SCHEMA,
        )
        raw = call_llm(
            prompt,
            tag=f"hom_round_{round_idx}_agent_{agent_id}",
            instructions=profile["round_system_prompt"],
            effort="low",
        )
        parsed = _extract_json(raw) or {}
        stance = parsed.get("stance", "")
        if stance not in {"support", "oppose"}:
            stance = _force_stance(raw)
        rationale = parsed.get("rationale", raw.strip())
        outputs.append(
            _exchange_record(
                agent_id=agent_id,
                stance=stance,
                rationale=rationale,
                assessment_type=str(parsed.get("assessment_type", "")).strip(),
                summary=parsed.get("summary", rationale[:220]),
                key_checks=parsed.get("key_checks"),
                confidence=parsed.get("confidence"),
                evidence_grade=parsed.get("evidence_grade", "weak"),
                swing_issue=parsed.get("swing_issue", ""),
                raw=raw,
                source="hom_round",
                changed_mind=parsed.get("changed_mind"),
                triggering_agent=parsed.get("triggering_agent"),
                change_reason=parsed.get("change_reason", ""),
                addressed_points=parsed.get("addressed_points"),
                outbound_broadcast=parsed.get("outbound_broadcast", ""),
                outbound_dm=_parse_outbound_dm(parsed.get("outbound_dm")),
            )
        )
    return {
        "broadcast": broadcast,
        "direct_inbox": direct_inbox,
        "outputs": outputs,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_debate_homogeneous(
    problem: str,
    candidate_answer: str,
    candidate_reasoning: str,
    num_agents: int = 5,
    max_rounds: int = 5,
    progress_hook=None,
) -> dict:
    """
    Run the debate with 5 homogeneous (identical system prompt) agents.
    Verdict rules identical to run_debate() in pipeline_debate_ours.py.
    """
    if num_agents != 5:
        raise ValueError("Homogeneous pipeline requires exactly 5 agents.")
    if progress_hook:
        progress_hook({"event": "variant_start"})

    initial_round = _initial_stances_hom(
        problem, candidate_answer, candidate_reasoning, num_agents
    )
    if progress_hook:
        for item in initial_round:
            progress_hook({
                "event": "agent_output", "round": 1,
                "agent_id": item["agent_id"], "stance": item["stance"],
                "assessment_type": item.get("assessment_type"),
                "evidence_grade": item.get("evidence_grade"),
            })

    rounds = [initial_round]
    exchanges = []
    final_round = initial_round
    stop_round = 1
    final_counts = _counts_from_stances(final_round)
    final_verdict, final_consensus = _consensus_from_counts(final_counts, num_agents)

    if final_consensus != "unanimous" or stop_round < _MIN_ROUNDS_BEFORE_STOP:
        for round_idx in range(2, max_rounds + 1):
            if progress_hook:
                progress_hook({"event": "round_start", "round": round_idx})
            exchange = _debate_round_hom(
                round_idx=round_idx,
                previous_round=final_round,
                problem=problem,
                candidate_answer=candidate_answer,
                candidate_reasoning=candidate_reasoning,
            )
            exchanges.append({
                "round": round_idx,
                "broadcast": exchange["broadcast"],
                "direct_inbox": exchange["direct_inbox"],
            })
            final_round = exchange["outputs"]
            if progress_hook:
                for item in final_round:
                    progress_hook({
                        "event": "agent_output", "round": round_idx,
                        "agent_id": item["agent_id"], "stance": item["stance"],
                        "assessment_type": item.get("assessment_type"),
                        "evidence_grade": item.get("evidence_grade"),
                        "changed_mind": item.get("changed_mind"),
                    })
            rounds.append(final_round)
            final_counts = _counts_from_stances(final_round)
            final_verdict, final_consensus = _consensus_from_counts(final_counts, num_agents)
            stop_round = round_idx
            if final_consensus == "unanimous" and round_idx >= _MIN_ROUNDS_BEFORE_STOP:
                break

    first_counts = _counts_from_stances(initial_round)
    first_verdict, first_consensus = _consensus_from_counts(first_counts, num_agents)
    final_counts = _counts_from_stances(final_round)
    final_verdict, final_consensus = _consensus_from_counts(final_counts, num_agents)

    if stop_round == max_rounds and final_consensus != "unanimous":
        final_verdict = "correct" if final_counts["correct"] > final_counts["wrong"] else "wrong"
        consensus_label = "majority_after_max_rounds"
    elif final_consensus == "fallback":
        final_verdict = "correct" if final_counts["correct"] > final_counts["wrong"] else "wrong"
        consensus_label = "majority_fallback"
    else:
        consensus_label = f"{final_consensus}_round{stop_round}"

    # Same supermajority rule
    if final_counts["correct"] < _SUPERMAJORITY_SUPPORT:
        final_verdict = "wrong"
        if final_counts["correct"] > 0:
            consensus_label += "_strict_overridden"

    # Same assessment gating
    if final_verdict == "correct":
        n_answer_supported = sum(
            1 for a in final_round
            if a.get("assessment_type") == "answer_supported"
        )
        if n_answer_supported < _MIN_ANSWER_SUPPORTED:
            final_verdict = "wrong"
            consensus_label += "_assessment_gated"

    out = {
        "final_verdict": final_verdict,
        "consensus_type": consensus_label,
        "stances": initial_round,
        "rounds": rounds,
        "exchanges": exchanges,
        "round_vote_counts": [_counts_from_stances(r) for r in rounds],
        "vote_counts": final_counts,
        "first_round_verdict": first_verdict,
        "final_round_verdict": final_verdict,
        "first_round_counts": first_counts,
        "final_round_counts": final_counts,
        "first_round_consensus": first_consensus,
        "final_round_consensus": final_consensus,
        "stop_round": stop_round,
    }
    if progress_hook:
        progress_hook({
            "event": "variant_done",
            "stop_round": stop_round,
            "final_verdict": final_verdict,
            "consensus_type": consensus_label,
        })
    return out
