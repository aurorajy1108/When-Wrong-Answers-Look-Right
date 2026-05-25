"""Core orchestration for the exchange-style debate pipeline."""

from __future__ import annotations

import json

from debate_pipeline_stance_only.pipeline import (
    call_llm,
    _clean_key_checks,
    _consensus_from_counts,
    _counts_from_stances,
    _extract_json,
    _force_stance,
    _profile_for_agent,
    _strip_problem,
    _verdict_from_stance,
)
from .prompts_debate_ours import AGENT_PROFILES, INITIAL_OUTPUT_SCHEMA, ROUND_OUTPUT_SCHEMA, ROUND_PROMPT

# Minimum number of debate rounds (beyond initial stances) before consensus-based early stopping.
_MIN_ROUNDS_BEFORE_STOP = 2

# Minimum support votes required to declare "correct".
# 3/5 = simple majority — same as original pipeline behaviour.
# Raising this to 4 was too aggressive (caused cascade flips to dominate).
_SUPERMAJORITY_SUPPORT = 3

# Assessment gating: among the supporting agents, require at least this many
# to have assessment_type == "answer_supported" (not "reasoning_insufficient").
# Rationale: "reasoning_insufficient_but_answer_not_refuted" means the agent
# cannot verify the answer — it should not count as positive evidence for correct.
_MIN_ANSWER_SUPPORTED = 3


def _coerce_confidence(value):
    try:
        return float(value)
    except Exception:
        return None


def _clean_message(text: str, limit: int = 240) -> str:
    return (text or "").replace("\n", " ").strip()[:limit]


def _clean_grade(value: str) -> str:
    value = str(value or "").strip().lower()
    return value if value in {"strong", "medium", "weak"} else "weak"


def _grade_weight(value: str) -> float:
    return {"strong": 1.0, "medium": 0.65, "weak": 0.3}.get(_clean_grade(value), 0.3)


def _priority(item: dict) -> float:
    conf = item.get("confidence")
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    return conf * _grade_weight(item.get("evidence_grade"))


def _parse_outbound_dm(value):
    if not isinstance(value, list):
        return []
    out = []
    for item in value[:2]:
        if not isinstance(item, dict):
            continue
        to = item.get("to")
        try:
            to = int(to)
        except Exception:
            continue
        if to not in {1, 2, 3, 4, 5}:
            continue
        msg = _clean_message(str(item.get("message", "")), limit=220)
        if msg:
            out.append({"to": to, "message": msg})
    return out


def _exchange_record(
    *,
    agent_id: int,
    stance: str,
    rationale: str,
    assessment_type: str,
    summary: str,
    key_checks,
    confidence,
    raw: str,
    source: str,
    evidence_grade="weak",
    swing_issue="",
    changed_mind=None,
    triggering_agent=None,
    change_reason="",
    addressed_points=None,
    outbound_broadcast="",
    outbound_dm=None,
):
    return {
        "agent_id": agent_id,
        "stance": stance,
        "verdict": _verdict_from_stance(stance),
        "assessment_type": assessment_type,
        "rationale": (rationale or "").strip(),
        "summary": _clean_message(summary or rationale, limit=400),
        "key_checks": _clean_key_checks(key_checks),
        "confidence": _coerce_confidence(confidence),
        "evidence_grade": _clean_grade(evidence_grade),
        "swing_issue": _clean_message(swing_issue, limit=180),
        "changed_mind": changed_mind,
        "triggering_agent": triggering_agent if isinstance(triggering_agent, int) else None,
        "change_reason": _clean_message(change_reason, limit=260),
        "addressed_points": _clean_key_checks(addressed_points),
        "outbound_broadcast": _clean_message(outbound_broadcast, limit=220),
        "outbound_dm": outbound_dm or [],
        "raw": raw,
        "source": source,
    }


def _initial_stances(problem: str, candidate_answer: str, candidate_reasoning: str, num_agents: int):
    outputs = []
    for idx in range(num_agents):
        profile = AGENT_PROFILES[idx]
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
            tag=f"exchange_init_agent_{idx + 1}_{profile['initial_mode']}",
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
                source=f"{profile['initial_mode']}_initial",
                outbound_broadcast=parsed.get("outbound_broadcast", ""),
                outbound_dm=_parse_outbound_dm(parsed.get("outbound_dm")),
            )
        )
    return outputs


def _build_disagreement_digest(previous_round: list[dict]) -> str:
    """Summarise the specific claims in contention between support and oppose agents.

    Returns an empty string when there is no disagreement.
    """
    support = sorted(
        [x for x in previous_round if x["stance"] == "support"],
        key=_priority, reverse=True,
    )
    oppose = sorted(
        [x for x in previous_round if x["stance"] == "oppose"],
        key=_priority, reverse=True,
    )
    if not support or not oppose:
        return ""

    lines = ["OPEN DISAGREEMENTS — every agent must address at least one of these:"]

    for s in support[:2]:
        claim = _clean_message(s.get("swing_issue") or s.get("summary", ""), 160)
        if claim:
            lines.append(f"  [Support] Agent {s['agent_id']} ({_clean_grade(s.get('evidence_grade'))}): {claim}")

    for o in oppose[:2]:
        claim = _clean_message(o.get("swing_issue") or o.get("summary", ""), 160)
        if claim:
            lines.append(f"  [Oppose]  Agent {o['agent_id']} ({_clean_grade(o.get('evidence_grade'))}): {claim}")

    lines.append(
        "  → Support agents: directly refute the strongest oppose claim above."
    )
    lines.append(
        "  → Oppose agents: directly refute the strongest support claim above."
    )
    lines.append(
        "  → Undecided agents: pick a side based on which concrete claim you can independently verify."
    )
    return "\n".join(lines)


def _build_scheduler_broadcast(previous_round: list[dict], round_idx: int) -> str:
    counts = _counts_from_stances(previous_round)
    support = [x for x in previous_round if x["stance"] == "support"]
    oppose = [x for x in previous_round if x["stance"] == "oppose"]
    support_sorted = sorted(support, key=_priority, reverse=True)
    oppose_sorted = sorted(oppose, key=_priority, reverse=True)
    assessment_counts = {}
    for item in previous_round:
        key = item.get("assessment_type") or "(missing)"
        assessment_counts[key] = assessment_counts.get(key, 0) + 1
    lines = [
        f"Round {round_idx} scheduler summary:",
        f"- support votes: {counts['correct']}  |  oppose votes: {counts['wrong']}",
        "- assessment mix: "
        + ", ".join(f"{k}={v}" for k, v in sorted(assessment_counts.items())),
    ]
    if support_sorted:
        s = support_sorted[0]
        lines.append(
            f"- strongest support: Agent {s['agent_id']} [{s.get('evidence_grade','weak')}] :: "
            f"{_clean_message(s.get('summary',''), 180)}"
        )
    if oppose_sorted:
        s = oppose_sorted[0]
        lines.append(
            f"- strongest oppose: Agent {s['agent_id']} [{s.get('evidence_grade','weak')}] :: "
            f"{_clean_message(s.get('summary',''), 180)}"
        )
    weak_supporters = [x["agent_id"] for x in support if _clean_grade(x.get("evidence_grade")) == "weak"]
    insufficient = [x["agent_id"] for x in previous_round if x.get("assessment_type") == "reasoning_insufficient_but_answer_not_refuted"]
    if weak_supporters:
        lines.append(f"- weak-support agents (must provide a concrete check or switch to oppose): {', '.join(map(str, weak_supporters))}")
    if insufficient:
        lines.append(f"- undecided agents (strict mode requires these to vote oppose unless resolved): {', '.join(map(str, insufficient))}")

    # Explicit disagreement digest — the most important addition for focusing debate.
    digest = _build_disagreement_digest(previous_round)
    if digest:
        lines.append("")
        lines.append(digest)

    broadcasts = []
    for item in previous_round:
        msg = _clean_message(item.get("outbound_broadcast", ""), 180)
        if msg:
            broadcasts.append(f"Agent {item['agent_id']}: {msg}")
    if broadcasts:
        lines.append("")
        lines.append("Previous agent broadcasts (concrete claims):")
        lines.extend(f"  {b}" for b in broadcasts[:6])
    return "\n".join(lines)


def _pick_auto_direct_messages(previous_round: list[dict]) -> dict[int, list[str]]:
    inbox = {i: [] for i in range(1, 6)}
    support = sorted(
        [x for x in previous_round if x["stance"] == "support"],
        key=_priority,
        reverse=True,
    )
    oppose = sorted(
        [x for x in previous_round if x["stance"] == "oppose"],
        key=_priority,
        reverse=True,
    )
    if support and oppose:
        s = support[0]
        o = oppose[0]
        # Cross-challenge the two strongest agents on opposite sides.
        inbox[s["agent_id"]].append(
            f"Scheduler challenge from Agent {o['agent_id']}: rebut this exact oppose claim -> "
            f"{_clean_message(o.get('swing_issue') or o.get('summary',''), 180)}"
        )
        inbox[o["agent_id"]].append(
            f"Scheduler challenge from Agent {s['agent_id']}: rebut this exact support claim -> "
            f"{_clean_message(s.get('swing_issue') or s.get('summary',''), 180)}"
        )
    if oppose:
        lead_oppose = oppose[0]
        for item in previous_round:
            if item["stance"] == "support" and _clean_grade(item.get("evidence_grade")) == "weak":
                # Weak supporters must either strengthen their evidence or switch to oppose.
                inbox[item["agent_id"]].append(
                    f"Scheduler: your support was weak-grade. In strict mode you must either produce a "
                    f"concrete independent check or switch to oppose. "
                    f"Agent {lead_oppose['agent_id']}'s objection: "
                    f"{_clean_message(lead_oppose.get('swing_issue') or lead_oppose.get('summary',''), 160)}"
                )
    if support and oppose:
        lead_support = support[0]
        lead_oppose = oppose[0]
        for item in previous_round:
            if item.get("assessment_type") == "reasoning_insufficient_but_answer_not_refuted":
                # Undecided agents are nudged to resolve by comparing the two strongest claims.
                inbox[item["agent_id"]].append(
                    f"Scheduler: you were undecided (insufficient reasoning). Strict mode requires you to vote "
                    f"'oppose' unless you can independently verify the answer. "
                    f"Compare Agent {lead_support['agent_id']}'s best support against "
                    f"Agent {lead_oppose['agent_id']}'s best refutation and state which one you actually checked."
                )
    # Relay peer-to-peer DMs from the previous round.
    for item in previous_round:
        for dm in item.get("outbound_dm", []):
            inbox.setdefault(dm["to"], []).append(
                f"From Agent {item['agent_id']}: {_clean_message(dm['message'], 180)}"
            )
    return inbox


def _debate_round(*, round_idx: int, previous_round: list[dict], problem: str, candidate_answer: str, candidate_reasoning: str):
    broadcast = _build_scheduler_broadcast(previous_round, round_idx)
    direct_inbox = _pick_auto_direct_messages(previous_round)
    outputs = []
    for prior in previous_round:
        agent_id = prior["agent_id"]
        profile = _profile_for_agent(agent_id)
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
            tag=f"exchange_round_{round_idx}_agent_{agent_id}",
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
                source="exchange_round",
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


def run_debate(
    problem: str,
    candidate_answer: str,
    candidate_reasoning: str,
    num_agents: int = 5,
    max_rounds: int = 5,
    progress_hook=None,
    skip_assessment_gate: bool = False,
) -> dict:
    if num_agents != 5:
        raise ValueError("This pipeline is designed for exactly 5 agents.")
    if progress_hook:
        progress_hook({"event": "variant_start"})
    initial_round = _initial_stances(problem, candidate_answer, candidate_reasoning, num_agents)
    if progress_hook:
        for item in initial_round:
            progress_hook({
                "event": "agent_output",
                "round": 1,
                "agent_id": item["agent_id"],
                "stance": item["stance"],
                "assessment_type": item.get("assessment_type"),
                "evidence_grade": item.get("evidence_grade"),
            })
    rounds = [initial_round]
    exchanges = []

    final_round = initial_round
    stop_round = 1
    final_counts = _counts_from_stances(final_round)
    final_verdict, final_consensus = _consensus_from_counts(final_counts, num_agents)

    # Always run at least _MIN_ROUNDS_BEFORE_STOP rounds total.
    # Consensus-based early stopping is only allowed once we have done enough rounds.
    # This ensures thorough deliberation even when agents agree quickly.
    if final_consensus != "unanimous" or stop_round < _MIN_ROUNDS_BEFORE_STOP:
        for round_idx in range(2, max_rounds + 1):
            if progress_hook:
                progress_hook({"event": "round_start", "round": round_idx})
            exchange = _debate_round(
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
                        "event": "agent_output",
                        "round": round_idx,
                        "agent_id": item["agent_id"],
                        "stance": item["stance"],
                        "assessment_type": item.get("assessment_type"),
                        "evidence_grade": item.get("evidence_grade"),
                        "changed_mind": item.get("changed_mind"),
                        "triggering_agent": item.get("triggering_agent"),
                    })
            rounds.append(final_round)
            final_counts = _counts_from_stances(final_round)
            final_verdict, final_consensus = _consensus_from_counts(final_counts, num_agents)
            stop_round = round_idx
            # Allow early stopping only after enough rounds of deliberation.
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

    # --- Strict supermajority override ---
    # Require at least _SUPERMAJORITY_SUPPORT agents to vote "support" before declaring "correct".
    if final_counts["correct"] < _SUPERMAJORITY_SUPPORT:
        final_verdict = "wrong"
        if final_counts["correct"] > 0:
            consensus_label = consensus_label + "_strict_overridden"

    # --- Assessment gating ---
    # Even if enough agents support, require _MIN_ANSWER_SUPPORTED agents to have
    # assessment_type == "answer_supported". Agents who support but say
    # "reasoning_insufficient_but_answer_not_refuted" are uncertain, not confirming.
    # Disabled when skip_assessment_gate=True (ablation experiment).
    if final_verdict == "correct" and not skip_assessment_gate:
        n_answer_supported = sum(
            1 for a in final_round
            if a.get("assessment_type") == "answer_supported"
        )
        if n_answer_supported < _MIN_ANSWER_SUPPORTED:
            final_verdict = "wrong"
            consensus_label = consensus_label + "_assessment_gated"

    out = {
        "final_verdict": final_verdict,
        "consensus_type": consensus_label,
        "stances": initial_round,
        "rounds": rounds,
        "exchanges": exchanges,
        "round_vote_counts": [_counts_from_stances(round_output) for round_output in rounds],
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
