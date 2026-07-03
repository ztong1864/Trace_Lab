"""Narrative and audit report builders for Scientist Workflow Agent runs."""

from __future__ import annotations

import json
from typing import Any


WORKFLOW_REPORT_VERSION = "v1"


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_or_none(value: Any, digits: int = 4) -> float | None:
    number = _as_float(value, default=None)
    if number is None:
        return None
    return round(number, digits)


def _best_value_from_observation(best_observation: dict[str, Any] | None) -> float | None:
    if not isinstance(best_observation, dict):
        return None
    numeric_values: list[float] = []
    for value in best_observation.values():
        numeric = _as_float(value, default=None)
        if numeric is not None:
            numeric_values.append(numeric)
    if not numeric_values:
        return None
    return numeric_values[-1]


def _action_package(record: dict[str, Any]) -> dict[str, Any]:
    action_package = record.get("action_package")
    if isinstance(action_package, dict):
        return dict(action_package)
    intervention_plan = record.get("intervention_plan_full")
    if isinstance(intervention_plan, dict) and isinstance(intervention_plan.get("action_package"), dict):
        return dict(intervention_plan.get("action_package", {}))
    mode = str(record.get("controller_mode", "bo_direct"))
    if mode == "bo_focus_then_rerank":
        return {
            "intent": "probe",
            "shortlist_policy": "coverage_shape",
            "repeat_policy": "avoid_anchor_repeat",
            "selection_policy": "select_from_shaped_shortlist",
            "verification_policy": "normal",
            "focus_policy": "temporary_focus",
        }
    if mode == "bo_rerank_topk":
        return {
            "intent": "balance",
            "shortlist_policy": "plain",
            "repeat_policy": "allow",
            "selection_policy": "select_from_shaped_shortlist",
            "verification_policy": "normal",
            "focus_policy": "full_space",
        }
    return {
        "intent": "exploit",
        "shortlist_policy": "plain",
        "repeat_policy": "allow",
        "selection_policy": "bo_top1",
        "verification_policy": "normal",
        "focus_policy": "full_space",
    }


def _execution_strategy(record: dict[str, Any]) -> str:
    executed = str(record.get("executed_execution_action", "") or "").strip()
    if executed:
        return executed
    requested = str(record.get("requested_execution_action", "") or "").strip()
    if requested:
        return requested
    return str(record.get("controller_mode", "bo_direct"))


def _shortlist_shaping_summary(record: dict[str, Any]) -> str:
    trace = record.get("shortlist_shaping_trace") or {}
    if not isinstance(trace, dict):
        return "Shortlist shaping trace unavailable."
    summary = str(trace.get("summary", "")).strip()
    if summary:
        return summary
    if trace.get("enabled"):
        return "Shortlist shaping ran without a custom summary."
    return "Shortlist shaping not used."


def _verification_feedback_summary(record: dict[str, Any]) -> str:
    verification = record.get("verification_pass") or {}
    if not isinstance(verification, dict) or not verification:
        return "Verification not run."
    status = str(verification.get("status", "pass"))
    risk_flags = list(verification.get("risk_flags", []) or [])
    if risk_flags:
        return f"Verification status={status}; risk_flags={risk_flags}."
    return f"Verification status={status}."


def _problem_state(record: dict[str, Any], decision_context: dict[str, Any]) -> str:
    diagnosis = record.get("stagnation_diagnosis") or {}
    trigger_reasons = set(record.get("controller_trigger_reasons") or record.get("trigger_reasons") or [])
    duplicate_ratio = _as_float(decision_context.get("recent_duplicate_ratio"), 0.0) or 0.0
    coverage_ratio = _as_float(decision_context.get("coverage_overall_ratio"), 1.0) or 1.0
    scaffold_concentration = (
        _as_float(decision_context.get("recent_scaffold_concentration"), 0.0) or 0.0
    )
    no_improvement_rounds = int(decision_context.get("no_improvement_rounds", 0) or 0)

    if "duplicate_high" in trigger_reasons or duplicate_ratio >= 0.35:
        return "repetitive_sampling"
    if (
        "scaffold_concentration_high" in trigger_reasons
        or "post_breakthrough_stall" in trigger_reasons
        or scaffold_concentration >= 0.8
    ):
        return "local_overfocus"
    if "coverage_low" in trigger_reasons or coverage_ratio <= 0.4:
        return "low_coverage"
    if diagnosis.get("is_stagnating") or no_improvement_rounds >= 3:
        return "stagnation"
    if bool(record.get("improved_best")):
        return "healthy_progress"
    return "uncertain"


def _diagnosis_evidence(record: dict[str, Any], decision_context: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    signal_map = {
        "recent_duplicate_ratio": decision_context.get("recent_duplicate_ratio"),
        "coverage_overall_ratio": decision_context.get("coverage_overall_ratio"),
        "no_improvement_rounds": decision_context.get("no_improvement_rounds"),
        "recent_scaffold_concentration": decision_context.get("recent_scaffold_concentration"),
        "current_subpool_ratio": decision_context.get("current_subpool_ratio"),
        "trigger_reasons": record.get("controller_trigger_reasons") or record.get("trigger_reasons"),
    }
    for signal, value in signal_map.items():
        if value in (None, "", [], {}):
            continue
        evidence.append({"signal": signal, "value": value})
    return evidence


def _diagnosis_confidence(problem_state: str, evidence: list[dict[str, Any]]) -> str:
    if problem_state == "uncertain":
        return "low"
    if len(evidence) >= 3:
        return "high"
    return "medium"


def _diagnosis_summary(problem_state: str, decision_context: dict[str, Any]) -> str:
    no_improvement_rounds = int(decision_context.get("no_improvement_rounds", 0) or 0)
    coverage_ratio = _round_or_none(decision_context.get("coverage_overall_ratio"), digits=3)
    scaffold_concentration = _round_or_none(
        decision_context.get("recent_scaffold_concentration"),
        digits=3,
    )
    if problem_state == "healthy_progress":
        return "The run is still making useful progress without strong intervention signals."
    if problem_state == "stagnation":
        return (
            f"Progress appears stalled: no-improvement rounds={no_improvement_rounds}, "
            f"coverage={coverage_ratio}."
        )
    if problem_state == "local_overfocus":
        return (
            "Search appears locally concentrated, "
            f"with recent scaffold concentration={scaffold_concentration}."
        )
    if problem_state == "low_coverage":
        return f"Coverage remains limited, with overall coverage ratio={coverage_ratio}."
    if problem_state == "repetitive_sampling":
        duplicate_ratio = _round_or_none(decision_context.get("recent_duplicate_ratio"), digits=3)
        return f"Recent sampling is repetitive, with duplicate ratio={duplicate_ratio}."
    return "The run state is mixed and no single issue clearly dominates."


def _plan_goal(record: dict[str, Any], problem_state: str) -> str:
    action_package = _action_package(record)
    intent = str(action_package.get("intent", "balance"))
    selection_policy = str(action_package.get("selection_policy", "bo_top1"))
    strategy = _execution_strategy(record)
    selected_source = str(record.get("selected_candidate_source", "bo_top_ranked"))
    if intent == "probe":
        return "buy_information"
    if intent == "exploit":
        return "harvest_objective"
    if selection_policy == "bo_top1_from_shaped_shortlist":
        return "stabilize_shortlist"
    if strategy in {"bo_focus_then_rerank", "focused_shortlist_alt_pick"}:
        return "local_refinement"
    if strategy in {"bo_rerank_topk", "shortlist_alt_pick"}:
        if selected_source == "diversity_injected" or bool(record.get("diversity_pool_used")):
            return "increase_diversity"
        if bool(record.get("selected_differs_from_bo_top1")):
            return "test_transfer"
        return "stabilize_progress"
    if problem_state in {"healthy_progress", "uncertain"}:
        return "stabilize_progress"
    return "stabilize_progress"


def _plan_scope(record: dict[str, Any]) -> str:
    if bool(record.get("focus_filter_applied")) or str(record.get("constraint_mode")) == "focused_filter":
        return "focused_pool"
    return "full_pool"


def _expected_benefit(strategy: str, plan_goal: str) -> str:
    if plan_goal == "buy_information":
        return "Use the next experiment to test a more informative or contrastive hypothesis."
    if plan_goal == "harvest_objective":
        return "Use the next experiment to protect near-term objective gain."
    if plan_goal == "stabilize_shortlist":
        return "Clean shortlist structure while keeping BO in charge of the final surviving candidate."
    if strategy in {"bo_focus_then_rerank", "focused_shortlist_alt_pick"}:
        return "Reduce local search noise and refine a bounded legal candidate pool."
    if strategy in {"bo_rerank_topk", "shortlist_alt_pick"} and plan_goal == "increase_diversity":
        return "Preserve BO signal while injecting more diverse transfer candidates."
    if strategy in {"bo_rerank_topk", "shortlist_alt_pick"} and plan_goal == "test_transfer":
        return "Probe alternatives beyond BO top-1 without leaving the BO shortlist."
    return "Keep the optimization loop stable and let BO drive candidate proposal directly."


def _plan_summary(record: dict[str, Any], plan_goal: str, plan_scope: str) -> str:
    action_package = _action_package(record)
    strategy = _execution_strategy(record)
    intent = str(action_package.get("intent", "balance"))
    shortlist_policy = str(action_package.get("shortlist_policy", "plain"))
    selection_policy = str(action_package.get("selection_policy", "bo_top1"))
    if selection_policy == "bo_top1_from_shaped_shortlist":
        return (
            f"Spend this round in {intent} mode using {shortlist_policy} shortlist shaping "
            f"in {plan_scope} mode, then keep the BO-preferred surviving candidate."
        )
    if strategy not in {"bo_direct", "direct_bo_pick"}:
        return (
            f"Spend this round in {intent} mode using {shortlist_policy} shortlist shaping "
            f"in {plan_scope} mode."
        )
    return "Trust the BO proposal directly because no stronger intervention is justified."


def _counterfactual_alternatives(strategy: str) -> dict[str, str]:
    alternatives = {
        "bo_direct": "Skipped because direct execution would ignore shortlist-level scientific judgment.",
        "bo_rerank_topk": "Skipped because shortlist reranking was not necessary or not strong enough.",
        "bo_focus_then_rerank": "Skipped because focused filtering would likely over-constrain the legal pool.",
        "direct_bo_pick": "Skipped because direct execution would ignore shortlist-level scientific judgment.",
        "shape_only_bo_pick": "Skipped because shortlist structure intervention was not necessary or not strong enough.",
        "shortlist_alt_pick": "Skipped because shortlist-internal alternative selection was not necessary or not strong enough.",
        "focused_shortlist_alt_pick": "Skipped because focused shortlist selection would likely over-constrain the legal pool.",
    }
    if strategy in alternatives:
        alternatives[strategy] = "Chosen strategy."
    return alternatives


def _exploration_exploitation_label(record: dict[str, Any]) -> str:
    if str(record.get("selected_candidate_source")) == "diversity_injected":
        return "exploration"
    if bool(record.get("selected_differs_from_bo_top1")):
        return "balanced"
    return "exploitation"


def _override_reason_summary(record: dict[str, Any], plan_goal: str) -> str:
    selection_policy = str(_action_package(record).get("selection_policy", "bo_top1"))
    if selection_policy == "bo_top1_from_shaped_shortlist" and not bool(
        record.get("selected_differs_from_bo_top1")
    ):
        return "Kept the BO-preferred surviving candidate after shortlist shaping."
    if bool(record.get("llm_requested_override")) and not bool(record.get("selected_differs_from_bo_top1")):
        return (
            "The controller selected a shortlist alternative, but the guardrail kept the original BO top proposal: "
            f"{record.get('override_guardrail_reason', 'guardrail_blocked')}."
        )
    if not bool(record.get("selected_differs_from_bo_top1")):
        return "Accepted BO top-1 candidate without override."
    selected_source = str(record.get("selected_candidate_source", "bo_top_ranked"))
    if selected_source == "diversity_injected":
        return "Selected a diversity-injected shortlist candidate with better transfer value than the original BO top proposal."
    if plan_goal == "test_transfer":
        return "Selected a shortlist alternative to test stronger transfer potential."
    return "Selected a different shortlist candidate after shortlist-level review favored it over the original BO top proposal."


def _execution_summary(record: dict[str, Any], plan_goal: str) -> str:
    rank = record.get("selected_candidate_rank_in_shortlist")
    strategy = _execution_strategy(record)
    if strategy in {"bo_direct", "direct_bo_pick"}:
        return "Executed the BO top-1 proposal directly."
    if strategy == "shape_only_bo_pick":
        return "Shaped the shortlist structure, then kept the BO-preferred surviving candidate."
    if bool(record.get("selected_differs_from_bo_top1")):
        return (
            f"Executed shortlist candidate rank={rank} instead of BO top-1 "
            f"to {plan_goal.replace('_', ' ')}."
        )
    return "Reviewed a BO shortlist but retained the BO top-ranked candidate."


def _reflect_summary(record: dict[str, Any], plan_summary: str) -> dict[str, Any]:
    reflection = record.get("reflection") or {}
    result = _as_float(record.get("result"), default=None)
    improved_best = bool(record.get("improved_best"))
    insight = str(reflection.get("insight", "")).strip()
    if not insight:
        insight = (
            "The selected action improved the best-so-far objective."
            if improved_best
            else "The selected action produced a valid outcome but did not improve the best-so-far objective."
        )
    hypothesis = str(reflection.get("next_step_hypothesis", "")).strip()
    if not hypothesis:
        hypothesis = plan_summary
    suggested_focus = reflection.get("suggested_focus", []) or []
    next_round = (
        f"Next round should monitor suggested focus {suggested_focus}."
        if suggested_focus
        else "Next round should reassess whether the current workflow still matches the observed evidence."
    )
    return {
        "what_we_learned": insight,
        "did_plan_work": improved_best,
        "updated_hypothesis": hypothesis,
        "next_round_implication": next_round,
        "learning_summary": (
            f"Observed result={result}; "
            f"{'best improved' if improved_best else 'best did not improve'}."
        ),
    }


def build_workflow_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    decision_context = record.get("decision_context_snapshot") or {}
    best_observation = decision_context.get("best_observation")
    best_value = _best_value_from_observation(best_observation)
    observe = {
        "current_best_observation": best_observation,
        "current_best_value": best_value,
        "best_improvement_last_3": _round_or_none(decision_context.get("best_improvement_last_3")),
        "no_improvement_rounds": int(decision_context.get("no_improvement_rounds", 0) or 0),
        "coverage_overall_ratio": _round_or_none(decision_context.get("coverage_overall_ratio")),
        "recent_duplicate_ratio": _round_or_none(decision_context.get("recent_duplicate_ratio")),
        "recent_scaffold_concentration": _round_or_none(
            decision_context.get("recent_scaffold_concentration")
        ),
        "visited_candidate_ratio": _round_or_none(decision_context.get("visited_candidate_ratio")),
        "candidate_pool_size": record.get("candidate_pool_size"),
        "candidate_pool_total": record.get("candidate_pool_total"),
        "working_hypothesis_summary": (
            (record.get("working_memory") or {})
            .get("research_state", {})
            .get("summary", "")
        ),
    }

    problem_state = _problem_state(record, decision_context)
    evidence = _diagnosis_evidence(record, decision_context)
    diagnose = {
        "problem_state": problem_state,
        "evidence": evidence,
        "confidence": _diagnosis_confidence(problem_state, evidence),
        "diagnosis_summary": _diagnosis_summary(problem_state, decision_context),
        "trigger_reasons": record.get("controller_trigger_reasons") or record.get("trigger_reasons", []),
    }

    action_package = _action_package(record)
    strategy = _execution_strategy(record)
    plan_goal = _plan_goal(record, problem_state)
    plan_scope = _plan_scope(record)
    plan = {
        "selected_strategy": strategy,
        "action_package": action_package,
        "budget_intent": action_package.get("intent"),
        "plan_goal": plan_goal,
        "plan_scope": plan_scope,
        "shortlist_policy": action_package.get("shortlist_policy"),
        "repeat_policy": action_package.get("repeat_policy"),
        "selection_policy": action_package.get("selection_policy"),
        "verification_policy": action_package.get("verification_policy"),
        "requested_execution_action": record.get("requested_execution_action"),
        "executed_execution_action": record.get("executed_execution_action"),
        "contract_satisfied": record.get("contract_satisfied"),
        "execution_fallback_reason": record.get("execution_fallback_reason"),
        "focus_variables": record.get("focus_variables", []),
        "expected_benefit": _expected_benefit(strategy, plan_goal),
        "shortlist_intervention": str(action_package.get("selection_policy", "bo_top1")) != "bo_top1",
        "override_intent": strategy in {"shortlist_alt_pick", "focused_shortlist_alt_pick", "bo_rerank_topk", "bo_focus_then_rerank"}
        and str(action_package.get("selection_policy", "bo_top1")) == "select_from_shaped_shortlist",
        "plan_summary": _plan_summary(record, plan_goal, plan_scope),
        "shortlist_shaping_summary": _shortlist_shaping_summary(record),
        "verification_feedback_summary": _verification_feedback_summary(record),
        "counterfactual_alternatives": _counterfactual_alternatives(strategy),
    }

    execute = {
        "proposal": {
            "bo_top1_candidate": record.get("bo_top1_candidate"),
            "shortlist_candidates": record.get("shortlist_candidates", []),
        },
        "selection": {
            "llm_selected_candidate": record.get("llm_selected_candidate"),
            "llm_selected_candidate_rank_in_shortlist": record.get(
                "llm_selected_candidate_rank_in_shortlist"
            ),
            "selected_candidate": record.get("candidate"),
            "selected_candidate_rank_in_shortlist": record.get("selected_candidate_rank_in_shortlist"),
            "selected_differs_from_bo_top1": record.get("selected_differs_from_bo_top1", False),
            "selected_candidate_source": record.get("selected_candidate_source"),
            "selected_candidate_pool_source": record.get("selected_candidate_pool_source"),
        },
        "guardrail": {
            "enabled": record.get("override_guardrail_enabled"),
            "passed": record.get("override_guardrail_passed"),
            "action": record.get("override_guardrail_action"),
            "reason": record.get("override_guardrail_reason"),
            "score_margin": record.get("override_guardrail_score_margin"),
            "structural_shift": record.get("override_guardrail_structural_shift"),
            "strong_structural_shift": record.get("override_guardrail_strong_structural_shift"),
            "bo_scaffold_shift_count": record.get("override_guardrail_bo_scaffold_shift_count"),
            "dominant_scaffold_shift_count": record.get(
                "override_guardrail_dominant_scaffold_shift_count"
            ),
            "planner_trust_policy": record.get("planner_trust_policy"),
            "trusted_planner_mode_active": record.get("trusted_planner_mode_active"),
            "trusted_planner_override_allowed": record.get("trusted_planner_override_allowed"),
            "trusted_planner_block_reason": record.get("trusted_planner_block_reason"),
            "blocked_by_trusted_planner_policy": record.get("blocked_by_trusted_planner_policy"),
        },
        "structural_context": {
            "bo_top1_scaffold_key": record.get("bo_top1_scaffold_key"),
            "dominant_scaffold_key": record.get("dominant_scaffold_key"),
            "diversity_pool_quality": record.get("diversity_pool_quality"),
            "structural_shift_candidate_count": record.get("structural_shift_candidate_count"),
            "dominant_shift_candidate_count": record.get("dominant_shift_candidate_count"),
        },
        "override_reason_summary": _override_reason_summary(record, plan_goal),
        "selection_bias": _exploration_exploitation_label(record),
        "execution_summary": _execution_summary(record, plan_goal),
    }

    reflect = _reflect_summary(record, plan["plan_summary"])

    return {
        "observe": observe,
        "diagnose": diagnose,
        "plan": plan,
        "execute": execute,
        "reflect": reflect,
    }


def enrich_trace_records_with_workflow(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for raw_record in records:
        record = dict(raw_record)
        record["workflow_snapshot"] = build_workflow_snapshot(record)
        enriched.append(record)
    return enriched


def _one_screen_summary(record: dict[str, Any], workflow: dict[str, Any]) -> str:
    observe = workflow["observe"]
    diagnose = workflow["diagnose"]
    plan = workflow["plan"]
    execute = workflow["execute"]
    reflect = workflow["reflect"]
    result = _round_or_none(record.get("result"), digits=4)
    lines = [
        (
            "Observe: best improvement over the last 3 rounds="
            f"{observe['best_improvement_last_3']}, coverage={observe['coverage_overall_ratio']}, "
            f"duplicate ratio={observe['recent_duplicate_ratio']}."
        ),
        f"Diagnose: {diagnose['diagnosis_summary']}",
        (
            "Plan: "
            f"{plan['plan_summary']} "
            f"(intent={plan.get('budget_intent')}, shortlist_policy={plan.get('shortlist_policy')})."
        ),
        (
            "Execute: "
            f"{execute['execution_summary']} Result={result}. "
            f"Override={execute['selection']['selected_differs_from_bo_top1']}."
        ),
        f"Reflect: {reflect['what_we_learned']}",
    ]
    return " ".join(lines)


def build_decision_flow(records: list[dict[str, Any]], summary: dict[str, Any]) -> list[dict[str, Any]]:
    flow: list[dict[str, Any]] = []
    objective_name = summary.get("objective_name", "objective")
    for record in records:
        workflow = record.get("workflow_snapshot") or build_workflow_snapshot(record)
        flow.append(
            {
                "iteration": record.get("iteration"),
                "workflow_stage_bundle": workflow,
                "decision_card": {
                    "current_best": workflow["observe"]["current_best_value"],
                    "diagnosed_issue": workflow["diagnose"]["problem_state"],
                    "chosen_strategy": workflow["plan"]["selected_strategy"],
                    "budget_intent": workflow["plan"].get("budget_intent"),
                    "shortlist_policy": workflow["plan"].get("shortlist_policy"),
                    "bo_top1_overridden": workflow["execute"]["selection"]["selected_differs_from_bo_top1"],
                    "llm_requested_override": record.get("llm_requested_override", False),
                    "override_guardrail_action": record.get("override_guardrail_action"),
                    "final_candidate": record.get("candidate"),
                    "observed_outcome": {
                        objective_name: record.get("result"),
                        "improved_best": record.get("improved_best"),
                    },
                },
                "one_screen_summary": _one_screen_summary(record, workflow),
            }
        )
    return flow


def _previous_best(records: list[dict[str, Any]], index: int) -> float | None:
    best: float | None = None
    for record in records[:index]:
        result = _as_float(record.get("result"), default=None)
        if result is None:
            continue
        best = result if best is None else max(best, result)
    return best


def build_override_report(records: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    strategy_counts = {
        "direct_pick_count": sum(1 for row in records if _execution_strategy(row) in {"bo_direct", "direct_bo_pick"}),
        "shape_only_count": sum(1 for row in records if _execution_strategy(row) == "shape_only_bo_pick"),
        "alt_pick_count": sum(
            1 for row in records if _execution_strategy(row) in {"bo_rerank_topk", "shortlist_alt_pick"}
        ),
        "focused_alt_pick_count": sum(1 for row in records if _execution_strategy(row) in {"bo_focus_then_rerank", "focused_shortlist_alt_pick"}),
    }
    rerank_count = strategy_counts["alt_pick_count"] + strategy_counts["focused_alt_pick_count"]
    override_records = [row for row in records if bool(row.get("selected_differs_from_bo_top1"))]
    requested_override_records = [row for row in records if bool(row.get("llm_requested_override"))]
    blocked_override_records = [
        row
        for row in requested_override_records
        if str(row.get("override_guardrail_action")) == "fallback_to_bo_top1"
    ]
    trusted_policy_blocked_records = [
        row for row in blocked_override_records if bool(row.get("blocked_by_trusted_planner_policy"))
    ]
    focus_records = [row for row in records if _execution_strategy(row) in {"bo_focus_then_rerank", "focused_shortlist_alt_pick"}]
    override_cases = []
    override_successes = 0
    for idx, row in enumerate(records):
        if not bool(row.get("selected_differs_from_bo_top1")):
            continue
        previous_best = _previous_best(records, idx)
        result = _as_float(row.get("result"), default=None)
        observed_delta = None
        if previous_best is not None and result is not None:
            observed_delta = round(result - previous_best, 6)
        if bool(row.get("improved_best")):
            override_successes += 1
        workflow = row.get("workflow_snapshot") or build_workflow_snapshot(row)
        override_cases.append(
            {
                "iteration": row.get("iteration"),
                "plain_bo_would_choose": row.get("bo_top1_candidate"),
                "llm_requested_candidate": row.get("llm_selected_candidate"),
                "agent_chose": row.get("candidate"),
                "selected_candidate_rank_in_shortlist": row.get("selected_candidate_rank_in_shortlist"),
                "selected_candidate_source": row.get("selected_candidate_source"),
                "override_guardrail_action": row.get("override_guardrail_action"),
                "override_guardrail_reason": row.get("override_guardrail_reason"),
                "override_guardrail_score_margin": row.get("override_guardrail_score_margin"),
                "override_guardrail_structural_shift": row.get("override_guardrail_structural_shift"),
                "override_guardrail_strong_structural_shift": row.get(
                    "override_guardrail_strong_structural_shift"
                ),
                "planner_trust_policy": row.get("planner_trust_policy"),
                "trusted_planner_mode_active": row.get("trusted_planner_mode_active"),
                "trusted_planner_block_reason": row.get("trusted_planner_block_reason"),
                "blocked_by_trusted_planner_policy": row.get("blocked_by_trusted_planner_policy"),
                "override_reason": workflow["execute"]["override_reason_summary"],
                "observed_result": row.get("result"),
                "observed_best_delta": observed_delta,
                "improved_best": row.get("improved_best"),
            }
        )

    override_count = len(override_records)
    num_rounds = len(records)
    report = {
        "workflow_report_version": WORKFLOW_REPORT_VERSION,
        "dataset": summary.get("dataset"),
        "planner_name": summary.get("planner_name"),
        "method_name": summary.get("method_name"),
        "method_family": summary.get("method_family"),
        "num_decision_rounds": num_rounds,
        **strategy_counts,
        "rerank_intervention_count": rerank_count,
        "contract_failed_count": sum(1 for row in records if row.get("contract_satisfied") is False),
        "shape_contract_satisfied_count": sum(
            1
            for row in records
            if _execution_strategy(row) == "shape_only_bo_pick" and bool(row.get("shape_contract_satisfied"))
        ),
        "alt_pick_success_count": sum(
            1
            for row in records
            if _execution_strategy(row) in {
                "bo_rerank_topk",
                "shortlist_alt_pick",
                "bo_focus_then_rerank",
                "focused_shortlist_alt_pick",
            }
            and bool(row.get("selected_differs_from_bo_top1"))
        ),
        "llm_requested_override_count": len(requested_override_records),
        "guardrail_blocked_override_count": len(blocked_override_records),
        "blocked_by_trusted_planner_policy": len(trusted_policy_blocked_records),
        "executed_override_count": override_count,
        "override_count": override_count,
        "override_rate": round(override_count / num_rounds, 6) if num_rounds else 0.0,
        "cross_scaffold_override_count": sum(
            1 for row in override_records if str(row.get("selected_candidate_source")) == "diversity_injected"
        ),
        "focus_activation_count": len(focus_records),
        "focus_fallback_count": sum(1 for row in records if row.get("focus_fallback_reason")),
        "override_success_rate": (
            round(override_successes / override_count, 6) if override_count else 0.0
        ),
        "executed_override_success_rate": (
            round(override_successes / override_count, 6) if override_count else 0.0
        ),
        "override_success_rate_definition": (
            "Immediate success rate: fraction of override rounds where the chosen candidate "
            "improved best-so-far in the same iteration."
        ),
        "plain_bo_counterfactual_summary": override_cases,
    }
    report["shape_only_effective_count"] = report["shape_contract_satisfied_count"]
    return report


def render_decision_flow_markdown(flow: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Scientist Workflow Decision Flow",
        "",
        f"- Dataset: `{summary.get('dataset')}`",
        f"- Planner: `{summary.get('planner_name')}`",
        f"- Method: `{summary.get('method_name')}`",
        "",
    ]
    for entry in flow:
        card = entry["decision_card"]
        lines.extend(
            [
                f"## Iteration {entry['iteration']}",
                "",
                entry["one_screen_summary"],
                "",
                f"- Current best: `{card['current_best']}`",
                f"- Diagnosed issue: `{card['diagnosed_issue']}`",
                f"- Chosen strategy: `{card['chosen_strategy']}`",
                f"- BO top-1 overridden: `{card['bo_top1_overridden']}`",
                f"- Final candidate: `{json.dumps(card['final_candidate'], ensure_ascii=False)}`",
                f"- Observed outcome: `{json.dumps(card['observed_outcome'], ensure_ascii=False)}`",
                "",
            ]
        )
    return "\n".join(lines)


def render_override_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Plain BO Override Report",
        "",
        f"- Dataset: `{report.get('dataset')}`",
        f"- Planner: `{report.get('planner_name')}`",
        f"- Method: `{report.get('method_name')}`",
        f"- Decision rounds: `{report.get('num_decision_rounds')}`",
        f"- Rerank intervention count: `{report.get('rerank_intervention_count')}`",
        f"- LLM requested override count: `{report.get('llm_requested_override_count')}`",
        f"- Guardrail blocked override count: `{report.get('guardrail_blocked_override_count')}`",
        f"- Blocked by trusted planner policy: `{report.get('blocked_by_trusted_planner_policy')}`",
        f"- Executed override count: `{report.get('executed_override_count')}`",
        f"- Override count: `{report.get('override_count')}`",
        f"- Override rate: `{report.get('override_rate')}`",
        f"- Cross-scaffold override count: `{report.get('cross_scaffold_override_count')}`",
        f"- Focus activation count: `{report.get('focus_activation_count')}`",
        f"- Focus fallback count: `{report.get('focus_fallback_count')}`",
        f"- Override success rate: `{report.get('override_success_rate')}`",
        f"- Override success definition: {report.get('override_success_rate_definition')}",
        "",
    ]
    cases = report.get("plain_bo_counterfactual_summary", [])
    if cases:
        lines.extend(
            [
                "## Override Cases",
                "",
                "| Iter | Source | Improved best | Observed best delta |",
                "|---|---|---:|---:|",
            ]
        )
        for case in cases:
            lines.append(
                f"| {case['iteration']} | {case['selected_candidate_source']} | "
                f"{case['improved_best']} | {case['observed_best_delta']} |"
            )
        lines.append("")
    return "\n".join(lines)
