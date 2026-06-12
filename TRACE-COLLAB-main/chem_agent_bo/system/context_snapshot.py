"""Lightweight helpers for audit-friendly context snapshots."""

from __future__ import annotations

from typing import Any


def build_decision_context_snapshot(decision_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "iteration": decision_context.get("iteration"),
        "observations": decision_context.get("observations"),
        "total_budget": decision_context.get("total_budget"),
        "remaining_budget": decision_context.get("remaining_budget"),
        "num_history": decision_context.get("num_history"),
        "best_observation": decision_context.get("best_observation"),
        "best_value": decision_context.get("best_value"),
        "best_improvement_last_3": decision_context.get("best_improvement_last_3"),
        "no_improvement_rounds": decision_context.get("no_improvement_rounds"),
        "recent_duplicate_ratio": decision_context.get("recent_duplicate_ratio"),
        "recent_result_std": decision_context.get("recent_result_std"),
        "coverage_overall_ratio": decision_context.get("coverage_overall_ratio"),
        "coverage_dimension_ratios": decision_context.get("coverage_dimension_ratios"),
        "coverage_dimension_metric_types": decision_context.get(
            "coverage_dimension_metric_types"
        ),
        "key_dimensions": decision_context.get("key_dimensions", []),
        "key_dimension_ratios": decision_context.get("key_dimension_ratios", {}),
        "coverage_min_key_dim_ratio": decision_context.get(
            "coverage_min_key_dim_ratio"
        ),
        "coverage_mean_key_dim_ratio": decision_context.get(
            "coverage_mean_key_dim_ratio"
        ),
        "coverage_weighted_key_dim_ratio": decision_context.get(
            "coverage_weighted_key_dim_ratio"
        ),
        "underexplored_dimensions": decision_context.get("underexplored_dimensions"),
        "recent_scaffold_concentration": decision_context.get(
            "recent_scaffold_concentration"
        ),
        "dominant_scaffold": decision_context.get("dominant_scaffold"),
        "recent_primary_dim_concentration": decision_context.get(
            "recent_primary_dim_concentration"
        ),
        "recent_secondary_dim_concentration": decision_context.get(
            "recent_secondary_dim_concentration"
        ),
        "dominant_values_by_dim": decision_context.get("dominant_values_by_dim", {}),
        "one_axis_sweep_detected": decision_context.get("one_axis_sweep_detected"),
        "one_axis_sweep_dimension": decision_context.get("one_axis_sweep_dimension"),
        "one_axis_sweep_anchor_dims": decision_context.get(
            "one_axis_sweep_anchor_dims",
            [],
        ),
        "one_axis_sweep_anchor_values": decision_context.get(
            "one_axis_sweep_anchor_values",
            {},
        ),
        "anchor_repeat_count": decision_context.get("anchor_repeat_count"),
        "local_lock_score": decision_context.get("local_lock_score"),
        "scaffold_plane_lock_score": decision_context.get(
            "scaffold_plane_lock_score"
        ),
        "scaffold_plane_lock_detected": decision_context.get(
            "scaffold_plane_lock_detected"
        ),
        "recent_feasibility_counts": decision_context.get("recent_feasibility_counts"),
        "current_stage_streak": decision_context.get("current_stage_streak"),
        "planner_name": decision_context.get("planner_name"),
        "planner_action_policy": decision_context.get("planner_action_policy", {}),
        "last_action_package": decision_context.get("last_action_package"),
        "last_action_effective": decision_context.get("last_action_effective"),
        "last_action_family": decision_context.get("last_action_family"),
        "last_requested_execution_action": decision_context.get(
            "last_requested_execution_action"
        ),
        "last_executed_execution_action": decision_context.get(
            "last_executed_execution_action"
        ),
        "last_contract_satisfied": decision_context.get("last_contract_satisfied"),
        "last_selection_policy": decision_context.get("last_selection_policy"),
        "last_shortlist_policy": decision_context.get("last_shortlist_policy"),
        "consecutive_failed_action_family_rounds": decision_context.get(
            "consecutive_failed_action_family_rounds"
        ),
        "consecutive_failed_selection_policy_rounds": decision_context.get(
            "consecutive_failed_selection_policy_rounds"
        ),
        "consecutive_failed_shortlist_policy_rounds": decision_context.get(
            "consecutive_failed_shortlist_policy_rounds"
        ),
        "action_feedback_state": {
            "last_action_effective": decision_context.get("last_action_effective"),
            "last_action_family": decision_context.get("last_action_family"),
            "last_requested_execution_action": decision_context.get(
                "last_requested_execution_action"
            ),
            "last_executed_execution_action": decision_context.get(
                "last_executed_execution_action"
            ),
            "last_contract_satisfied": decision_context.get("last_contract_satisfied"),
            "last_selection_policy": decision_context.get("last_selection_policy"),
            "last_shortlist_policy": decision_context.get("last_shortlist_policy"),
            "consecutive_failed_action_family_rounds": decision_context.get(
                "consecutive_failed_action_family_rounds"
            ),
            "consecutive_failed_selection_policy_rounds": decision_context.get(
                "consecutive_failed_selection_policy_rounds"
            ),
            "consecutive_failed_shortlist_policy_rounds": decision_context.get(
                "consecutive_failed_shortlist_policy_rounds"
            ),
        },
        "verification_status_last_round": decision_context.get(
            "verification_status_last_round"
        ),
        "verification_risk_flags_last_round": decision_context.get(
            "verification_risk_flags_last_round",
            [],
        ),
        "verification_warns_extension": decision_context.get(
            "verification_warns_extension",
            False,
        ),
        "verification_identity_uncertain": decision_context.get(
            "verification_identity_uncertain",
            False,
        ),
        "trace_compression": decision_context.get("trace_compression", {}),
        "trace_compressed_prefix": decision_context.get("trace_compressed_prefix", {}),
        "knowledge_meta": decision_context.get("knowledge_meta", {}),
        "working_memory_summary": decision_context.get("working_memory_summary", {}),
        "recent_history_tail": decision_context.get("recent_history_tail", []),
        "online_decision_state": decision_context.get("online_decision_state", {}),
    }
