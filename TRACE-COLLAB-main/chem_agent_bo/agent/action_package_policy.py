"""Policy helpers for Action Package stabilization and v0.6 execution actions."""

from __future__ import annotations

from typing import Any


V06_EXECUTION_ACTIONS = (
    "direct_bo_pick",
    "shape_only_bo_pick",
    "shape_then_probe_topk",
    "shortlist_alt_pick",
    "focused_shortlist_alt_pick",
    "finite_pool_candidate_probe",
    "mask_scaffold_corridor_resuggest",
    "mask_dominant_resuggest",
    "mask_low_repeat_resuggest",
)

PRE_SHORTLIST_RESUGGEST_ACTIONS = (
    "finite_pool_candidate_probe",
    "mask_scaffold_corridor_resuggest",
    "mask_dominant_resuggest",
    "mask_low_repeat_resuggest",
)

SHORTLIST_LEVEL_ACTIONS = (
    "shape_only_bo_pick",
    "shape_then_probe_topk",
    "shortlist_alt_pick",
    "focused_shortlist_alt_pick",
)

RARE_SHORTLIST_ACTIONS = (
    "shape_then_probe_topk",
    "shortlist_alt_pick",
    "focused_shortlist_alt_pick",
)

DEFAULT_SELECTION_AUTHORITY_LEVEL = "planner_only"


def default_planner_trigger_thresholds() -> dict[str, Any]:
    return {
        "stagnation_no_improvement_rounds": 4,
        "local_lock_score_threshold": 0.62,
        "anchor_repeat_threshold": 2,
        "recent_duplicate_ratio_threshold": 0.25,
        "scaffold_plane_lock_score_threshold": 0.58,
        "coverage_weighted_ratio_threshold": 0.34,
        "coverage_min_ratio_threshold": 0.30,
        "scaffold_plane_lock_no_improvement_rounds": 6,
        "failed_action_family_rounds_for_scaffold_plane_lock": 2,
        "focus_admissible_no_improvement_rounds": 6,
        "failed_shape_only_rounds_for_resuggest": 2,
        "failed_alt_rounds_threshold": 1,
        "shape_probe_no_improvement_rounds": 4,
        "resuggest_no_improvement_rounds": 5,
        "strong_alt_no_improvement_rounds": 6,
        "strong_focus_no_improvement_rounds": 8,
        "strong_focus_failed_action_rounds": 3,
        "early_trajectory_resuggest_no_improvement_rounds": 9,
        "early_trajectory_recent_primary_concentration_threshold": 0.50,
        "deescalate_failed_action_family_rounds": 2,
        "resuggest_requires_coverage_pressure": True,
        "shape_probe_requires_coverage_pressure": True,
        "early_trajectory_requires_coverage_pressure": True,
        "early_trajectory_allow_local_repeat_pressure": False,
    }


def default_planner_action_policies() -> dict[str, dict[str, Any]]:
    base_discrete_like_mainline = [
        "direct_bo_pick",
        "shape_only_bo_pick",
        "finite_pool_candidate_probe",
        "mask_low_repeat_resuggest",
        "mask_dominant_resuggest",
        "mask_scaffold_corridor_resuggest",
    ]
    discrete_like_rare = [
        "shape_then_probe_topk",
        "shortlist_alt_pick",
        "focused_shortlist_alt_pick",
    ]
    thresholds = default_planner_trigger_thresholds()
    return {
        "default": {
            "supports_problem_shaping": True,
            "supports_shortlist_probe": False,
            "default_selection_authority_level": "planner_only",
            "allowed_mainline_actions": list(base_discrete_like_mainline),
            "allowed_rare_actions": list(discrete_like_rare),
            "allowed_prompt_styles": ["default", "shape_only", "resuggest_bo_pick"],
            "allow_runtime_promotion": False,
            "allow_non_top_final_replacement": False,
            "non_top_replacement_requires_evidence_gate": True,
            "guardrail_mode": "veto_only",
            **thresholds,
        },
        "discrete": {
            "supports_problem_shaping": True,
            "supports_shortlist_probe": False,
            "default_selection_authority_level": "planner_only",
            "allowed_mainline_actions": list(base_discrete_like_mainline),
            "allowed_rare_actions": list(discrete_like_rare),
            "allowed_prompt_styles": ["default", "shape_only", "resuggest_bo_pick"],
            "allow_runtime_promotion": False,
            "allow_non_top_final_replacement": False,
            "non_top_replacement_requires_evidence_gate": True,
            "guardrail_mode": "veto_only",
            **thresholds,
        },
        "atlas": {
            "supports_problem_shaping": True,
            "supports_shortlist_probe": True,
            "default_selection_authority_level": "planner_only",
            "allowed_mainline_actions": [
                "direct_bo_pick",
                "shape_only_bo_pick",
            ],
            "allowed_rare_actions": list(discrete_like_rare),
            "allowed_prompt_styles": [
                "default",
                "shape_only",
                "challenger_with_incumbent",
                "shape_probe_topk",
                "candidate_direction_review",
            ],
            "allow_runtime_promotion": False,
            "allow_non_top_final_replacement": True,
            "non_top_replacement_requires_evidence_gate": True,
            "guardrail_mode": "veto_only",
            **thresholds,
        },
        "botorch_qei": {
            "supports_problem_shaping": True,
            "supports_shortlist_probe": False,
            "default_selection_authority_level": "planner_only",
            "allowed_mainline_actions": list(base_discrete_like_mainline),
            "allowed_rare_actions": list(discrete_like_rare),
            "allowed_prompt_styles": ["default", "shape_only", "resuggest_bo_pick"],
            "allow_runtime_promotion": False,
            "allow_non_top_final_replacement": False,
            "non_top_replacement_requires_evidence_gate": True,
            "guardrail_mode": "veto_only",
            **thresholds,
        },
        "botorch_qlogei": {
            "supports_problem_shaping": True,
            "supports_shortlist_probe": False,
            "default_selection_authority_level": "planner_only",
            "allowed_mainline_actions": list(base_discrete_like_mainline),
            "allowed_rare_actions": list(discrete_like_rare),
            "allowed_prompt_styles": ["default", "shape_only", "resuggest_bo_pick"],
            "allow_runtime_promotion": False,
            "allow_non_top_final_replacement": False,
            "non_top_replacement_requires_evidence_gate": True,
            "guardrail_mode": "veto_only",
            **thresholds,
        },
    }


def _normalize_string_list(values: Any) -> list[str]:
    if values is None:
        return []
    result: list[str] = []
    for item in list(values):
        text = str(item or "").strip()
        if text:
            result.append(text)
    return result


def resolve_planner_action_policy(
    *,
    planner_name: str,
    planner_action_policies: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    merged = default_planner_action_policies()
    for key, value in dict(planner_action_policies or {}).items():
        if not isinstance(value, dict):
            continue
        base = dict(merged.get(str(key), {}))
        base.update(value)
        merged[str(key)] = base
    requested = str(planner_name or "").strip()
    policy = dict(merged.get(requested) or merged.get("default") or {})
    mainline = _normalize_string_list(policy.get("allowed_mainline_actions"))
    rare = _normalize_string_list(policy.get("allowed_rare_actions"))
    policy["allowed_mainline_actions"] = mainline
    policy["allowed_rare_actions"] = rare
    policy["allowed_actions"] = list(dict.fromkeys([*mainline, *rare]))
    policy["allowed_prompt_styles"] = _normalize_string_list(policy.get("allowed_prompt_styles"))
    policy["supports_problem_shaping"] = bool(policy.get("supports_problem_shaping", True))
    policy["supports_shortlist_probe"] = bool(policy.get("supports_shortlist_probe", False))
    policy["allow_runtime_promotion"] = bool(policy.get("allow_runtime_promotion", False))
    policy["allow_non_top_final_replacement"] = bool(
        policy.get("allow_non_top_final_replacement", False)
    )
    policy["non_top_replacement_requires_evidence_gate"] = bool(
        policy.get("non_top_replacement_requires_evidence_gate", True)
    )
    policy["default_selection_authority_level"] = str(
        policy.get("default_selection_authority_level", DEFAULT_SELECTION_AUTHORITY_LEVEL)
        or DEFAULT_SELECTION_AUTHORITY_LEVEL
    )
    policy["guardrail_mode"] = str(policy.get("guardrail_mode", "veto_only") or "veto_only")
    policy["planner_policy_name"] = requested or "default"
    return policy


def planner_policy_allows_runtime_promotion(policy: dict[str, Any] | None) -> bool:
    return bool((policy or {}).get("allow_runtime_promotion", False))


def planner_policy_shortlist_probe_enabled(policy: dict[str, Any] | None) -> bool:
    return bool((policy or {}).get("supports_shortlist_probe", False))


def planner_policy_allows_non_top_final_replacement(policy: dict[str, Any] | None) -> bool:
    return bool((policy or {}).get("allow_non_top_final_replacement", False))


def append_reasoning_note(reasoning: str, note: str) -> str:
    tag = f"[policy realignment: {note}]"
    base = str(reasoning or "").strip()
    if tag in base:
        return base
    return f"{base} {tag}".strip()


def controller_policy_signals(
    *,
    decision_context: dict[str, Any],
    controller_trigger_reasons: list[str],
) -> dict[str, Any]:
    planner_policy = resolve_planner_action_policy(
        planner_name=str(decision_context.get("planner_name", "default") or "default"),
        planner_action_policies=decision_context.get("planner_action_policies"),
    )
    trigger_set = {str(item).strip() for item in (controller_trigger_reasons or []) if str(item).strip()}
    stagnation_no_improvement_rounds = int(
        planner_policy.get("stagnation_no_improvement_rounds", 4) or 4
    )
    local_lock_score_threshold = float(
        planner_policy.get("local_lock_score_threshold", 0.62) or 0.62
    )
    anchor_repeat_threshold = int(planner_policy.get("anchor_repeat_threshold", 2) or 2)
    recent_duplicate_ratio_threshold = float(
        planner_policy.get("recent_duplicate_ratio_threshold", 0.25) or 0.25
    )
    scaffold_plane_lock_score_threshold = float(
        planner_policy.get("scaffold_plane_lock_score_threshold", 0.58) or 0.58
    )
    coverage_weighted_ratio_threshold = float(
        planner_policy.get("coverage_weighted_ratio_threshold", 0.34) or 0.34
    )
    coverage_min_ratio_threshold = float(
        planner_policy.get("coverage_min_ratio_threshold", 0.30) or 0.30
    )
    scaffold_plane_lock_no_improvement_rounds = int(
        planner_policy.get("scaffold_plane_lock_no_improvement_rounds", 6) or 6
    )
    failed_action_family_rounds_for_scaffold_plane_lock = int(
        planner_policy.get("failed_action_family_rounds_for_scaffold_plane_lock", 2) or 2
    )
    no_improvement_rounds = int(decision_context.get("no_improvement_rounds", 0) or 0)
    remaining_budget = int(decision_context.get("remaining_budget", 0) or 0)
    total_budget = int(decision_context.get("total_budget", 0) or 0)
    best_improvement_last_3 = float(decision_context.get("best_improvement_last_3") or 0.0)
    coverage_weighted_key_dim_ratio = float(
        decision_context.get("coverage_weighted_key_dim_ratio", 0.0) or 0.0
    )
    coverage_min_key_dim_ratio = float(
        decision_context.get("coverage_min_key_dim_ratio", 0.0) or 0.0
    )
    local_lock_score = float(decision_context.get("local_lock_score", 0.0) or 0.0)
    scaffold_plane_lock_score = float(
        decision_context.get("scaffold_plane_lock_score", 0.0) or 0.0
    )
    scaffold_plane_lock_detected = bool(
        decision_context.get("scaffold_plane_lock_detected", False)
    )
    recent_scaffold_concentration = float(
        decision_context.get("recent_scaffold_concentration", 0.0) or 0.0
    )
    anchor_repeat_count = int(decision_context.get("anchor_repeat_count", 0) or 0)
    verification_warns_extension = bool(
        decision_context.get("verification_warns_extension", False)
    )
    verification_identity_uncertain = bool(
        decision_context.get("verification_identity_uncertain", False)
    )
    verification_status_last_round = str(
        decision_context.get("verification_status_last_round", "not_run") or "not_run"
    ).strip()
    iteration = int(decision_context.get("iteration", 0) or 0)
    last_action_package = decision_context.get("last_action_package")
    recent_duplicate_ratio = float(decision_context.get("recent_duplicate_ratio", 0.0) or 0.0)
    underexplored_dimensions = [
        str(item).strip()
        for item in list(decision_context.get("underexplored_dimensions", []) or [])
        if str(item).strip()
    ]
    last_action_effective = bool(decision_context.get("last_action_effective", False))
    consecutive_failed_action_family_rounds = int(
        decision_context.get("consecutive_failed_action_family_rounds", 0) or 0
    )
    consecutive_failed_selection_policy_rounds = int(
        decision_context.get("consecutive_failed_selection_policy_rounds", 0) or 0
    )
    last_selection_policy = str(
        decision_context.get("last_selection_policy", "") or ""
    ).strip()

    late_budget = (
        remaining_budget <= max(3, int(total_budget * 0.25))
        if total_budget > 0
        else remaining_budget <= 3
    )
    fresh_progress = best_improvement_last_3 > 0.0
    stagnation_pressure = (
        "stagnation" in trigger_set
        or no_improvement_rounds >= stagnation_no_improvement_rounds
    )
    local_repeat_pressure = (
        local_lock_score >= local_lock_score_threshold
        or anchor_repeat_count >= anchor_repeat_threshold
        or bool(decision_context.get("one_axis_sweep_detected", False))
        or recent_duplicate_ratio >= recent_duplicate_ratio_threshold
        or any(
            item in trigger_set
            for item in (
                "local_lock_stall",
                "one_axis_local_lock",
                "anchor_repeat",
                "key_dim_concentration_high",
            )
        )
    )
    scaffold_plane_lock_pressure = (
        scaffold_plane_lock_score >= scaffold_plane_lock_score_threshold
        or scaffold_plane_lock_detected
        or (
            no_improvement_rounds >= scaffold_plane_lock_no_improvement_rounds
            and coverage_weighted_key_dim_ratio < coverage_weighted_ratio_threshold
            and bool(underexplored_dimensions)
        )
        or (
            no_improvement_rounds >= scaffold_plane_lock_no_improvement_rounds
            and coverage_min_key_dim_ratio < coverage_min_ratio_threshold
            and consecutive_failed_action_family_rounds
            >= failed_action_family_rounds_for_scaffold_plane_lock
        )
    )
    escalation_pressure = local_repeat_pressure or scaffold_plane_lock_pressure
    coverage_pressure = (
        coverage_weighted_key_dim_ratio < coverage_weighted_ratio_threshold
        or any(
            item in trigger_set
            for item in (
                "coverage_low",
                "key_dim_coverage_low",
                "key_dim_underexplored",
            )
        )
    )
    bootstrap_direct = (
        iteration <= 1
        and not isinstance(last_action_package, dict)
        and verification_status_last_round == "not_run"
        and not verification_warns_extension
        and not verification_identity_uncertain
        and not local_repeat_pressure
    )
    healthy_direct = (
        bootstrap_direct
        or (
            no_improvement_rounds <= 1
            and fresh_progress
            and not verification_warns_extension
            and not verification_identity_uncertain
            and not stagnation_pressure
            and not escalation_pressure
        )
    )
    contrast_needed = (
        escalation_pressure
        or verification_warns_extension
        or verification_identity_uncertain
    )
    probe_pressure = (
        (stagnation_pressure and (escalation_pressure or coverage_pressure))
        or (
            verification_warns_extension
            and (not fresh_progress or escalation_pressure)
        )
        or (
            verification_identity_uncertain
            and late_budget
            and (not fresh_progress or escalation_pressure)
        )
    )
    shortlist_balance_pressure = (
        not healthy_direct
        and (
            coverage_pressure
            or escalation_pressure
            or verification_warns_extension
            or verification_identity_uncertain
            or (verification_status_last_round == "caution" and no_improvement_rounds > 0)
        )
    )
    degrade_to_shape_only = (
        shortlist_balance_pressure
        and not probe_pressure
    ) or (
        last_selection_policy == "select_from_shaped_shortlist"
        and not last_action_effective
        and consecutive_failed_selection_policy_rounds >= 1
    )
    deescalate_to_bo_direct = (
        last_selection_policy in {
            "bo_top1_from_shaped_shortlist",
            "select_from_shaped_shortlist",
        }
        and consecutive_failed_action_family_rounds >= 2
        and not escalation_pressure
        and not verification_warns_extension
        and not verification_identity_uncertain
    )
    prefer_strict_verification = late_budget and (
        probe_pressure
        or escalation_pressure
        or verification_warns_extension
        or verification_identity_uncertain
    )
    return {
        "trigger_set": trigger_set,
        "no_improvement_rounds": no_improvement_rounds,
        "late_budget": late_budget,
        "bootstrap_direct": bootstrap_direct,
        "fresh_progress": fresh_progress,
        "stagnation_pressure": stagnation_pressure,
        "local_repeat_pressure": local_repeat_pressure,
        "scaffold_plane_lock_pressure": scaffold_plane_lock_pressure,
        "recent_scaffold_concentration": recent_scaffold_concentration,
        "escalation_pressure": escalation_pressure,
        "anchor_repeat_count": anchor_repeat_count,
        "coverage_pressure": coverage_pressure,
        "healthy_direct": healthy_direct,
        "probe_pressure": probe_pressure,
        "shortlist_balance_pressure": shortlist_balance_pressure,
        "degrade_to_shape_only": degrade_to_shape_only,
        "deescalate_to_bo_direct": deescalate_to_bo_direct,
        "contrast_needed": contrast_needed,
        "prefer_strict_verification": prefer_strict_verification,
        "planner_trigger_thresholds": {
            "stagnation_no_improvement_rounds": stagnation_no_improvement_rounds,
            "local_lock_score_threshold": local_lock_score_threshold,
            "anchor_repeat_threshold": anchor_repeat_threshold,
            "recent_duplicate_ratio_threshold": recent_duplicate_ratio_threshold,
            "scaffold_plane_lock_score_threshold": scaffold_plane_lock_score_threshold,
            "coverage_weighted_ratio_threshold": coverage_weighted_ratio_threshold,
            "coverage_min_ratio_threshold": coverage_min_ratio_threshold,
            "scaffold_plane_lock_no_improvement_rounds": scaffold_plane_lock_no_improvement_rounds,
            "failed_action_family_rounds_for_scaffold_plane_lock": failed_action_family_rounds_for_scaffold_plane_lock,
        },
    }


def realign_action_package_fields(
    *,
    intent: str,
    shortlist_policy: str,
    repeat_policy: str,
    selection_policy: str,
    verification_policy: str,
    focus_policy: str,
    reasoning: str,
    decision_context: dict[str, Any],
    controller_trigger_reasons: list[str],
) -> dict[str, Any]:
    signals = controller_policy_signals(
        decision_context=decision_context,
        controller_trigger_reasons=controller_trigger_reasons,
    )
    updated_reasoning = str(reasoning or "")
    if signals["healthy_direct"]:
        intent = "exploit"
        shortlist_policy = "plain"
        repeat_policy = "allow"
        selection_policy = "bo_top1"
        verification_policy = "normal"
        focus_policy = "full_space"
        updated_reasoning = append_reasoning_note(
            updated_reasoning,
            "fresh_progress_supports_bo_top1_execution",
        )
    elif signals["probe_pressure"]:
        if intent == "exploit":
            intent = "probe"
        selection_policy = "select_from_shaped_shortlist"
        if shortlist_policy == "plain":
            shortlist_policy = (
                "contrast_shape" if signals["contrast_needed"] else "coverage_shape"
            )
        if repeat_policy == "allow":
            repeat_policy = (
                "avoid_anchor_repeat"
                if signals["local_repeat_pressure"]
                else "avoid_near_duplicate"
            )
        if signals["prefer_strict_verification"]:
            verification_policy = "strict"
        updated_reasoning = append_reasoning_note(
            updated_reasoning,
            "stagnation_or_verification_pressure_requires_shaped_shortlist",
        )
    elif signals["deescalate_to_bo_direct"]:
        intent = "exploit"
        shortlist_policy = "plain"
        repeat_policy = "allow"
        selection_policy = "bo_top1"
        verification_policy = "normal"
        focus_policy = "full_space"
        updated_reasoning = append_reasoning_note(
            updated_reasoning,
            "repeated_ineffective_shortlist_intervention_deescalates_to_bo_top1",
        )
    elif signals["degrade_to_shape_only"]:
        if intent == "exploit":
            intent = "balance"
        selection_policy = "bo_top1_from_shaped_shortlist"
        if shortlist_policy == "plain":
            shortlist_policy = (
                "contrast_shape" if signals["contrast_needed"] else "coverage_shape"
            )
        if repeat_policy == "allow" and signals["local_repeat_pressure"]:
            repeat_policy = "avoid_near_duplicate"
        if signals["prefer_strict_verification"] and verification_policy == "normal":
            verification_policy = "strict"
        updated_reasoning = append_reasoning_note(
            updated_reasoning,
            "shortlist_structure_needs_intervention_without_full_selection_override",
        )
    elif signals["shortlist_balance_pressure"]:
        if selection_policy == "bo_top1":
            selection_policy = "bo_top1_from_shaped_shortlist"
        if intent == "exploit":
            intent = "balance"
        if shortlist_policy == "plain":
            shortlist_policy = (
                "contrast_shape" if signals["contrast_needed"] else "coverage_shape"
            )
        if repeat_policy == "allow" and signals["local_repeat_pressure"]:
            repeat_policy = "avoid_near_duplicate"
        if signals["prefer_strict_verification"] and verification_policy == "normal":
            verification_policy = "strict"
        updated_reasoning = append_reasoning_note(
            updated_reasoning,
            "coverage_or_repeat_pressure_keeps_shortlist_shaping_active",
        )
    return {
        "intent": intent,
        "shortlist_policy": shortlist_policy,
        "repeat_policy": repeat_policy,
        "selection_policy": selection_policy,
        "verification_policy": verification_policy,
        "focus_policy": focus_policy,
        "reasoning": updated_reasoning,
        "signals": signals,
    }


def execution_action_defaults(
    *,
    execution_action: str,
    signals: dict[str, Any],
    intent: str,
    shortlist_policy: str,
    repeat_policy: str,
    verification_policy: str,
    focus_variables: list[str],
    window_rounds: int,
    reasoning: str,
) -> dict[str, Any]:
    normalized_action = str(execution_action or "direct_bo_pick").strip()
    normalized_intent = str(intent or "balance").strip().lower()
    normalized_shortlist_policy = str(shortlist_policy or "plain").strip().lower()
    normalized_repeat_policy = str(repeat_policy or "allow").strip().lower()
    normalized_verification_policy = str(verification_policy or "normal").strip().lower()
    normalized_focus_variables = [str(item) for item in list(focus_variables or []) if str(item).strip()]
    normalized_window_rounds = max(0, int(window_rounds or 0))
    normalized_reasoning = str(reasoning or "")
    contrast_needed = bool(signals.get("contrast_needed", False))
    local_repeat_pressure = bool(signals.get("local_repeat_pressure", False))
    prefer_strict_verification = bool(signals.get("prefer_strict_verification", False))
    anchor_repeat_count = int(signals.get("anchor_repeat_count", 0) or 0)
    if normalized_shortlist_policy == "plain" and normalized_action != "direct_bo_pick":
        normalized_shortlist_policy = "contrast_shape" if contrast_needed else "coverage_shape"
    if normalized_repeat_policy == "allow" and normalized_action != "direct_bo_pick" and local_repeat_pressure:
        normalized_repeat_policy = (
            "avoid_anchor_repeat" if anchor_repeat_count >= 2 else "avoid_near_duplicate"
        )
    if prefer_strict_verification and normalized_verification_policy not in {"strict", "normal"}:
        normalized_verification_policy = "strict"
    if normalized_action in {
        "finite_pool_candidate_probe",
        "mask_scaffold_corridor_resuggest",
        "mask_dominant_resuggest",
        "mask_low_repeat_resuggest",
    }:
        return {
            "requested_execution_action": normalized_action,
            "intent": "probe" if normalized_intent == "exploit" else (normalized_intent or "probe"),
            "shortlist_policy": "contrast_shape" if contrast_needed else "coverage_shape",
            "repeat_policy": (
                "avoid_anchor_repeat" if local_repeat_pressure else "avoid_near_duplicate"
            ),
            "selection_policy": (
                "bo_top1_from_shaped_shortlist"
                if normalized_action == "finite_pool_candidate_probe"
                else "bo_top1"
            ),
            "verification_policy": (
                "strict" if prefer_strict_verification else normalized_verification_policy
            ),
            "focus_policy": "full_space",
            "focus_variables": [],
            "window_rounds": 0,
            "reasoning": normalized_reasoning,
        }
    if normalized_action == "direct_bo_pick":
        return {
            "requested_execution_action": normalized_action,
            "intent": "exploit" if normalized_intent == "probe" else (normalized_intent or "exploit"),
            "shortlist_policy": "plain",
            "repeat_policy": "allow",
            "selection_policy": "bo_top1",
            "verification_policy": (
                "strict" if normalized_verification_policy == "strict" else "normal"
            ),
            "focus_policy": "full_space",
            "focus_variables": [],
            "window_rounds": 0,
            "reasoning": normalized_reasoning,
        }
    if normalized_action == "shape_only_bo_pick":
        return {
            "requested_execution_action": normalized_action,
            "intent": "balance" if normalized_intent == "exploit" else (normalized_intent or "balance"),
            "shortlist_policy": normalized_shortlist_policy,
            "repeat_policy": normalized_repeat_policy,
            "selection_policy": "bo_top1_from_shaped_shortlist",
            "verification_policy": (
                "strict" if prefer_strict_verification else normalized_verification_policy
            ),
            "focus_policy": "full_space",
            "focus_variables": [],
            "window_rounds": 0,
            "reasoning": normalized_reasoning,
        }
    if normalized_action == "shape_then_probe_topk":
        return {
            "requested_execution_action": normalized_action,
            "intent": "probe" if normalized_intent == "exploit" else (normalized_intent or "probe"),
            "shortlist_policy": normalized_shortlist_policy,
            "repeat_policy": normalized_repeat_policy,
            "selection_policy": "select_from_shaped_shortlist",
            "verification_policy": (
                "strict" if prefer_strict_verification else normalized_verification_policy
            ),
            "focus_policy": "full_space",
            "focus_variables": [],
            "window_rounds": 0,
            "reasoning": normalized_reasoning,
        }
    if normalized_action == "focused_shortlist_alt_pick":
        return {
            "requested_execution_action": normalized_action,
            "intent": normalized_intent or "probe",
            "shortlist_policy": normalized_shortlist_policy,
            "repeat_policy": normalized_repeat_policy,
            "selection_policy": "select_from_shaped_shortlist",
            "verification_policy": (
                "strict" if prefer_strict_verification else normalized_verification_policy
            ),
            "focus_policy": "temporary_focus",
            "focus_variables": normalized_focus_variables,
            "window_rounds": max(1, normalized_window_rounds),
            "reasoning": normalized_reasoning,
        }
    return {
        "requested_execution_action": "shortlist_alt_pick",
        "intent": normalized_intent or "balance",
        "shortlist_policy": normalized_shortlist_policy,
        "repeat_policy": normalized_repeat_policy,
        "selection_policy": "select_from_shaped_shortlist",
        "verification_policy": (
            "strict" if prefer_strict_verification else normalized_verification_policy
        ),
        "focus_policy": "full_space",
        "focus_variables": [],
        "window_rounds": 0,
        "reasoning": normalized_reasoning,
    }


def build_v06_action_admissibility(
    *,
    decision_context: dict[str, Any],
    controller_trigger_reasons: list[str],
) -> dict[str, Any]:
    signals = controller_policy_signals(
        decision_context=decision_context,
        controller_trigger_reasons=controller_trigger_reasons,
    )
    planner_policy = resolve_planner_action_policy(
        planner_name=str(decision_context.get("planner_name", "default") or "default"),
        planner_action_policies=decision_context.get("planner_action_policies"),
    )
    trigger_set = set(signals["trigger_set"])
    no_improvement_rounds = int(signals["no_improvement_rounds"])
    focus_admissible_no_improvement_rounds = int(
        planner_policy.get("focus_admissible_no_improvement_rounds", 6) or 6
    )
    failed_shape_only_rounds_for_resuggest = int(
        planner_policy.get("failed_shape_only_rounds_for_resuggest", 2) or 2
    )
    failed_alt_rounds_threshold = int(
        planner_policy.get("failed_alt_rounds_threshold", 1) or 1
    )
    shape_probe_no_improvement_rounds = int(
        planner_policy.get("shape_probe_no_improvement_rounds", 4) or 4
    )
    resuggest_no_improvement_rounds = int(
        planner_policy.get("resuggest_no_improvement_rounds", 5) or 5
    )
    strong_alt_no_improvement_rounds = int(
        planner_policy.get("strong_alt_no_improvement_rounds", 6) or 6
    )
    strong_focus_no_improvement_rounds = int(
        planner_policy.get("strong_focus_no_improvement_rounds", 8) or 8
    )
    strong_focus_failed_action_rounds = int(
        planner_policy.get("strong_focus_failed_action_rounds", 3) or 3
    )
    early_trajectory_resuggest_no_improvement_rounds = int(
        planner_policy.get("early_trajectory_resuggest_no_improvement_rounds", 9) or 9
    )
    early_trajectory_recent_primary_concentration_threshold = float(
        planner_policy.get(
            "early_trajectory_recent_primary_concentration_threshold",
            0.50,
        )
        or 0.50
    )
    resuggest_requires_coverage_pressure = bool(
        planner_policy.get("resuggest_requires_coverage_pressure", True)
    )
    shape_probe_requires_coverage_pressure = bool(
        planner_policy.get("shape_probe_requires_coverage_pressure", True)
    )
    early_trajectory_requires_coverage_pressure = bool(
        planner_policy.get("early_trajectory_requires_coverage_pressure", True)
    )
    early_trajectory_allow_local_repeat_pressure = bool(
        planner_policy.get("early_trajectory_allow_local_repeat_pressure", False)
    )
    focus_admissible = bool(
        (
            "coverage_low" in trigger_set
            or "key_dim_coverage_low" in trigger_set
            or signals["coverage_pressure"]
        )
        and (
            "scaffold_concentration_high" in trigger_set
            or "local_lock_stall" in trigger_set
            or signals["local_repeat_pressure"]
        )
        and no_improvement_rounds >= focus_admissible_no_improvement_rounds
    )

    last_executed_execution_action = str(
        decision_context.get("last_executed_execution_action", "") or ""
    ).strip()
    last_action_family = str(decision_context.get("last_action_family", "") or "").strip()
    last_action_effective = bool(decision_context.get("last_action_effective", False))
    failed_action_family_rounds = int(
        decision_context.get("consecutive_failed_action_family_rounds", 0) or 0
    )

    failed_shape_only_streak = (
        last_executed_execution_action == "shape_only_bo_pick"
        or last_action_family == "shape_only"
    ) and failed_action_family_rounds >= failed_shape_only_rounds_for_resuggest
    failed_alt_streak = (
        last_executed_execution_action in {"shortlist_alt_pick", "focused_shortlist_alt_pick"}
        or last_action_family == "alt_pick"
    ) and failed_action_family_rounds >= failed_alt_rounds_threshold
    early_trajectory_resuggest_evidence = bool(
        (
            signals["scaffold_plane_lock_pressure"]
            or (early_trajectory_allow_local_repeat_pressure and signals["local_repeat_pressure"])
        )
        and float(decision_context.get("recent_primary_dim_concentration", 0.0) or 0.0)
        >= early_trajectory_recent_primary_concentration_threshold
        and (
            signals["coverage_pressure"]
            or not early_trajectory_requires_coverage_pressure
        )
        and no_improvement_rounds >= early_trajectory_resuggest_no_improvement_rounds
        and (
            last_executed_execution_action == "shape_only_bo_pick"
            or last_action_family == "shape_only"
        )
        and failed_action_family_rounds >= 1
        and not last_action_effective
    )

    strong_alt_evidence = bool(
        signals["coverage_pressure"]
        and signals["escalation_pressure"]
        and no_improvement_rounds >= strong_alt_no_improvement_rounds
        and failed_shape_only_streak
        and not failed_alt_streak
        and not last_action_effective
    )
    strong_focus_evidence = bool(
        focus_admissible
        and strong_alt_evidence
        and no_improvement_rounds >= strong_focus_no_improvement_rounds
        and failed_action_family_rounds >= strong_focus_failed_action_rounds
    )
    shape_probe_evidence = bool(
        not signals["healthy_direct"]
        and (signals["probe_pressure"] or signals["shortlist_balance_pressure"])
        and (
            signals["coverage_pressure"]
            or not shape_probe_requires_coverage_pressure
        )
        and no_improvement_rounds >= shape_probe_no_improvement_rounds
        and failed_shape_only_streak
        and not last_action_effective
    )
    resuggest_evidence = bool(
        signals["escalation_pressure"]
        and (
            signals["coverage_pressure"]
            or not resuggest_requires_coverage_pressure
        )
        and no_improvement_rounds >= resuggest_no_improvement_rounds
        and failed_shape_only_streak
        and not last_action_effective
    )

    if signals["healthy_direct"]:
        admissible = ["direct_bo_pick", "shape_only_bo_pick"]
        preferred = "direct_bo_pick"
    elif signals["probe_pressure"] or signals["shortlist_balance_pressure"]:
        admissible = ["direct_bo_pick", "shape_only_bo_pick"]
        preferred = "shape_only_bo_pick"
    else:
        admissible = ["direct_bo_pick", "shape_only_bo_pick"]
        preferred = "shape_only_bo_pick" if signals["coverage_pressure"] else "direct_bo_pick"

    if shape_probe_evidence and "shape_then_probe_topk" not in admissible:
        admissible.append("shape_then_probe_topk")
        preferred = "shape_then_probe_topk"
    finite_pool_probe_evidence = bool(
        signals["coverage_pressure"]
        and signals["escalation_pressure"]
        and no_improvement_rounds >= max(3, shape_probe_no_improvement_rounds)
        and failed_shape_only_streak
        and not last_action_effective
    )
    if finite_pool_probe_evidence and "finite_pool_candidate_probe" not in admissible:
        admissible.append("finite_pool_candidate_probe")
        preferred = "finite_pool_candidate_probe"
    if resuggest_evidence or early_trajectory_resuggest_evidence:
        if "mask_scaffold_corridor_resuggest" not in admissible:
            admissible.append("mask_scaffold_corridor_resuggest")
        if "mask_low_repeat_resuggest" not in admissible:
            admissible.append("mask_low_repeat_resuggest")
        if "mask_dominant_resuggest" not in admissible:
            admissible.append("mask_dominant_resuggest")
        preferred = (
            "mask_scaffold_corridor_resuggest"
            if early_trajectory_resuggest_evidence
            else "mask_low_repeat_resuggest"
        )
    if strong_alt_evidence and "shortlist_alt_pick" not in admissible:
        admissible.append("shortlist_alt_pick")
        if not resuggest_evidence:
            preferred = "shortlist_alt_pick"
    if strong_focus_evidence and "focused_shortlist_alt_pick" not in admissible:
        admissible.append("focused_shortlist_alt_pick")
        if not resuggest_evidence:
            preferred = "focused_shortlist_alt_pick"
    return {
        "signals": signals,
        "admissible_execution_actions": admissible,
        "preferred_execution_action": preferred,
        "focus_admissible": focus_admissible,
        "shape_probe_evidence": shape_probe_evidence,
        "strong_alt_evidence": strong_alt_evidence,
        "strong_focus_evidence": strong_focus_evidence,
        "finite_pool_probe_evidence": finite_pool_probe_evidence,
        "resuggest_evidence": resuggest_evidence,
        "early_trajectory_resuggest_evidence": early_trajectory_resuggest_evidence,
    }


def normalize_v06_action_package(
    *,
    requested_execution_action: str,
    intent: str,
    shortlist_policy: str,
    repeat_policy: str,
    verification_policy: str,
    focus_variables: list[str],
    window_rounds: int,
    reasoning: str,
    decision_context: dict[str, Any],
    controller_trigger_reasons: list[str],
) -> dict[str, Any]:
    policy = build_v06_action_admissibility(
        decision_context=decision_context,
        controller_trigger_reasons=controller_trigger_reasons,
    )
    planner_policy = resolve_planner_action_policy(
        planner_name=str(decision_context.get("planner_name", "default") or "default"),
        planner_action_policies=decision_context.get("planner_action_policies"),
    )
    allowed_actions = set(planner_policy.get("allowed_actions", []) or [])
    admissible = [
        action for action in list(policy["admissible_execution_actions"]) if action in allowed_actions
    ]
    if not planner_policy_shortlist_probe_enabled(planner_policy):
        admissible = [
            action for action in admissible if action not in RARE_SHORTLIST_ACTIONS
        ]
    if not admissible:
        admissible = ["direct_bo_pick"]
    preferred = str(policy["preferred_execution_action"])
    if preferred not in admissible:
        preferred = (
            "shape_only_bo_pick"
            if "shape_only_bo_pick" in admissible
            else admissible[0]
        )
    requested = str(requested_execution_action or preferred).strip()
    fallback_reason = None
    updated_reasoning = str(reasoning or "")
    if requested not in admissible:
        fallback_reason = "requested_execution_action_not_admissible"
        requested = preferred
        updated_reasoning = append_reasoning_note(
            updated_reasoning,
            "requested_execution_action_fell_back_to_admissible_default",
        )
    elif (
        bool(policy.get("resuggest_evidence", False) or policy.get("early_trajectory_resuggest_evidence", False))
        and preferred in PRE_SHORTLIST_RESUGGEST_ACTIONS
        and requested in SHORTLIST_LEVEL_ACTIONS + ("direct_bo_pick",)
    ):
        fallback_reason = "preferred_resuggest_overrides_weaker_execution_action"
        requested = preferred
        updated_reasoning = append_reasoning_note(
            updated_reasoning,
            "strong_resuggest_evidence_promotes_pre_shortlist_action",
        )
    elif (
        bool(policy.get("shape_probe_evidence", False))
        and preferred == "shape_then_probe_topk"
        and requested in {"direct_bo_pick", "shape_only_bo_pick", "shortlist_alt_pick"}
    ):
        fallback_reason = "preferred_shape_probe_promotes_shortlist_selection"
        requested = preferred
        updated_reasoning = append_reasoning_note(
            updated_reasoning,
            "shape_probe_evidence_promotes_shortlist_comparison",
        )
    defaults = execution_action_defaults(
        execution_action=requested,
        signals=policy["signals"],
        intent=intent,
        shortlist_policy=shortlist_policy,
        repeat_policy=repeat_policy,
        verification_policy=verification_policy,
        focus_variables=focus_variables,
        window_rounds=window_rounds,
        reasoning=updated_reasoning,
    )
    if requested == "focused_shortlist_alt_pick" and not defaults["focus_variables"]:
        fallback_reason = "focused_action_missing_focus_variables"
        requested = "shortlist_alt_pick"
        defaults = execution_action_defaults(
            execution_action=requested,
            signals=policy["signals"],
            intent=intent,
            shortlist_policy=shortlist_policy,
            repeat_policy=repeat_policy,
            verification_policy=verification_policy,
            focus_variables=[],
            window_rounds=0,
            reasoning=append_reasoning_note(
                defaults["reasoning"],
                "focused_execution_action_fell_back_without_valid_focus_variables",
            ),
        )
    defaults["schema_version"] = "v0.6"
    defaults["admissible_execution_actions"] = admissible
    defaults["preferred_execution_action"] = preferred
    defaults["requested_execution_action"] = requested
    defaults["fallback_reason"] = fallback_reason
    defaults["policy_signals"] = policy["signals"]
    defaults["planner_action_policy_name"] = planner_policy.get("planner_policy_name")
    defaults["selection_authority_level"] = planner_policy.get(
        "default_selection_authority_level",
        DEFAULT_SELECTION_AUTHORITY_LEVEL,
    )
    defaults["allowed_mainline_actions"] = list(planner_policy.get("allowed_mainline_actions", []))
    defaults["allowed_rare_actions"] = list(planner_policy.get("allowed_rare_actions", []))
    return defaults
