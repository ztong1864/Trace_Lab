"""Deterministic guardrails for hypothesis-driven shortlist overrides."""

from __future__ import annotations

from typing import Any


def _shift_strength(item: dict[str, Any], reference_field: str) -> tuple[bool, int]:
    candidate = [str(value) for value in item.get("candidate_scaffold_key", []) or []]
    reference = [str(value) for value in item.get(reference_field, []) or []]
    if not candidate or not reference or len(candidate) != len(reference):
        return False, 0
    changed = [left != right for left, right in zip(candidate, reference)]
    primary_changed = bool(changed[0]) if changed else False
    return primary_changed, sum(1 for flag in changed if flag)


def score_by_index(
    rerank_action: dict[str, Any] | None,
    candidate_index: int | None,
) -> dict[str, Any]:
    if rerank_action is None or candidate_index is None:
        return {}
    for item in rerank_action.get("candidate_scores", []) or []:
        try:
            if int(item.get("candidate_index", -1)) == int(candidate_index):
                return dict(item)
        except (TypeError, ValueError):
            continue
    return {}


def _contrastive_evidence_by_index(
    candidate_contrastive_evidence: dict[str, Any] | None,
    candidate_index: int | None,
) -> dict[str, Any]:
    if candidate_contrastive_evidence is None or candidate_index is None:
        return {}
    by_index = dict(candidate_contrastive_evidence.get("by_candidate_index") or {})
    return dict(by_index.get(str(int(candidate_index))) or {})


def _support_gap_vs_bo_top1(evidence: dict[str, Any]) -> float | None:
    support_blocks = [
        dict(evidence.get("same_scaffold_support") or {}),
        dict(evidence.get("same_anchor_support") or {}),
        dict(evidence.get("analogue_support") or {}),
    ]
    deltas: list[float] = []
    for block in support_blocks:
        value = block.get("candidate_minus_bo_best")
        try:
            deltas.append(float(value))
        except (TypeError, ValueError):
            continue
    for item in evidence.get("changed_dimensions") or []:
        value = item.get("candidate_minus_bo_best")
        try:
            deltas.append(float(value))
        except (TypeError, ValueError):
            continue
    if not deltas:
        return None
    return max(deltas)


def _downside_risk_level(
    evidence: dict[str, Any],
    *,
    support_gap_vs_bo_top1: float | None,
) -> str:
    changed_scaffold_dims = list(evidence.get("changed_scaffold_dims") or [])
    changed_condition_dims = list(evidence.get("changed_condition_dims") or [])
    candidate_better_dimension_count = int(evidence.get("candidate_better_dimension_count", 0) or 0)
    bo_top1_better_dimension_count = int(evidence.get("bo_top1_better_dimension_count", 0) or 0)
    if support_gap_vs_bo_top1 is not None and support_gap_vs_bo_top1 <= -20.0:
        return "high"
    if (
        support_gap_vs_bo_top1 is not None
        and support_gap_vs_bo_top1 < -5.0
        and len(changed_condition_dims) >= 1
    ):
        return "high"
    if (
        not changed_scaffold_dims
        and len(changed_condition_dims) >= 2
        and support_gap_vs_bo_top1 is not None
        and support_gap_vs_bo_top1 < 0.0
    ):
        return "high"
    if (
        candidate_better_dimension_count == 0
        and bo_top1_better_dimension_count >= 2
        and support_gap_vs_bo_top1 is not None
        and support_gap_vs_bo_top1 < 0.0
    ):
        return "high"
    if support_gap_vs_bo_top1 is not None and support_gap_vs_bo_top1 < 0.0:
        return "medium"
    return "low"


def apply_override_guardrail(
    *,
    shortlist_candidates: list[dict[str, Any]],
    chosen_item: dict[str, Any],
    rerank_action: dict[str, Any] | None,
    candidate_contrastive_evidence: dict[str, Any] | None = None,
    enabled: bool = True,
    score_margin_threshold: float = 0.05,
    min_transfer_score: float = 0.65,
    require_structural_shift: bool = True,
    trusted_planner_mode_active: bool = False,
    trusted_planner_override_allowed: bool = True,
    trusted_planner_block_reason: str | None = None,
    trusted_main_pool_soft_override_allowed: bool = False,
    trusted_planner_score_margin_threshold: float = 0.10,
    trusted_planner_min_transfer_score: float = 0.75,
    trusted_planner_min_hypothesis_score: float = 0.80,
    late_stage_active: bool = False,
    strong_incumbent_present: bool = False,
    current_best_percentile: float | None = None,
    late_stage_incumbent_protection_enabled: bool = True,
    late_stage_override_score_margin_threshold: float = 0.12,
    late_stage_min_transfer_score: float = 0.80,
    late_stage_min_hypothesis_score: float = 0.85,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Allow only structurally meaningful or hypothesis-backed BO top-1 overrides."""

    chosen_shortlist_rank = next(
        (
            idx
            for idx, item in enumerate(shortlist_candidates)
            if int(item.get("candidate_index", idx))
            == int(chosen_item.get("candidate_index", -1))
        ),
        0,
    )
    bo_top1_item = next(
        (
            item
            for item in shortlist_candidates
            if bool(item.get("is_main_bo_top1"))
        ),
        next(
            (
                item
                for item in shortlist_candidates
                if item.get("main_pool_rank") is not None
                and int(item.get("main_pool_rank", -1) or -1) == 1
            ),
            next(
                (
                    item
                    for item in shortlist_candidates
                    if str(item.get("pool_source", "")) == "main_pool"
                    and int(item.get("bo_rank", -1) or -1) == 1
                ),
                shortlist_candidates[0] if shortlist_candidates else chosen_item,
            ),
        ),
    )
    bo_top1_index = int(bo_top1_item.get("candidate_index", 0))
    chosen_index = int(chosen_item.get("candidate_index", chosen_shortlist_rank))
    llm_requested_override = chosen_index != bo_top1_index
    router_preferred_probe_path = False
    effective_trusted_margin_threshold = float(trusted_planner_score_margin_threshold)
    if not enabled:
        return chosen_item, {
            "enabled": False,
            "passed": True,
            "action": "allow",
            "reason": "override_guardrail_disabled",
            "score_margin": None,
            "structural_shift": False,
            "llm_requested_override": llm_requested_override,
            "trusted_planner_mode_active": trusted_planner_mode_active,
            "trusted_planner_override_allowed": trusted_planner_override_allowed,
            "trusted_planner_block_reason": trusted_planner_block_reason,
            "trusted_main_pool_soft_override_allowed": trusted_main_pool_soft_override_allowed,
            "router_preferred_probe_path": router_preferred_probe_path,
            "effective_trusted_score_margin_threshold": (
                effective_trusted_margin_threshold if trusted_planner_mode_active else None
            ),
            "blocked_by_trusted_planner_policy": False,
            "late_stage_active": late_stage_active,
            "strong_incumbent_present": strong_incumbent_present,
            "current_best_percentile": current_best_percentile,
            "blocked_by_late_stage_incumbent_protection": False,
        }
    if not llm_requested_override:
        return chosen_item, {
            "enabled": True,
            "passed": True,
            "action": "allow",
            "reason": "llm_selected_bo_top1",
            "score_margin": 0.0,
            "structural_shift": False,
            "llm_requested_override": False,
            "trusted_planner_mode_active": trusted_planner_mode_active,
            "trusted_planner_override_allowed": trusted_planner_override_allowed,
            "trusted_planner_block_reason": trusted_planner_block_reason,
            "trusted_main_pool_soft_override_allowed": trusted_main_pool_soft_override_allowed,
            "router_preferred_probe_path": router_preferred_probe_path,
            "effective_trusted_score_margin_threshold": (
                effective_trusted_margin_threshold if trusted_planner_mode_active else None
            ),
            "blocked_by_trusted_planner_policy": False,
            "late_stage_active": late_stage_active,
            "strong_incumbent_present": strong_incumbent_present,
            "current_best_percentile": current_best_percentile,
            "blocked_by_late_stage_incumbent_protection": False,
        }

    chosen_score = score_by_index(rerank_action, chosen_index)
    bo_score = score_by_index(rerank_action, bo_top1_index)
    chosen_evidence = _contrastive_evidence_by_index(candidate_contrastive_evidence, chosen_index)
    support_gap_vs_bo_top1 = _support_gap_vs_bo_top1(chosen_evidence)
    downside_risk_level = _downside_risk_level(
        chosen_evidence,
        support_gap_vs_bo_top1=support_gap_vs_bo_top1,
    )
    state_router_guidance = dict((rerank_action or {}).get("state_router_guidance") or {})
    prompt_style = str((rerank_action or {}).get("prompt_style", "") or "")
    visible_evidence_state = str(state_router_guidance.get("visible_evidence_state", "") or "")
    preferred_selection_policy = str(
        state_router_guidance.get("preferred_selection_policy", "") or ""
    )
    preferred_candidate_indices: set[int] = set()
    for value in state_router_guidance.get("preferred_candidate_indices", []) or []:
        try:
            preferred_candidate_indices.add(int(value))
        except (TypeError, ValueError):
            continue
    chosen_overall = float(chosen_score.get("overall_score", 0.0) or 0.0)
    bo_overall = float(bo_score.get("overall_score", 0.0) or 0.0)
    score_margin = chosen_overall - bo_overall
    transfer_score = float(chosen_score.get("transfer_value_score", 0.0) or 0.0)
    hypothesis_score = float(chosen_score.get("hypothesis_value_score", 0.0) or 0.0)
    risk = str(chosen_score.get("local_overfit_risk", "medium"))
    bo_risk = str(bo_score.get("local_overfit_risk", "medium"))
    shift_type = str(chosen_score.get("structural_shift_type", "none"))
    primary_bo_shift, bo_shift_count = _shift_strength(chosen_item, "bo_top1_scaffold_key")
    primary_dominant_shift, dominant_shift_count = _shift_strength(
        chosen_item,
        "dominant_scaffold_key",
    )
    strong_structural_shift = (
        primary_bo_shift
        or primary_dominant_shift
        or bo_shift_count >= 2
        or dominant_shift_count >= 2
    )
    structural_shift = bool(
        chosen_item.get("structural_shift_from_bo_top1")
        or chosen_item.get("structural_shift_from_dominant")
        or shift_type in {"cross_scaffold_transfer", "mechanistic_contrast"}
    )
    from_diversity = str(chosen_item.get("pool_source")) == "diversity_pool"
    trusted_soft_main_pool_path = bool(
        trusted_planner_mode_active
        and not trusted_planner_override_allowed
        and trusted_main_pool_soft_override_allowed
        and not from_diversity
    )
    router_preferred_probe_path = bool(
        visible_evidence_state in {"local_lock", "rank_uncertain"}
        and preferred_selection_policy
        in {
            "low_repeat_probe",
            "challenger_pick",
            "shape_probe_topk",
            "deeper_diversity_probe",
        }
        and chosen_index in preferred_candidate_indices
    )
    resuggest_probe_topk_path = bool(
        prompt_style == "resuggest_probe_topk"
        and router_preferred_probe_path
    )
    same_scaffold_condition_probe_path = bool(
        resuggest_probe_topk_path
        and not bool(chosen_item.get("structural_shift_from_bo_top1"))
        and shift_type == "local_refinement"
        and int(chosen_item.get("bo_rank", 9999) or 9999) <= 3
        and bo_risk == "high"
        and risk in {"low", "medium"}
        and risk != bo_risk
        and downside_risk_level != "high"
    )
    score_margin_ok = score_margin >= float(score_margin_threshold)
    transfer_ok = (
        from_diversity
        and transfer_score >= float(min_transfer_score)
        and risk != "high"
        and downside_risk_level != "high"
    )
    hypothesis_ok = (
        score_margin_ok
        and hypothesis_score >= 0.6
        and shift_type in {"cross_scaffold_transfer", "mechanistic_contrast"}
        and downside_risk_level != "high"
    )
    structural_ok = (
        strong_structural_shift
        and (not require_structural_shift or risk != "high")
        and downside_risk_level != "high"
    )
    transfer_ok = transfer_ok and strong_structural_shift
    hypothesis_ok = hypothesis_ok and strong_structural_shift
    if trusted_planner_mode_active and not trusted_soft_main_pool_path:
        if router_preferred_probe_path:
            effective_trusted_margin_threshold = min(
                effective_trusted_margin_threshold,
                float(score_margin_threshold),
            )
        resuggest_probe_threshold_ok = bool(
            resuggest_probe_topk_path
            and score_margin >= min(effective_trusted_margin_threshold, max(float(score_margin_threshold), 0.05))
            and transfer_score >= max(float(min_transfer_score), 0.70)
            and hypothesis_score >= 0.75
            and (
                strong_structural_shift
                or shift_type in {"cross_scaffold_transfer", "mechanistic_contrast"}
            )
            and risk != "high"
            and downside_risk_level != "high"
        )
        same_scaffold_condition_threshold_ok = bool(
            same_scaffold_condition_probe_path
            and score_margin >= 0.01
            and transfer_score >= max(float(min_transfer_score), 0.75)
            and hypothesis_score >= 0.60
            and downside_risk_level != "high"
            and (
                support_gap_vs_bo_top1 is None
                or support_gap_vs_bo_top1 >= -1.0
            )
        )
        trusted_threshold_ok = (
            score_margin >= effective_trusted_margin_threshold
            and transfer_score >= float(trusted_planner_min_transfer_score)
            and hypothesis_score >= float(trusted_planner_min_hypothesis_score)
            and strong_structural_shift
            and risk != "high"
            and downside_risk_level != "high"
        )
        trusted_threshold_ok = (
            trusted_threshold_ok
            or resuggest_probe_threshold_ok
            or same_scaffold_condition_threshold_ok
        )
        if not trusted_planner_override_allowed or not trusted_threshold_ok:
            reason = trusted_planner_block_reason or "trusted_planner_requires_stronger_evidence"
            if trusted_planner_override_allowed and not trusted_threshold_ok:
                reason = "trusted_planner_override_evidence_below_threshold"
            return bo_top1_item, {
                "enabled": True,
                "passed": False,
                "action": "fallback_to_bo_top1",
                "reason": reason,
                "score_margin": round(score_margin, 6),
                "structural_shift": structural_shift,
                "strong_structural_shift": strong_structural_shift,
                "bo_scaffold_shift_count": bo_shift_count,
                "dominant_scaffold_shift_count": dominant_shift_count,
                "llm_requested_override": True,
                "transfer_value_score": transfer_score,
                "hypothesis_value_score": hypothesis_score,
                "structural_shift_type": shift_type,
                "trusted_planner_mode_active": True,
                "trusted_planner_override_allowed": trusted_planner_override_allowed,
                "trusted_planner_block_reason": reason,
                "trusted_main_pool_soft_override_allowed": trusted_main_pool_soft_override_allowed,
                "router_preferred_probe_path": router_preferred_probe_path,
                "resuggest_probe_topk_path": resuggest_probe_topk_path,
                "same_scaffold_condition_probe_path": same_scaffold_condition_probe_path,
                "support_gap_vs_bo_top1": support_gap_vs_bo_top1,
                "downside_risk_level": downside_risk_level,
                "effective_trusted_score_margin_threshold": effective_trusted_margin_threshold,
                "blocked_by_trusted_planner_policy": True,
                "late_stage_active": late_stage_active,
                "strong_incumbent_present": strong_incumbent_present,
                "current_best_percentile": current_best_percentile,
                "blocked_by_late_stage_incumbent_protection": False,
            }
    if late_stage_incumbent_protection_enabled and late_stage_active and strong_incumbent_present:
        late_stage_threshold_ok = (
            score_margin >= float(late_stage_override_score_margin_threshold)
            and transfer_score >= float(late_stage_min_transfer_score)
            and hypothesis_score >= float(late_stage_min_hypothesis_score)
            and strong_structural_shift
            and shift_type in {"cross_scaffold_transfer", "mechanistic_contrast"}
            and risk != "high"
        )
        if not late_stage_threshold_ok:
            return bo_top1_item, {
                "enabled": True,
                "passed": False,
                "action": "fallback_to_bo_top1",
                "reason": "late_stage_incumbent_protection_kept_bo_top1",
                "score_margin": round(score_margin, 6),
                "structural_shift": structural_shift,
                "strong_structural_shift": strong_structural_shift,
                "bo_scaffold_shift_count": bo_shift_count,
                "dominant_scaffold_shift_count": dominant_shift_count,
                "llm_requested_override": True,
                "transfer_value_score": transfer_score,
                "hypothesis_value_score": hypothesis_score,
                "structural_shift_type": shift_type,
                "trusted_planner_mode_active": trusted_planner_mode_active,
                "trusted_planner_override_allowed": trusted_planner_override_allowed,
                "trusted_planner_block_reason": trusted_planner_block_reason,
                "trusted_main_pool_soft_override_allowed": trusted_main_pool_soft_override_allowed,
                "blocked_by_trusted_planner_policy": False,
                "late_stage_active": True,
                "strong_incumbent_present": True,
                "current_best_percentile": current_best_percentile,
                "blocked_by_late_stage_incumbent_protection": True,
            }
    if structural_ok or transfer_ok or hypothesis_ok:
        reasons = []
        if structural_ok:
            reasons.append("structural_shift")
        if transfer_ok:
            reasons.append("diversity_transfer_score")
        if hypothesis_ok:
            reasons.append("hypothesis_score_margin")
        return chosen_item, {
            "enabled": True,
            "passed": True,
            "action": "allow",
            "reason": "+".join(reasons),
            "score_margin": round(score_margin, 6),
            "structural_shift": structural_shift,
            "strong_structural_shift": strong_structural_shift,
            "bo_scaffold_shift_count": bo_shift_count,
            "dominant_scaffold_shift_count": dominant_shift_count,
            "llm_requested_override": True,
            "transfer_value_score": transfer_score,
            "hypothesis_value_score": hypothesis_score,
            "structural_shift_type": shift_type,
            "support_gap_vs_bo_top1": support_gap_vs_bo_top1,
            "downside_risk_level": downside_risk_level,
            "trusted_planner_mode_active": trusted_planner_mode_active,
            "trusted_planner_override_allowed": trusted_planner_override_allowed,
            "trusted_planner_block_reason": trusted_planner_block_reason,
        "trusted_main_pool_soft_override_allowed": trusted_main_pool_soft_override_allowed,
        "router_preferred_probe_path": router_preferred_probe_path,
        "resuggest_probe_topk_path": resuggest_probe_topk_path,
        "same_scaffold_condition_probe_path": same_scaffold_condition_probe_path,
        "effective_trusted_score_margin_threshold": (
            effective_trusted_margin_threshold if trusted_planner_mode_active else None
        ),
            "blocked_by_trusted_planner_policy": False,
            "late_stage_active": late_stage_active,
            "strong_incumbent_present": strong_incumbent_present,
            "current_best_percentile": current_best_percentile,
            "blocked_by_late_stage_incumbent_protection": False,
        }

    return bo_top1_item, {
        "enabled": True,
        "passed": False,
        "action": "fallback_to_bo_top1",
        "reason": "blocked_override_without_sufficient_structural_shift_or_hypothesis_value",
        "score_margin": round(score_margin, 6),
        "structural_shift": structural_shift,
        "strong_structural_shift": strong_structural_shift,
        "bo_scaffold_shift_count": bo_shift_count,
        "dominant_scaffold_shift_count": dominant_shift_count,
        "llm_requested_override": True,
        "transfer_value_score": transfer_score,
        "hypothesis_value_score": hypothesis_score,
        "structural_shift_type": shift_type,
        "support_gap_vs_bo_top1": support_gap_vs_bo_top1,
        "downside_risk_level": downside_risk_level,
        "trusted_planner_mode_active": trusted_planner_mode_active,
        "trusted_planner_override_allowed": trusted_planner_override_allowed,
        "trusted_planner_block_reason": trusted_planner_block_reason,
        "trusted_main_pool_soft_override_allowed": trusted_main_pool_soft_override_allowed,
        "blocked_by_trusted_planner_policy": False,
        "late_stage_active": late_stage_active,
        "strong_incumbent_present": strong_incumbent_present,
        "current_best_percentile": current_best_percentile,
        "blocked_by_late_stage_incumbent_protection": False,
    }
