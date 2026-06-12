"""Shared controller runtime for TRACE recommendation decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chem_agent_bo.agent.action_package_policy import resolve_planner_action_policy
from chem_agent_bo.state import OnlineDecisionState

from .batch_composer import (
    LabBatchComposer,
    contract_to_dict,
    validation_report_to_dict,
)
from .policy import ActionCapabilityPolicy


@dataclass
class ControllerRuntimeConfig:
    planner_name: str = "atlas"
    planner_action_policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    enable_action_package_v2: bool = False
    enable_action_package_v06: bool = True
    verification_mode: str = "advisory"
    controller_reflection_input_mode: str = "full"
    lab_mode: bool = False


@dataclass
class ControllerDecision:
    controller_plan: dict[str, Any]
    action_package: dict[str, Any]
    selected_candidates: list[dict[str, Any]]
    shortlist_candidates: list[dict[str, Any]]
    semantic_assessments: list[dict[str, Any]]
    verification_passes: list[dict[str, Any] | None]
    evidence_trace: dict[str, Any]
    skill_trace: dict[str, Any]
    trace_records: list[dict[str, Any]]
    decision_context: dict[str, Any]
    diagnosis: dict[str, Any]
    hypothesis_action: dict[str, Any]
    coverage_insight: dict[str, Any]
    action_capability: dict[str, Any]
    rerank_action: dict[str, Any] | None = None
    batch_contract: dict[str, Any] | None = None
    batch_validation_report: dict[str, Any] | None = None


class ControllerRuntime:
    """Outcome-agnostic TRACE controller core.

    The runtime plans candidates and records action/evidence/skill traces. It
    does not evaluate candidates or call any environment oracle.
    """

    def __init__(
        self,
        *,
        decision_engine,
        config: ControllerRuntimeConfig | None = None,
        action_policy: ActionCapabilityPolicy | None = None,
    ) -> None:  # noqa: ANN001
        self.decision_engine = decision_engine
        self.config = config or ControllerRuntimeConfig()
        self.action_policy = action_policy or (
            ActionCapabilityPolicy.for_lab()
            if self.config.lab_mode
            else ActionCapabilityPolicy.for_benchmark()
        )
        self.online_state = OnlineDecisionState(
            prompt_config=getattr(self.decision_engine, "prompt_config", None),
            controller_reflection_input_mode=self.config.controller_reflection_input_mode,
        )

    def plan_batch(
        self,
        *,
        history: list[dict[str, Any]],
        candidate_pool: list[dict[str, Any]],
        batch_size: int,
        search_space,  # noqa: ANN001
        search_space_meta: dict[str, Any],
        reaction_context: dict[str, Any],
        goal: str,
        objective_name: str,
        iteration: int,
        observations: int,
        total_budget: int | None,
        knowledge_units: list[dict[str, Any]] | None = None,
        knowledge_meta: dict[str, Any] | None = None,
        evidence_trace: dict[str, Any] | None = None,
    ) -> ControllerDecision:
        if hasattr(self.decision_engine, "reset_skill_trace"):
            self.decision_engine.reset_skill_trace()
        best_observation = _best_observation(history, goal=goal, objective_name=objective_name)
        current_best = (
            None
            if best_observation is None
            else _safe_float(best_observation.get(objective_name))
        )
        self.online_state.refresh_from_history(
            history=history,
            best_observation=best_observation,
            current_best=current_best,
            iteration=iteration,
            observations=observations,
            total_budget=total_budget,
            search_space=search_space,
            search_space_meta=search_space_meta,
            goal=goal,
            knowledge_units=knowledge_units or [],
            knowledge_query=str((knowledge_meta or {}).get("query", "")),
            knowledge_meta=knowledge_meta or {},
        )
        decision_context = self._attach_planner_policy(
            self.online_state.build_decision_context(search_space_meta=search_space_meta)
        )
        trigger_reasons = _controller_trigger_reasons(decision_context, history)
        diagnosis = self.decision_engine.diagnose_stagnation(
            decision_context=decision_context,
            history_tail=decision_context.get("recent_history_tail", []),
        )
        hypothesis_action = self.decision_engine.generate_hypothesis(
            decision_context=decision_context,
            reaction_context=reaction_context,
            search_space=search_space,
        )
        coverage_insight = self.decision_engine.analyze_coverage(
            decision_context=decision_context,
            search_space=search_space,
        )
        controller_plan = self.decision_engine.choose_controller_plan(
            decision_context=decision_context,
            diagnosis=diagnosis,
            hypothesis_action=hypothesis_action,
            coverage_insight=coverage_insight,
            controller_trigger_reasons=trigger_reasons,
            search_space=search_space,
            enable_action_package_v2=bool(self.config.enable_action_package_v2),
            enable_action_package_v06=bool(self.config.enable_action_package_v06),
        )
        action_package = _normalize_action_request(
            controller_plan=controller_plan,
            action_package=dict(controller_plan.get("action_package") or {}),
            lab_mode=bool(self.config.lab_mode),
        )
        capability = self.action_policy.resolve(action_package)
        action_package["requested_execution_action"] = capability.requested_action
        action_package["executed_execution_action"] = capability.executed_action
        if capability.fallback_reason:
            action_package["fallback_reason"] = capability.fallback_reason
        controller_plan = {
            **controller_plan,
            "requested_execution_action": capability.requested_action,
            "executed_execution_action": capability.executed_action,
            "raw_requested_execution_action": action_package.get("raw_requested_execution_action"),
            "normalized_requested_execution_action": action_package.get(
                "normalized_requested_execution_action"
            ),
            "action_normalization_reason": action_package.get("action_normalization_reason"),
            "fallback_reason": capability.fallback_reason,
            "action_package": action_package,
        }

        batch_contract = None
        batch_validation_report = None
        if self.config.lab_mode:
            composition = LabBatchComposer(self.decision_engine).compose(
                candidate_pool=candidate_pool,
                batch_size=batch_size,
                decision_context=decision_context,
                controller_plan=controller_plan,
                diagnosis=diagnosis,
                hypothesis_action=hypothesis_action,
                coverage_insight=coverage_insight,
                search_space=search_space,
                reaction_context=reaction_context,
            )
            selected_items = composition.selected_items
            rerank_action = None
            batch_contract = contract_to_dict(composition.batch_contract)
            batch_validation_report = validation_report_to_dict(
                composition.validation_report
            )
            selection_trace = {
                "action": capability.executed_action,
                "batch_size": max(1, int(batch_size)),
                "pool_size": len(candidate_pool),
                "strategy": batch_contract.get("batch_strategy", "lab_batch_composition"),
                "batch_strategy": batch_contract.get("batch_strategy"),
                "batch_roles": [
                    item.get("batch_role") for item in selected_items
                ],
                "selected_candidate_indices": [
                    item.get("candidate_index") for item in selected_items
                ],
                "batch_validation_report": batch_validation_report,
            }
        else:
            selected_items, rerank_action, selection_trace = self._select_batch(
                action=capability.executed_action,
                batch_size=batch_size,
                candidate_pool=candidate_pool,
                decision_context=decision_context,
                controller_plan=controller_plan,
            )
        semantic_assessments: list[dict[str, Any]] = []
        verification_passes: list[dict[str, Any] | None] = []
        trace_records: list[dict[str, Any]] = []
        for rank, item in enumerate(selected_items, start=1):
            candidate = dict(item["candidate"])
            semantic = self.decision_engine.semantic_assessment(
                candidate=candidate,
                decision_context=decision_context,
                reaction_context=reaction_context,
            )
            verification = None
            verification_mode = str(self.config.verification_mode or "off").strip().lower()
            verification_policy = str(action_package.get("verification_policy", "normal")).strip().lower()
            if verification_mode != "off" and (
                verification_policy == "strict"
                or semantic.get("risk_level", "low") in {"medium", "high"}
            ):
                verification = self.decision_engine.candidate_verification(
                    candidate=candidate,
                    decision_context=decision_context,
                    reaction_context=reaction_context,
                    semantic_assessment=semantic,
                    controller_plan=controller_plan,
                )
                verification = {**verification, "verification_policy": verification_policy}
            semantic_assessments.append(semantic)
            verification_passes.append(verification)
            trace_records.append(
                {
                    "candidate": candidate,
                    "rank": rank,
                    "candidate_index": item.get("candidate_index"),
                    "planner_rank": item.get("bo_rank"),
                    "batch_role": item.get("batch_role"),
                    "batch_role_reason": item.get("batch_role_reason"),
                    "batch_slot": item.get("batch_slot"),
                    "descriptor_profile": item.get("descriptor_profile", {}),
                    "descriptor_contrast_to_anchor": item.get(
                        "descriptor_contrast_to_anchor",
                        {},
                    ),
                    "descriptor_signal": item.get("descriptor_signal", {}),
                    "batch_contract": batch_contract,
                    "batch_validation_report": batch_validation_report,
                    "controller_action": capability.executed_action,
                    "requested_controller_action": capability.requested_action,
                    "raw_requested_controller_action": action_package.get(
                        "raw_requested_execution_action"
                    ),
                    "action_normalization_reason": action_package.get(
                        "action_normalization_reason"
                    ),
                    "action_fallback_reason": capability.fallback_reason,
                    "controller_reasoning": controller_plan.get("reasoning", ""),
                    "action_package": action_package,
                    "selection_trace": selection_trace,
                    "semantic_assessment": semantic,
                    "verification_pass": verification,
                    "evidence_trace": evidence_trace or {},
                }
            )
        self.online_state.apply_round_outputs(
            diagnosis=diagnosis,
            hypothesis_action=hypothesis_action,
            coverage_insight=coverage_insight,
            semantic_assessment=semantic_assessments[0] if semantic_assessments else None,
            verification_pass=verification_passes[0] if verification_passes else None,
            controller_plan=controller_plan,
        )
        skill_trace = (
            self.decision_engine.skill_trace_snapshot()
            if hasattr(self.decision_engine, "skill_trace_snapshot")
            else {}
        )
        return ControllerDecision(
            controller_plan=controller_plan,
            action_package=action_package,
            selected_candidates=[dict(item["candidate"]) for item in selected_items],
            shortlist_candidates=candidate_pool,
            semantic_assessments=semantic_assessments,
            verification_passes=verification_passes,
            evidence_trace=evidence_trace or {},
            skill_trace=skill_trace,
            trace_records=trace_records,
            decision_context=decision_context,
            diagnosis=diagnosis,
            hypothesis_action=hypothesis_action,
            coverage_insight=coverage_insight,
            action_capability={
                **capability.to_trace(),
                "raw_requested_action": action_package.get("raw_requested_execution_action"),
                "normalized_requested_action": action_package.get(
                    "normalized_requested_execution_action"
                ),
                "normalization_reason": action_package.get("action_normalization_reason"),
            },
            rerank_action=rerank_action,
            batch_contract=batch_contract,
            batch_validation_report=batch_validation_report,
        )

    def reflect_after_result(
        self,
        *,
        history: list[dict[str, Any]],
        candidate: dict[str, Any],
        result: float,
        action_package: dict[str, Any] | None,
        semantic_assessment: dict[str, Any] | None,
        search_space,  # noqa: ANN001
        search_space_meta: dict[str, Any],
        goal: str,
        objective_name: str,
        iteration: int,
        observations: int,
        total_budget: int | None,
        knowledge_units: list[dict[str, Any]] | None = None,
        knowledge_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        best_observation = _best_observation(history, goal=goal, objective_name=objective_name)
        self.online_state.refresh_from_history(
            history=history,
            best_observation=best_observation,
            current_best=None if best_observation is None else _safe_float(best_observation.get(objective_name)),
            iteration=iteration,
            observations=observations,
            total_budget=total_budget,
            search_space=search_space,
            search_space_meta=search_space_meta,
            goal=goal,
            knowledge_units=knowledge_units or [],
            knowledge_meta=knowledge_meta or {},
        )
        decision_context = self._attach_planner_policy(
            self.online_state.build_decision_context(search_space_meta=search_space_meta)
        )
        feasibility = {
            "action": "accept",
            "reasoning": (
                (semantic_assessment or {}).get("soft_comment")
                or "Measured result imported through lab tell."
            ),
        }
        reflection = self.decision_engine.reflection_action(
            decision_action={
                "stage": "lab_tell",
                "active_variables": list(candidate.keys()),
                "decision_mode": "mixed",
                "reasoning": str((action_package or {}).get("reasoning", "")),
                "action_package": dict(action_package or {}),
            },
            candidate=candidate,
            feasibility_action=feasibility,
            result=float(result),
            decision_context=decision_context,
        )
        self.online_state.apply_round_outputs(
            semantic_assessment=semantic_assessment,
            reflection_action=reflection,
        )
        return reflection

    def _attach_planner_policy(self, decision_context: dict[str, Any]) -> dict[str, Any]:
        policy = resolve_planner_action_policy(
            planner_name=self.config.planner_name,
            planner_action_policies=self.config.planner_action_policies,
        )
        return {
            **decision_context,
            "planner_name": self.config.planner_name,
            "planner_action_policies": dict(self.config.planner_action_policies or {}),
            "planner_action_policy": policy,
        }

    def _select_batch(
        self,
        *,
        action: str,
        batch_size: int,
        candidate_pool: list[dict[str, Any]],
        decision_context: dict[str, Any],
        controller_plan: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
        if not candidate_pool:
            raise RuntimeError("ControllerRuntime requires a non-empty candidate pool.")
        requested = max(1, int(batch_size))
        rerank_action = None
        selection_trace = {
            "action": action,
            "batch_size": requested,
            "pool_size": len(candidate_pool),
            "strategy": "planner_order",
        }
        if action == "shape_only_bo_pick":
            ordered = _with_planner_anchor(
                anchor_pool=candidate_pool,
                ordered_pool=_shape_for_coverage(candidate_pool, decision_context),
            )
            selected = _select_diverse_batch(
                ordered,
                requested=requested,
                decision_context=decision_context,
                strategy="coverage_diversity_shape",
            )
            selection_trace["strategy"] = "coverage_diversity_shape"
            selection_trace.update(_batch_selection_summary(selected, decision_context))
            return selected, None, selection_trace
        if action == "shortlist_alt_pick":
            rerank_action = self.decision_engine.rerank_shortlist(
                decision_context=decision_context,
                shortlist_candidates=candidate_pool,
                controller_plan=controller_plan,
                rerank_policy={
                    "prompt_style": "default",
                    "state_router_guidance": {
                        "fallback_candidate_index": candidate_pool[0].get("candidate_index", 0),
                        "admissible_candidate_indices": [
                            item.get("candidate_index", idx)
                            for idx, item in enumerate(candidate_pool)
                        ],
                        "preferred_candidate_indices": [
                            item.get("candidate_index", idx)
                            for idx, item in enumerate(candidate_pool[:requested])
                        ],
                        "rationale": "Lab batch selection keeps planner top candidate visible while allowing bounded rerank.",
                    },
                },
            )
            ordered = _order_by_rerank(candidate_pool, rerank_action)
            selected = _select_diverse_batch(
                ordered,
                requested=requested,
                decision_context=decision_context,
                strategy="llm_shortlist_rerank",
            )
            selection_trace["strategy"] = "llm_shortlist_rerank"
            selection_trace["rerank_selected_index"] = rerank_action.get("selected_index")
            selection_trace.update(_batch_selection_summary(selected, decision_context))
            return selected, rerank_action, selection_trace
        selected = _select_diverse_batch(
            candidate_pool,
            requested=requested,
            decision_context=decision_context,
            strategy="planner_anchor_diversity",
        )
        selection_trace["strategy"] = "planner_anchor_diversity"
        selection_trace.update(_batch_selection_summary(selected, decision_context))
        return selected, None, selection_trace


def _controller_trigger_reasons(
    decision_context: dict[str, Any],
    history: list[dict[str, Any]],
) -> list[str]:
    reasons = ["lab_batch_request"]
    if not history:
        reasons.append("no_completed_observations")
    if int(decision_context.get("no_improvement_rounds", 0) or 0) > 0:
        reasons.append("recent_no_improvement")
    if decision_context.get("underexplored_dimensions"):
        reasons.append("coverage_low")
    return list(dict.fromkeys(reasons))


def _normalize_action_request(
    *,
    controller_plan: dict[str, Any],
    action_package: dict[str, Any],
    lab_mode: bool,
) -> dict[str, Any]:
    raw = str(
        action_package.get("requested_execution_action")
        or controller_plan.get("requested_execution_action")
        or controller_plan.get("intervention_type")
        or "direct_bo_pick"
    ).strip() or "direct_bo_pick"
    if not lab_mode:
        action_package["raw_requested_execution_action"] = raw
        action_package["normalized_requested_execution_action"] = raw
        action_package["action_normalization_reason"] = "benchmark_preserve_request"
        action_package["requested_execution_action"] = raw
        return action_package

    intervention = str(controller_plan.get("intervention_type") or "").strip().lower()
    selection = str(action_package.get("selection_policy") or "").strip().lower()
    shortlist = str(action_package.get("shortlist_policy") or "").strip().lower()
    focus = str(action_package.get("focus_policy") or "").strip().lower()
    raw_lower = raw.lower()
    normalized = _canonical_lab_action(raw_lower)
    reason = "raw_requested_action"

    rerank_signal = any(
        token in " ".join([intervention, selection, shortlist, raw_lower])
        for token in ("rerank", "alternative", "alt_pick")
    )
    shape_signal = any(
        token in " ".join([intervention, selection, shortlist, focus, raw_lower])
        for token in ("shape", "shaped", "coverage", "scaffold", "corridor", "focus")
    )
    if rerank_signal:
        normalized = "shortlist_alt_pick"
        reason = "controller_plan_indicates_shortlist_rerank"
    elif shape_signal:
        normalized = "shape_only_bo_pick"
        reason = "controller_plan_indicates_coverage_or_focus_shaping"

    action_package["raw_requested_execution_action"] = raw
    action_package["normalized_requested_execution_action"] = normalized
    action_package["action_normalization_reason"] = reason
    action_package["requested_execution_action"] = normalized
    return action_package


def _canonical_lab_action(raw_action: str) -> str:
    action = str(raw_action or "").strip().lower() or "direct_bo_pick"
    if action in {"bo_direct", "bo_top1", "keep", "keep_planner_batch"}:
        return "direct_bo_pick"
    if action in {"bo_rerank_topk", "rerank_shortlist", "focus_then_rerank"}:
        return "shortlist_alt_pick"
    if action in {
        "coverage_shape",
        "bo_top1_from_shaped_shortlist",
        "select_from_shaped_shortlist",
        "focus_shape",
    }:
        return "shape_only_bo_pick"
    return action


def _shape_for_coverage(
    candidate_pool: list[dict[str, Any]],
    decision_context: dict[str, Any],
) -> list[dict[str, Any]]:
    underexplored = [str(item) for item in decision_context.get("underexplored_dimensions", [])]
    recent = [
        row.get("candidate", {})
        for row in decision_context.get("recent_history_tail", [])
        if isinstance(row.get("candidate"), dict)
    ]
    seen_by_dim = {
        dim: {str(candidate.get(dim, "")) for candidate in recent}
        for dim in underexplored
    }

    def _score(item: dict[str, Any]) -> tuple[float, int]:
        candidate = item.get("candidate") or {}
        novelty = 0.0
        for dim in underexplored:
            if str(candidate.get(dim, "")) not in seen_by_dim.get(dim, set()):
                novelty += 1.0
        rank = int(item.get("bo_rank", item.get("candidate_index", 0)) or 0)
        return (-novelty, rank)

    return sorted(candidate_pool, key=_score)


def _with_planner_anchor(
    *,
    anchor_pool: list[dict[str, Any]],
    ordered_pool: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not anchor_pool:
        return ordered_pool
    anchor = anchor_pool[0]
    ordered = [anchor]
    for item in ordered_pool:
        if item is anchor:
            continue
        if item.get("candidate") == anchor.get("candidate"):
            continue
        ordered.append(item)
    return ordered


def _select_diverse_batch(
    candidate_pool: list[dict[str, Any]],
    *,
    requested: int,
    decision_context: dict[str, Any],
    strategy: str,
) -> list[dict[str, Any]]:
    if not candidate_pool:
        return []
    requested = max(1, int(requested))
    dimensions = _batch_diversity_dimensions(candidate_pool, decision_context)
    selected: list[dict[str, Any]] = [
        {**candidate_pool[0], **_batch_role(candidate_pool[0], [], dimensions, first=True)}
    ]
    remaining = [item for item in candidate_pool[1:] if item.get("candidate") != candidate_pool[0].get("candidate")]
    while remaining and len(selected) < requested:
        scored = [
            (_diversity_score(item, selected, dimensions), idx, item)
            for idx, item in enumerate(remaining)
        ]
        _, selected_idx, item = max(scored, key=lambda entry: entry[0])
        selected.append({**item, **_batch_role(item, selected, dimensions, first=False)})
        remaining.pop(selected_idx)
    if len(selected) < requested:
        selected_signatures = {
            _candidate_signature(chosen.get("candidate") or {})
            for chosen in selected
        }
        for item in candidate_pool:
            if _candidate_signature(item.get("candidate") or {}) in selected_signatures:
                continue
            selected.append({**item, **_batch_role(item, selected, dimensions, first=False)})
            selected_signatures.add(_candidate_signature(item.get("candidate") or {}))
            if len(selected) >= requested:
                break
    for item in selected:
        item.setdefault("batch_selection_strategy", strategy)
    return selected[:requested]


def _batch_diversity_dimensions(
    candidate_pool: list[dict[str, Any]],
    decision_context: dict[str, Any],
) -> list[str]:
    underexplored = [
        str(item)
        for item in decision_context.get("underexplored_dimensions", [])
        if str(item).strip()
    ]
    first_candidate = next(
        (item.get("candidate") for item in candidate_pool if isinstance(item.get("candidate"), dict)),
        {},
    )
    all_dims = list(first_candidate.keys())
    ordered = underexplored + [dim for dim in all_dims if dim not in underexplored]
    return ordered


def _diversity_score(
    item: dict[str, Any],
    selected: list[dict[str, Any]],
    dimensions: list[str],
) -> tuple[int, int, int, int]:
    candidate = item.get("candidate") or {}
    selected_candidates = [entry.get("candidate") or {} for entry in selected]
    novelty = 0
    for dim in dimensions:
        selected_values = {str(entry.get(dim, "")) for entry in selected_candidates}
        if str(candidate.get(dim, "")) not in selected_values:
            novelty += 1
    pair_dims = [dim for dim in ("Additive", "Solvent", "TEMPO derivative", "Catalyst") if dim in candidate]
    pair = tuple(str(candidate.get(dim, "")) for dim in pair_dims[:2])
    selected_pairs = {
        tuple(str(entry.get(dim, "")) for dim in pair_dims[:2])
        for entry in selected_candidates
    }
    pair_novelty = 1 if pair_dims and pair not in selected_pairs else 0
    rank = int(item.get("bo_rank", item.get("candidate_index", 0)) or 0)
    return (pair_novelty, novelty, -rank, -int(item.get("candidate_index", rank) or rank))


def _batch_role(
    item: dict[str, Any],
    selected: list[dict[str, Any]],
    dimensions: list[str],
    *,
    first: bool,
) -> dict[str, str]:
    candidate = item.get("candidate") or {}
    if first:
        return {
            "batch_role": "planner_anchor",
            "batch_role_reason": "keeps the planner's highest-priority candidate visible in the batch",
        }
    anchor = (selected[0].get("candidate") or {}) if selected else {}
    changed = [dim for dim in dimensions if str(candidate.get(dim, "")) != str(anchor.get(dim, ""))]
    if "Additive" in changed:
        role = "additive_contrast"
    elif "Catalyst" in changed:
        role = "catalyst_contrast"
    elif "Solvent" in changed:
        role = "solvent_contrast"
    elif "TEMPO derivative" in changed:
        role = "tempo_probe"
    elif changed:
        role = f"{changed[0].lower().replace(' ', '_')}_probe"
    else:
        role = "local_refinement"
    reason = (
        "adds batch diversity relative to the planner anchor"
        if changed
        else "keeps a local refinement near the planner anchor"
    )
    if changed:
        reason += f" by changing {', '.join(changed[:3])}"
    return {"batch_role": role, "batch_role_reason": reason}


def _batch_selection_summary(
    selected: list[dict[str, Any]],
    decision_context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "selected_candidate_indices": [item.get("candidate_index") for item in selected],
        "batch_roles": [item.get("batch_role") for item in selected],
        "diversity_dimensions": _batch_diversity_dimensions(selected, decision_context),
    }


def _candidate_signature(candidate: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in candidate.items()))


def _order_by_rerank(
    candidate_pool: list[dict[str, Any]],
    rerank_action: dict[str, Any],
) -> list[dict[str, Any]]:
    by_index = {
        int(item.get("candidate_index", idx)): item
        for idx, item in enumerate(candidate_pool)
    }
    ordered: list[dict[str, Any]] = []
    if candidate_pool:
        ordered.append(candidate_pool[0])
    selected_index = _safe_int(rerank_action.get("selected_index"))
    if selected_index in by_index and by_index[selected_index] not in ordered:
        ordered.append(by_index[selected_index])
    scored = []
    for score in rerank_action.get("candidate_scores", []) or []:
        if not isinstance(score, dict):
            continue
        idx = _safe_int(score.get("candidate_index"))
        if idx not in by_index:
            continue
        scored.append((float(score.get("overall_score", 0.0) or 0.0), idx))
    for _, idx in sorted(scored, reverse=True):
        item = by_index[idx]
        if item not in ordered:
            ordered.append(item)
    for item in candidate_pool:
        if item not in ordered:
            ordered.append(item)
    return ordered


def _best_observation(
    history: list[dict[str, Any]],
    *,
    goal: str,
    objective_name: str,
) -> dict[str, Any] | None:
    rows = [row for row in history if _safe_float(row.get("result")) is not None]
    if not rows:
        return None
    reverse = str(goal).lower() != "minimize"
    best = sorted(rows, key=lambda row: float(row["result"]), reverse=reverse)[0]
    return {**dict(best.get("candidate") or {}), objective_name: float(best["result"])}


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
