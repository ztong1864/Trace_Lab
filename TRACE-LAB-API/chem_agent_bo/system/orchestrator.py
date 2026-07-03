"""Main optimization orchestrator for Agentic BO."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field as dataclass_field
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from olympus.campaigns import Campaign
from olympus.objects import ParameterVector
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from chem_agent_bo.bo.base import BasePlanner
from chem_agent_bo.bo.discrete_bo import DiscreteBOPlanner
from chem_agent_bo.env.reaction_env import ReactionEnv
from chem_agent_bo.experiment_core import initial_candidate_keys
from chem_agent_bo.experiment_core import runtime_metadata
from chem_agent_bo.memory.decision_memory import DecisionMemory
from chem_agent_bo.memory.long_term_store import LongTermMemoryStore
from chem_agent_bo.memory.working_memory import WorkingMemory
from chem_agent_bo.state import OnlineDecisionState
from chem_agent_bo.system.override_guardrail import apply_override_guardrail
from chem_agent_bo.system.context_snapshot import build_decision_context_snapshot
from chem_agent_bo.agent.action_package_policy import (
    RARE_SHORTLIST_ACTIONS,
    default_planner_action_policies,
    planner_policy_allows_non_top_final_replacement,
    planner_policy_allows_runtime_promotion,
    planner_policy_shortlist_probe_enabled,
    resolve_planner_action_policy,
)
from chem_agent_bo.system.actionspace_runtime import (
    build_candidate_contrastive_evidence,
    build_dataset_contrast_adapter,
    dominant_mask_keys,
    low_repeat_mask_keys,
    scaffold_corridor_mask_keys,
    shortlist_value_audit,
)
from chem_agent_bo.system.candidate_probe import (
    build_axis_companion_candidates,
    build_local_calibration_candidates,
    build_probe_candidates,
    build_suzuki_local_calibration_candidates,
    merge_probe_candidates_into_shortlist,
)
from chem_agent_bo.knowledge.experience import (
    build_experience_candidate,
    should_promote_experience,
)
from chem_agent_bo.knowledge.episodic_review import (
    EpisodicReviewQueue,
    build_episodic_review_candidate,
)
from chem_agent_bo.knowledge.provider import KnowledgeProvider
from chem_agent_bo.knowledge.query_builder import KnowledgeQueryBuilder
from chem_agent_bo.knowledge.reviewed_store import ReviewedKnowledgeStore
from chem_agent_bo.knowledge.reviewed_experience import ReviewedExperienceStore
from chem_agent_bo.knowledge.node_query_builder import ReviewedKnowledgeQueryBuilder
from chem_agent_bo.utils.subspace import (
    build_subspace_campaign,
    complete_candidate,
    extract_default_values,
)
from chem_agent_bo.protocol import PROTOCOL_VERSION, protocol_budget_metadata
from chem_agent_bo.runtime import BenchmarkExecutionAdapter
from chem_agent_bo.utils.run_io import capture_third_party_output

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from chem_agent_bo.agent.decision_engine import DecisionEngine


@dataclass
class OrchestratorConfig:
    budget: int = 20
    total_budget: int | None = None
    init_budget: int | None = None
    max_feasibility_retries: int = 1
    enable_auto_subspace_trigger: bool = True
    subspace_window_rounds: int = 3
    stagnation_threshold: float = 0.3
    duplicate_threshold: float = 0.35
    coverage_threshold: float = 0.4
    enable_multigranularity_lock_signals: bool = False
    controller_allow_rerank_without_space_narrowing: bool = False
    finite_pool_sparse_coverage_not_veto_rerank: bool = False
    key_dim_coverage_mode: str = "mean"
    controller_reflection_input_mode: str = "full"
    shortlist_size: int = 5
    shortlist_candidate_pool_size: int = 12
    shortlist_main_pool_size: int = 8
    shortlist_diversity_pool_size: int = 6
    shortlist_top_rank_keep: int = 2
    shortlist_min_scaffold_diversity: int = 2
    shortlist_target_scaffold_diversity: int = 3
    shortlist_early_diversity_scaffold_threshold: float = 0.67
    shortlist_early_diversity_no_improvement_rounds: int = 2
    enable_override_guardrail: bool = True
    override_score_margin: float = 0.05
    override_min_transfer_score: float = 0.65
    override_require_structural_shift: bool = True
    planner_trust_policy: str = "auto"
    planner_action_policies: dict[str, dict[str, Any]] = dataclass_field(
        default_factory=default_planner_action_policies
    )
    trusted_planner_names: tuple[str, ...] = ("discrete", "botorch_qei", "botorch_qlogei")
    trusted_planner_min_no_improvement_rounds: int = 6
    trusted_planner_disable_early_diversity: bool = True
    trusted_planner_override_score_margin: float = 0.10
    trusted_planner_min_transfer_score: float = 0.75
    trusted_planner_min_hypothesis_score: float = 0.80
    enable_trusted_planner_pre_rerank_skip: bool = True
    enable_late_stage_incumbent_protection: bool = True
    late_stage_start_iteration: int = 10
    late_stage_start_fraction: float = 0.70
    late_stage_strong_incumbent_percentile: float = 0.90
    late_stage_override_score_margin: float = 0.12
    late_stage_min_transfer_score: float = 0.80
    late_stage_min_hypothesis_score: float = 0.85
    allow_focus_then_rerank: bool = True
    candidate_direction_review_mode: str = "off"
    candidate_direction_review_planners: tuple[str, ...] = ("atlas",)
    candidate_direction_review_actions: tuple[str, ...] = (
        "direct_bo_pick",
        "shape_only_bo_pick",
    )
    candidate_direction_review_min_no_improvement_rounds: int = 0
    candidate_direction_review_max_candidates: int = 5
    candidate_direction_review_near_best_margin: float = 0.05
    candidate_direction_review_require_finite_pool: bool = True
    candidate_probe_enabled: bool = False
    candidate_probe_planners: tuple[str, ...] = ("atlas", "botorch_qei")
    candidate_probe_actions: tuple[str, ...] = ("finite_pool_candidate_probe",)
    candidate_probe_max_candidates: int = 4
    candidate_probe_top_history: int = 6
    candidate_probe_merge_max_size: int = 8
    candidate_probe_min_no_improvement_rounds: int = 3
    candidate_probe_min_iteration: int = 0
    candidate_probe_require_finite_pool: bool = True
    candidate_probe_selection_mode: str = "merge_only"
    candidate_probe_candidate_tool: str = "probe"
    candidate_probe_local_min_anchor_matches: int = 3
    candidate_probe_max_trigger_count: int = 0
    candidate_probe_repeat_min_iteration: int = 0
    candidate_probe_repeat_min_no_improvement_rounds: int = -1
    candidate_probe_cooldown_rounds: int = 0
    candidate_probe_repeat_selection_mode: str = ""
    candidate_probe_late_stage_min_gp_mean_margin: float = 1.0
    candidate_probe_recurrence_bonus: float = 0.0
    candidate_probe_additive_novelty_bonus: float = 0.0
    candidate_probe_additive_axis_prefer_patterns: tuple[str, ...] = ()
    candidate_probe_additive_axis_bonus_patterns: tuple[str, ...] = ()
    candidate_probe_additive_axis_avoid_patterns: tuple[str, ...] = ()
    candidate_probe_additive_axis_min_score: float = 80.0
    candidate_probe_base_axis_prefer_patterns: tuple[str, ...] = ()
    candidate_probe_reactant_axis_prefer_patterns: tuple[str, ...] = ()
    candidate_probe_reactant_axis_min_iteration: int = 12
    candidate_probe_suzuki_anchor_threshold: float = 0.0
    candidate_probe_suzuki_max_iteration: int = 0
    candidate_probe_suzuki_prefer_ligands: tuple[str, ...] = ()
    candidate_probe_suzuki_prefer_bases: tuple[str, ...] = ()
    candidate_probe_suzuki_prefer_solvents: tuple[str, ...] = ()
    candidate_probe_suzuki_warm_start_enabled: bool = False
    candidate_probe_suzuki_warm_start_anchor_threshold: float = 70.0
    candidate_probe_suzuki_warm_start_max_iteration: int = 6
    candidate_probe_suzuki_warm_start_max_trigger_count: int = 1
    candidate_probe_suzuki_warm_start_selection_mode: str = "solvent_base_first"
    shape_shortlist_selection_mode: str = "bo_top1"
    shape_shortlist_selection_top_k: int = 3
    shape_shortlist_selection_max_iteration: int = 0
    shape_only_min_iteration: int = 0
    knowledge_top_k: int = 5
    enable_experience_promotion: bool = True
    protocol_mode: str = "benchmark_clean"
    enable_action_package_v2: bool = False
    enable_action_package_v06: bool = False
    enable_reviewed_knowledge: bool = False
    reviewed_knowledge_top_k: int = 3
    init_reviewed_knowledge_mode: str = "off"
    diagnosis_reviewed_knowledge_mode: str = "off"
    semantic_reviewed_knowledge_mode: str = "off"
    verification_mode: str = "off"
    reviewed_knowledge_advisory_content_statuses: tuple[str, ...] = (
        "seed_placeholder",
        "pinned_curated",
        "reviewed_curated",
        "revised_curated",
    )
    reviewed_knowledge_decision_content_statuses: tuple[str, ...] = (
        "pinned_curated",
        "reviewed_curated",
        "revised_curated",
    )
    reviewed_knowledge_target_nodes: tuple[str, ...] = (
        "design_init_experiments",
        "stagnation_diagnosis",
        "hypothesis_action",
        "semantic_assessment",
        "reflection_action",
        "verification_pass",
    )
    enable_reviewed_experience: bool = False
    reviewed_experience_dir: str = "knowledge/reviewed_experience"
    reviewed_experience_top_k: int = 2
    enable_episodic_review_queue: bool = False
    episodic_review_queue_path: str | None = None
    enable_knowledge_hit_ledger: bool = True
    knowledge_hit_ledger_path: str | None = None
    enable_audit_artifacts: bool = True
    audit_artifact_dir: str | None = None
    filter_mode: str = "exact_match"
    method_name: str = "agent_bo"
    method_family: str = "agentic"
    planner_name: str = "atlas"
    uses_extra_training: bool = False
    uses_pseudodata: bool = False
    uses_external_knowledge: bool = False
    seed: int | None = None
    # Phase A: LLM Init Design
    init_strategy: str = "random"  # "random" | "llm"
    # Phase B: LLM Search Constraint
    enable_llm_search_constraint: bool = False
    constraint_update_freq: int = 5
    min_constraint_pool_fraction: float = 0.05
    constraint_max_duration_rounds: int = 8
    terminal_verbosity: str = "progress"  # quiet | progress | debug
    suppress_third_party_output: bool = True
    third_party_log_path: str | None = None
    progress_jsonl_path: str | None = None
    status_json_path: str | None = None


class Orchestrator:
    """Flow controller: decision -> search -> evaluation -> reflection."""

    def __init__(
        self,
        env: ReactionEnv,
        bo_tool: BasePlanner,
        decision_engine: DecisionEngine,
        memory: DecisionMemory,
        working_memory: WorkingMemory,
        config: OrchestratorConfig | None = None,
        knowledge_provider: KnowledgeProvider | None = None,
        knowledge_query_builder: KnowledgeQueryBuilder | None = None,
        reviewed_knowledge_store: ReviewedKnowledgeStore | None = None,
        reviewed_experience_store: ReviewedExperienceStore | None = None,
        reviewed_knowledge_query_builder: ReviewedKnowledgeQueryBuilder | None = None,
        episodic_review_queue: EpisodicReviewQueue | None = None,
        long_term_store: LongTermMemoryStore | None = None,
        initial_candidates: list[dict[str, Any]] | None = None,
    ) -> None:
        self.env = env
        self.bo_tool = bo_tool
        self.decision_engine = decision_engine
        self.memory = memory
        self.working_memory = working_memory
        self.config = config or OrchestratorConfig()
        self.online_decision_state = OnlineDecisionState(
            prompt_config=getattr(self.decision_engine, "prompt_config", None),
            controller_reflection_input_mode=self.config.controller_reflection_input_mode,
        )
        self.total_budget = int(
            self.config.total_budget if self.config.total_budget is not None else self.config.budget
        )
        self.init_budget = int(
            self.config.init_budget
            if self.config.init_budget is not None
            else min(getattr(self.bo_tool, "_num_init_design", 0), self.total_budget)
        )
        self.knowledge_provider = knowledge_provider
        self.knowledge_query_builder = knowledge_query_builder
        self.reviewed_knowledge_store = reviewed_knowledge_store
        self.reviewed_experience_store = reviewed_experience_store
        self.reviewed_knowledge_query_builder = reviewed_knowledge_query_builder
        self.episodic_review_queue = episodic_review_queue
        self.long_term_store = long_term_store

        self.campaign = Campaign()
        self.campaign.set_param_space(self.env.param_space)
        self.campaign.set_value_space(self.env.value_space)
        self.execution_adapter = BenchmarkExecutionAdapter(env=self.env, campaign=self.campaign)
        self.defaults = extract_default_values(self.env.param_space)
        self.dataset_meta = self.env.dataset_meta()
        self.reaction_context = {
            "dataset": self.env.dataset,
            "reaction_type": self.dataset_meta.get("reaction_type", ""),
            "objective": self.env.objective_name,
            "goal": self.env.goal,
            "backend": self.env.backend,
        }
        self.initial_candidates = list(initial_candidates or [])
        self._initial_state_history: list[dict[str, Any]] = []
        # Phase B: LLM Search Constraint state
        self._active_llm_constraints: list[dict[str, Any]] = []
        self._llm_constraint_expires_at: int = 0
        self._llm_constraint_last_updated_at: int = -999
        self._llm_constraint_summary: str = ""
        self._candidate_probe_seen_counts: dict[tuple[str, ...], int] = {}
        self._candidate_probe_trigger_count = 0
        self._candidate_probe_suzuki_warm_start_trigger_count = 0
        self._candidate_probe_last_trigger_iteration: int | None = None
        self.audit_artifact_root: Path | None = None
        if self.config.enable_audit_artifacts and self.config.audit_artifact_dir:
            self.audit_artifact_root = Path(self.config.audit_artifact_dir)
            self.audit_artifact_root.mkdir(parents=True, exist_ok=True)
        self.knowledge_hit_ledger_path = (
            Path(self.config.knowledge_hit_ledger_path)
            if self.config.enable_knowledge_hit_ledger and self.config.knowledge_hit_ledger_path
            else None
        )
        self.progress_jsonl_path = (
            Path(self.config.progress_jsonl_path)
            if self.config.progress_jsonl_path
            else None
        )
        self.status_json_path = (
            Path(self.config.status_json_path)
            if self.config.status_json_path
            else None
        )
        self.third_party_log_path = (
            Path(self.config.third_party_log_path)
            if self.config.third_party_log_path
            else None
        )
        for path in (
            self.progress_jsonl_path,
            self.status_json_path,
            self.third_party_log_path,
            self.knowledge_hit_ledger_path,
        ):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
        if self.progress_jsonl_path is not None:
            self.progress_jsonl_path.write_text("", encoding="utf-8")

    def _protocol_metadata(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_mode": self.config.protocol_mode,
            "method_name": self.config.method_name,
            "method_family": self.config.method_family,
            "planner_name": self.config.planner_name,
            "seed": self.config.seed,
            "override_policy_version": "v1.5",
            "planner_trust_policy": self.config.planner_trust_policy,
            "trusted_planner_names": list(self.config.trusted_planner_names),
            "trusted_planner_min_no_improvement_rounds": (
                self.config.trusted_planner_min_no_improvement_rounds
            ),
            "trusted_planner_disable_early_diversity": (
                self.config.trusted_planner_disable_early_diversity
            ),
            **protocol_budget_metadata(self.total_budget, self.init_budget),
            "candidate_pool_name": self.env.dataset,
            "candidate_pool_size": self.dataset_meta.get("candidate_count"),
            "uses_extra_training": self.config.uses_extra_training,
            "uses_pseudodata": self.config.uses_pseudodata,
            "uses_external_knowledge": self.config.uses_external_knowledge,
            "init_strategy": self.config.init_strategy,
            "enable_llm_search_constraint": self.config.enable_llm_search_constraint,
        }

    def _planner_action_policy(self, planner_name: str | None = None) -> dict[str, Any]:
        return resolve_planner_action_policy(
            planner_name=str(planner_name or self.config.planner_name or "default"),
            planner_action_policies=self.config.planner_action_policies,
        )

    def _attach_planner_policy_context(
        self,
        decision_context: dict[str, Any],
    ) -> dict[str, Any]:
        context = dict(decision_context)
        context["planner_name"] = str(self.config.planner_name or "")
        context["planner_action_policies"] = deepcopy(self.config.planner_action_policies)
        context["planner_action_policy"] = self._planner_action_policy()
        return context

    @staticmethod
    def _numeric_result(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _support_gap_from_evidence(evidence: dict[str, Any]) -> float | None:
        deltas: list[float] = []
        for block_name in ("same_scaffold_support", "same_anchor_support", "analogue_support"):
            value = (evidence.get(block_name) or {}).get("candidate_minus_bo_best")
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

    def _evaluate_selection_authority(
        self,
        *,
        execution_action: str,
        state_router_guidance: dict[str, Any] | None,
        candidate_contrastive_evidence: dict[str, Any] | None,
        decision_context: dict[str, Any],
    ) -> dict[str, Any]:
        planner_policy = self._planner_action_policy()
        default_level = str(
            planner_policy.get("default_selection_authority_level", "planner_only")
            or "planner_only"
        )
        authority = {
            "planner_action_policy_name": planner_policy.get("planner_policy_name"),
            "selection_authority_level": default_level,
            "authority_source": "planner_policy_default",
            "evidence_sufficiency_passed": False,
            "evidence_failure_reasons": [],
            "shortlist_probe_authorized": False,
        }
        requested = str(execution_action or "direct_bo_pick")
        if requested not in RARE_SHORTLIST_ACTIONS:
            authority["evidence_failure_reasons"] = ["mainline_action_uses_planner_authority"]
            return authority
        if not planner_policy_shortlist_probe_enabled(planner_policy):
            authority["evidence_failure_reasons"] = ["planner_policy_disables_shortlist_probe"]
            return authority
        if not planner_policy_allows_non_top_final_replacement(planner_policy):
            authority["evidence_failure_reasons"] = [
                "planner_policy_disables_non_top_final_replacement"
            ]
            return authority
        guidance = dict(state_router_guidance or {})
        fallback_index = int(guidance.get("fallback_candidate_index", 0) or 0)
        admissible_indices = {
            int(value)
            for value in list(guidance.get("admissible_candidate_indices", []) or [])
            if value is not None
        }
        preferred_indices = {
            int(value)
            for value in list(guidance.get("preferred_candidate_indices", []) or [])
            if value is not None
        }
        candidate_indices = preferred_indices or admissible_indices
        candidate_indices.discard(fallback_index)
        evidence_by_index = dict((candidate_contrastive_evidence or {}).get("by_candidate_index") or {})
        if not candidate_indices:
            authority["evidence_failure_reasons"] = ["no_nonfallback_candidates_authorized"]
            return authority
        discriminative_candidates = 0
        positive_support_candidates = 0
        for candidate_index in candidate_indices:
            evidence = dict(evidence_by_index.get(str(candidate_index)) or {})
            same_scaffold_match_count = int(
                ((evidence.get("same_scaffold_support") or {}).get("candidate") or {}).get(
                    "match_count",
                    0,
                )
                or 0
            )
            same_anchor_match_count = int(
                ((evidence.get("same_anchor_support") or {}).get("candidate") or {}).get(
                    "match_count",
                    0,
                )
                or 0
            )
            support_gap = self._support_gap_from_evidence(evidence)
            has_local_support = (
                same_scaffold_match_count > 0
                or same_anchor_match_count > 0
                or (support_gap is not None and support_gap > 0.0)
            )
            if has_local_support:
                discriminative_candidates += 1
            if support_gap is not None and support_gap > 0.0:
                positive_support_candidates += 1
        if discriminative_candidates == 0:
            authority["evidence_failure_reasons"].append("no_discriminative_local_support")
        if positive_support_candidates == 0:
            authority["evidence_failure_reasons"].append("support_gap_not_positive")
        if bool(decision_context.get("verification_warns_extension", False)) and discriminative_candidates == 0:
            authority["evidence_failure_reasons"].append(
                "verification_warns_extension_without_local_support"
            )
        passed = len(authority["evidence_failure_reasons"]) == 0
        authority["evidence_sufficiency_passed"] = passed
        if passed:
            authority["selection_authority_level"] = "bounded_probe"
            authority["authority_source"] = "evidence_gate_promoted"
            authority["shortlist_probe_authorized"] = True
        return authority

    def _candidate_direction_review_state(
        self,
        *,
        execution_action: str,
        decision_context: dict[str, Any],
        shortlist_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        mode = str(self.config.candidate_direction_review_mode or "off").strip().lower()
        state: dict[str, Any] = {
            "enabled": mode != "off",
            "mode": mode,
            "eligible": False,
            "applied": False,
            "reason": "",
        }
        if mode == "off":
            state["reason"] = "disabled"
            return state
        if mode not in {"bounded_pick", "advisory"}:
            state["reason"] = f"unsupported_mode:{mode}"
            return state
        if self.config.candidate_direction_review_require_finite_pool and not self.env.is_finite_pool:
            state["reason"] = "requires_finite_pool"
            return state
        planner_allow = {str(item).strip().lower() for item in self.config.candidate_direction_review_planners}
        planner_name = str(self.config.planner_name or "").strip().lower()
        if "*" not in planner_allow and planner_name not in planner_allow:
            state["reason"] = "planner_not_allowed"
            return state
        action_allow = {str(item).strip() for item in self.config.candidate_direction_review_actions}
        if "*" not in action_allow and str(execution_action or "") not in action_allow:
            state["reason"] = "action_not_allowed"
            return state
        no_improvement_rounds = int(decision_context.get("no_improvement_rounds", 0) or 0)
        min_rounds = int(self.config.candidate_direction_review_min_no_improvement_rounds or 0)
        if no_improvement_rounds < min_rounds:
            state["reason"] = "below_no_improvement_threshold"
            state["no_improvement_rounds"] = no_improvement_rounds
            state["min_no_improvement_rounds"] = min_rounds
            return state
        if not shortlist_candidates:
            state["reason"] = "empty_shortlist"
            return state
        max_candidates = max(2, int(self.config.candidate_direction_review_max_candidates or 5))
        state.update(
            {
                "eligible": True,
                "reason": "eligible",
                "max_candidates": max_candidates,
                "candidate_indices": [
                    int(item.get("candidate_index", idx) or idx)
                    for idx, item in enumerate(shortlist_candidates[:max_candidates])
                ],
            }
        )
        return state

    def _candidate_probe_state(
        self,
        *,
        execution_action: str,
        decision_context: dict[str, Any],
        iteration: int,
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "enabled": bool(self.config.candidate_probe_enabled),
            "eligible": False,
            "applied": False,
            "reason": "",
        }
        if not self.config.candidate_probe_enabled:
            state["reason"] = "disabled"
            return state
        if self.config.candidate_probe_require_finite_pool and not self.env.is_finite_pool:
            state["reason"] = "requires_finite_pool"
            return state
        planner_allow = {str(item).strip().lower() for item in self.config.candidate_probe_planners}
        planner_name = str(self.config.planner_name or "").strip().lower()
        if "*" not in planner_allow and planner_name not in planner_allow:
            state["reason"] = "planner_not_allowed"
            return state
        action_allow = {str(item).strip() for item in self.config.candidate_probe_actions}
        if "*" not in action_allow and str(execution_action or "") not in action_allow:
            state["reason"] = "action_not_allowed"
            return state
        candidate_tool = str(self.config.candidate_probe_candidate_tool or "probe").strip().lower()
        if candidate_tool in {
            "suzuki_local_calibration",
            "suzuki_early_local_calibration",
            "suzuki_v05_local_calibration",
        }:
            max_iteration = int(self.config.candidate_probe_suzuki_max_iteration or 0)
            if max_iteration > 0 and int(iteration) > max_iteration:
                state["reason"] = "above_suzuki_local_calibration_iteration"
                state["iteration"] = int(iteration)
                state["max_iteration"] = max_iteration
                return state
        trigger_count = int(self._candidate_probe_trigger_count)
        no_improvement_rounds = int(decision_context.get("no_improvement_rounds", 0) or 0)
        min_rounds = int(self.config.candidate_probe_min_no_improvement_rounds or 0)
        if trigger_count > 0 and int(self.config.candidate_probe_repeat_min_no_improvement_rounds) >= 0:
            min_rounds = int(self.config.candidate_probe_repeat_min_no_improvement_rounds)
        if no_improvement_rounds < min_rounds:
            state["reason"] = "below_no_improvement_threshold"
            state["no_improvement_rounds"] = no_improvement_rounds
            state["min_no_improvement_rounds"] = min_rounds
            state["trigger_count"] = trigger_count
            return state
        min_iteration = int(self.config.candidate_probe_min_iteration or 0)
        repeat_min_iteration = int(self.config.candidate_probe_repeat_min_iteration or 0)
        if trigger_count > 0 and repeat_min_iteration > 0:
            min_iteration = repeat_min_iteration
        if min_iteration > 0 and int(iteration) < min_iteration:
            state["reason"] = "below_min_iteration"
            state["iteration"] = int(iteration)
            state["min_iteration"] = min_iteration
            state["trigger_count"] = trigger_count
            return state
        cooldown_rounds = int(self.config.candidate_probe_cooldown_rounds or 0)
        if (
            cooldown_rounds > 0
            and self._candidate_probe_last_trigger_iteration is not None
            and int(iteration) - int(self._candidate_probe_last_trigger_iteration) <= cooldown_rounds
        ):
            state["reason"] = "cooldown_active"
            state["iteration"] = int(iteration)
            state["last_trigger_iteration"] = int(self._candidate_probe_last_trigger_iteration)
            state["cooldown_rounds"] = cooldown_rounds
            state["trigger_count"] = trigger_count
            return state
        max_trigger_count = int(self.config.candidate_probe_max_trigger_count or 0)
        if max_trigger_count > 0 and self._candidate_probe_trigger_count >= max_trigger_count:
            state["reason"] = "max_trigger_count_reached"
            state["trigger_count"] = int(self._candidate_probe_trigger_count)
            state["max_trigger_count"] = max_trigger_count
            return state
        state.update(
            {
                "eligible": True,
                "reason": "eligible",
                "max_candidates": int(self.config.candidate_probe_max_candidates or 4),
                "top_history": int(self.config.candidate_probe_top_history or 6),
                "merge_max_size": int(self.config.candidate_probe_merge_max_size or 8),
                "candidate_tool": candidate_tool,
                "local_min_anchor_matches": int(
                    self.config.candidate_probe_local_min_anchor_matches or 3
                ),
                "iteration": int(iteration),
                "trigger_count": int(self._candidate_probe_trigger_count),
                "max_trigger_count": max_trigger_count,
                "min_iteration": min_iteration,
                "min_no_improvement_rounds": min_rounds,
                "cooldown_rounds": cooldown_rounds,
                "last_trigger_iteration": self._candidate_probe_last_trigger_iteration,
                "direction": dict((decision_context.get("candidate_probe_direction") or {}) or {}),
            }
        )
        return state

    def _apply_candidate_probe(
        self,
        *,
        history: list[dict[str, Any]],
        shortlist_candidates: list[dict[str, Any]],
        allowed_keys: set[tuple[str, ...]] | None,
        candidate_probe: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        trace = {
            **dict(candidate_probe or {}),
            "applied": False,
            "candidate_ids": [],
            "candidate_indices": [],
        }
        if not bool(trace.get("eligible", False)):
            return shortlist_candidates, trace
        finite_pool_table = getattr(self.env, "_finite_pool_table", None)
        if finite_pool_table is None or not hasattr(finite_pool_table, "candidate_records"):
            trace["reason"] = "finite_pool_table_unavailable"
            return shortlist_candidates, trace
        candidate_tool = str(trace.get("candidate_tool") or "probe").strip().lower()
        if candidate_tool in {
            "staged_base_reactant_axis_companion",
            "base_then_reactant_axis_companion",
            "staged_axis_companion",
        }:
            switch_iteration = int(self.config.candidate_probe_reactant_axis_min_iteration or 12)
            candidate_tool = (
                "reactant_axis_companion"
                if int(trace.get("iteration", 0) or 0) >= switch_iteration
                else "base_axis_companion"
            )
            trace["staged_candidate_tool"] = candidate_tool
            trace["reactant_axis_min_iteration"] = switch_iteration
        if candidate_tool in {
            "suzuki_local_calibration",
            "suzuki_early_local_calibration",
            "suzuki_v05_local_calibration",
        }:
            probe_candidates, probe_meta = build_suzuki_local_calibration_candidates(
                candidate_records=finite_pool_table.candidate_records(),
                history=history,
                shortlist_candidates=shortlist_candidates,
                feature_columns=self._feature_columns(),
                allowed_keys=allowed_keys,
                goal=self.env.goal,
                max_candidates=int(trace.get("max_candidates") or 4),
                top_history=int(trace.get("top_history") or 6),
                anchor_threshold=self._candidate_probe_suzuki_effective_anchor_threshold(
                    history=history,
                    trace=trace,
                ),
                prefer_ligands=self.config.candidate_probe_suzuki_prefer_ligands,
                prefer_bases=self.config.candidate_probe_suzuki_prefer_bases,
                prefer_solvents=self.config.candidate_probe_suzuki_prefer_solvents,
            )
            if bool(trace.get("suzuki_warm_start_active", False)):
                probe_meta = {
                    **dict(probe_meta or {}),
                    "suzuki_warm_start_active": True,
                    "suzuki_warm_start_selection_mode": str(
                        self.config.candidate_probe_suzuki_warm_start_selection_mode
                    ),
                    "suzuki_warm_start_trigger_count": int(
                        self._candidate_probe_suzuki_warm_start_trigger_count
                    ),
                }
        elif candidate_tool in {"local_calibration", "local_calibration_probe", "local"}:
            probe_candidates, probe_meta = build_local_calibration_candidates(
                candidate_records=finite_pool_table.candidate_records(),
                history=history,
                shortlist_candidates=shortlist_candidates,
                feature_columns=self._feature_columns(),
                allowed_keys=allowed_keys,
                goal=self.env.goal,
                max_candidates=int(trace.get("max_candidates") or 4),
                top_history=int(trace.get("top_history") or 6),
                min_anchor_matches=int(trace.get("local_min_anchor_matches") or 3),
            )
        elif candidate_tool in {"base_axis_companion", "base_companion", "axis_companion_base"}:
            probe_candidates, probe_meta = build_axis_companion_candidates(
                candidate_records=finite_pool_table.candidate_records(),
                history=history,
                shortlist_candidates=shortlist_candidates,
                feature_columns=self._feature_columns(),
                allowed_keys=allowed_keys,
                goal=self.env.goal,
                max_candidates=int(trace.get("max_candidates") or 4),
                top_history=int(trace.get("top_history") or 6),
                axis_name="Base",
                prefer_patterns=self.config.candidate_probe_base_axis_prefer_patterns,
            )
        elif candidate_tool in {
            "reactant_axis_companion",
            "reactant_companion",
            "axis_companion_reactant",
            "reactant2_axis_companion",
        }:
            reactant_axis = "Reactant2" if "Reactant2" in self._feature_columns() else self._feature_columns()[0]
            probe_candidates, probe_meta = build_axis_companion_candidates(
                candidate_records=finite_pool_table.candidate_records(),
                history=history,
                shortlist_candidates=shortlist_candidates,
                feature_columns=self._feature_columns(),
                allowed_keys=allowed_keys,
                goal=self.env.goal,
                max_candidates=int(trace.get("max_candidates") or 4),
                top_history=int(trace.get("top_history") or 6),
                axis_name=reactant_axis,
                prefer_patterns=self.config.candidate_probe_reactant_axis_prefer_patterns,
            )
        elif candidate_tool in {"combined", "local_plus_probe", "probe_plus_local"}:
            local_candidates, local_meta = build_local_calibration_candidates(
                candidate_records=finite_pool_table.candidate_records(),
                history=history,
                shortlist_candidates=shortlist_candidates,
                feature_columns=self._feature_columns(),
                allowed_keys=allowed_keys,
                goal=self.env.goal,
                max_candidates=int(trace.get("max_candidates") or 4),
                top_history=int(trace.get("top_history") or 6),
                min_anchor_matches=int(trace.get("local_min_anchor_matches") or 3),
            )
            global_candidates, global_meta = build_probe_candidates(
                candidate_records=finite_pool_table.candidate_records(),
                history=history,
                shortlist_candidates=shortlist_candidates,
                feature_columns=self._feature_columns(),
                scaffold_dims=self._scaffold_dimensions(),
                allowed_keys=allowed_keys,
                include_map=dict((trace.get("direction") or {}).get("include_map") or {}),
                goal=self.env.goal,
                max_candidates=int(trace.get("max_candidates") or 4),
                top_history=int(trace.get("top_history") or 6),
            )
            probe_candidates = []
            seen_keys: set[tuple[str, ...]] = set()
            for item in [*local_candidates, *global_candidates]:
                key = tuple(str(value) for value in item.get("key", ()))
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                probe_candidates.append(item)
                if len(probe_candidates) >= int(trace.get("max_candidates") or 4):
                    break
            probe_meta = {
                "enabled": True,
                "reason": None,
                "tool": "combined",
                "local_meta": local_meta,
                "probe_meta": global_meta,
                "selected_candidate_count": len(probe_candidates),
            }
        else:
            probe_candidates, probe_meta = build_probe_candidates(
                candidate_records=finite_pool_table.candidate_records(),
                history=history,
                shortlist_candidates=shortlist_candidates,
                feature_columns=self._feature_columns(),
                scaffold_dims=self._scaffold_dimensions(),
                allowed_keys=allowed_keys,
                include_map=dict((trace.get("direction") or {}).get("include_map") or {}),
                goal=self.env.goal,
                max_candidates=int(trace.get("max_candidates") or 4),
                top_history=int(trace.get("top_history") or 6),
            )
        merged, merge_meta = merge_probe_candidates_into_shortlist(
            shortlist_candidates=shortlist_candidates,
            probe_candidates=probe_candidates,
            history=history,
            feature_columns=self._feature_columns(),
            scaffold_dims=self._scaffold_dimensions(),
            max_size=int(trace.get("merge_max_size") or len(shortlist_candidates) + len(probe_candidates)),
        )
        added = [
            item
            for item in merged
            if str(item.get("shortlist_source", "")) == "candidate_probe_injected"
        ]
        trace.update(
            {
                **probe_meta,
                **merge_meta,
                "applied": bool(added),
                "reason": None if added else probe_meta.get("reason") or "no_probe_candidate_added",
                "candidate_ids": [str(item.get("candidate_id")) for item in added if item.get("candidate_id")],
                "candidate_indices": [
                    int(item.get("candidate_index", idx) or idx)
                    for idx, item in enumerate(added)
                ],
                "candidates": [
                    {
                        "candidate_index": item.get("candidate_index"),
                        "candidate_id": item.get("candidate_id"),
                        "candidate_probe_rank": item.get("candidate_probe_rank"),
                        "probe_score": item.get("probe_score"),
                        "probe_evidence": item.get("probe_evidence"),
                        "candidate": item.get("candidate"),
                    }
                    for item in added
                ],
            }
        )
        if added:
            self._candidate_probe_trigger_count += 1
            if bool(trace.get("suzuki_warm_start_active", False)):
                self._candidate_probe_suzuki_warm_start_trigger_count += 1
                trace["suzuki_warm_start_trigger_count_after"] = int(
                    self._candidate_probe_suzuki_warm_start_trigger_count
                )
            trace["trigger_count_after"] = int(self._candidate_probe_trigger_count)
            self._candidate_probe_last_trigger_iteration = int(trace.get("iteration", 0) or 0) or int(
                candidate_probe.get("iteration", 0) or 0
            ) or self._candidate_probe_last_trigger_iteration
            trace["last_trigger_iteration"] = self._candidate_probe_last_trigger_iteration
        return merged, trace

    def _candidate_probe_suzuki_effective_anchor_threshold(
        self,
        *,
        history: list[dict[str, Any]],
        trace: dict[str, Any],
    ) -> float:
        strong_threshold = float(self.config.candidate_probe_suzuki_anchor_threshold or 0.0)
        trace["suzuki_warm_start_active"] = False
        if not bool(self.config.candidate_probe_suzuki_warm_start_enabled):
            return strong_threshold
        iteration = int(trace.get("iteration", 0) or 0)
        warm_max_iteration = int(self.config.candidate_probe_suzuki_warm_start_max_iteration or 0)
        warm_max_triggers = int(self.config.candidate_probe_suzuki_warm_start_max_trigger_count or 0)
        if warm_max_iteration > 0 and iteration > warm_max_iteration:
            return strong_threshold
        if (
            warm_max_triggers > 0
            and self._candidate_probe_suzuki_warm_start_trigger_count >= warm_max_triggers
        ):
            return strong_threshold
        best_result = None
        for row in history:
            value = self._numeric_result(row.get("result"))
            if value is None:
                continue
            if best_result is None:
                best_result = value
            elif str(self.env.goal).strip().lower() == "minimize":
                best_result = min(best_result, value)
            else:
                best_result = max(best_result, value)
        warm_threshold = float(self.config.candidate_probe_suzuki_warm_start_anchor_threshold or 0.0)
        if best_result is None:
            return strong_threshold
        is_moderate_anchor = (
            warm_threshold > 0.0
            and float(best_result) >= warm_threshold
            and (strong_threshold <= 0.0 or float(best_result) < strong_threshold)
        )
        if is_moderate_anchor:
            trace["suzuki_warm_start_active"] = True
            trace["suzuki_warm_start_best_result"] = float(best_result)
            trace["suzuki_warm_start_anchor_threshold"] = warm_threshold
            trace["suzuki_strong_anchor_threshold"] = strong_threshold
            return warm_threshold
        return strong_threshold

    def _candidate_probe_preferred_item(
        self,
        *,
        shortlist_candidates: list[dict[str, Any]],
        candidate_probe_trace: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        trigger_count = int(candidate_probe_trace.get("trigger_count", 0) or 0)
        repeat_mode = str(self.config.candidate_probe_repeat_selection_mode or "").strip().lower()
        if trigger_count > 0 and repeat_mode:
            mode = repeat_mode
        else:
            mode = str(self.config.candidate_probe_selection_mode or "merge_only").strip().lower()
        candidate_probe_trace["effective_selection_mode"] = mode
        if mode in {"", "merge_only", "off", "none"}:
            return None
        if not bool(candidate_probe_trace.get("applied", False)):
            return None
        probe_items = [
            item
            for item in shortlist_candidates
            if str(item.get("shortlist_source", "")) == "candidate_probe_injected"
        ]
        if not probe_items:
            return None
        probe_items = sorted(
            probe_items,
            key=lambda item: int(item.get("candidate_probe_rank", item.get("candidate_index", 999999)) or 0),
        )
        if bool(candidate_probe_trace.get("suzuki_warm_start_active", False)):
            warm_item = self._candidate_probe_suzuki_warm_start_preferred_item(
                probe_items=probe_items,
                candidate_probe_trace=candidate_probe_trace,
            )
            if warm_item is not None:
                return warm_item
        feature_columns = self._feature_columns()
        for item in probe_items:
            candidate = item.get("candidate")
            if not isinstance(candidate, dict):
                continue
            key = tuple(str(candidate.get(name)) for name in feature_columns)
            self._candidate_probe_seen_counts[key] = self._candidate_probe_seen_counts.get(key, 0) + 1
        if mode in {"rank1", "rank1_fallback", "probe_rank1"}:
            return probe_items[0]
        if mode in {"top3_discrete_ei", "discrete_ei_top3", "probe_top3_discrete_ei"}:
            top_items = probe_items[:3]
            ranked_item = self._rank_candidate_probe_items_with_discrete_ei(
                probe_items=top_items,
                history=history,
            )
            return ranked_item or top_items[0]
        if mode in {
            "top3_discrete_ei_vs_bo",
            "discrete_ei_top3_vs_bo",
            "probe_top3_discrete_ei_vs_bo",
        }:
            bo_top1_item = self._main_bo_top1_item(shortlist_candidates)
            selector_items = []
            if bo_top1_item is not None:
                selector_items.append(bo_top1_item)
            selector_items.extend(probe_items[:3])
            ranked_item = self._rank_candidate_probe_items_with_discrete_ei(
                probe_items=selector_items,
                history=history,
            )
            if ranked_item is None:
                return None
            if str(ranked_item.get("shortlist_source", "")) != "candidate_probe_injected":
                return None
            return ranked_item
        if mode in {
            "gp_mean_guarded_vs_shortlist",
            "top3_gp_mean_guarded_vs_shortlist",
            "probe_gp_mean_guarded_vs_shortlist",
        }:
            ranked_item = self._rank_candidate_probe_items_with_gp_mean_guarded(
                shortlist_candidates=shortlist_candidates,
                probe_items=probe_items[:3],
                history=history,
            )
            return ranked_item
        if mode in {
            "additive_axis_guarded",
            "probe_additive_axis_guarded",
            "additive_axis_guarded_vs_bo",
        }:
            ranked_item = self._rank_candidate_probe_items_with_additive_axis_guarded(
                shortlist_candidates=shortlist_candidates,
                probe_items=probe_items,
                candidate_probe_trace=candidate_probe_trace,
            )
            return ranked_item
        return None

    def _candidate_probe_suzuki_warm_start_preferred_item(
        self,
        *,
        probe_items: list[dict[str, Any]],
        candidate_probe_trace: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not probe_items:
            return None
        mode = str(
            self.config.candidate_probe_suzuki_warm_start_selection_mode
            or "solvent_base_first"
        ).strip().lower()
        if mode in {"rank1", "probe_rank1"}:
            return probe_items[0]
        axis_rank = {"solvent": 0, "base": 1, "ligand": 2}
        if mode in {"base_first", "base_solvent_first"}:
            axis_rank = {"base": 0, "solvent": 1, "ligand": 2}
        elif mode in {"ligand_first"}:
            axis_rank = {"ligand": 0, "base": 1, "solvent": 2}
        ranked = sorted(
            probe_items,
            key=lambda item: (
                axis_rank.get(
                    str((item.get("probe_evidence") or {}).get("changed_axis", "")),
                    9,
                ),
                -float(item.get("probe_score", 0.0) or 0.0),
                int(item.get("candidate_probe_rank", item.get("candidate_index", 999999)) or 0),
            ),
        )
        chosen = ranked[0] if ranked else None
        if chosen is not None:
            candidate_probe_trace["suzuki_warm_start_selector"] = {
                "mode": mode,
                "selected_changed_axis": str(
                    (chosen.get("probe_evidence") or {}).get("changed_axis", "")
                ),
                "selected_probe_score": chosen.get("probe_score"),
                "axis_order": axis_rank,
            }
        return chosen

    def _candidate_probe_has_core_anchor_support(self, item: dict[str, Any]) -> bool:
        evidence = item.get("probe_evidence")
        if not isinstance(evidence, dict):
            return True
        dim_support = evidence.get("dim_support")
        if not isinstance(dim_support, dict):
            return True
        candidate = item.get("candidate")
        if not isinstance(candidate, dict):
            return False
        core_dims = [name for name in ("Reactant2", "Ligand", "Base") if name in candidate]
        return all(float(dim_support.get(name, 0.0) or 0.0) > 0.0 for name in core_dims)

    def _candidate_probe_additive_axis_score(self, item: dict[str, Any]) -> float:
        candidate = item.get("candidate")
        if not isinstance(candidate, dict):
            return -float("inf")
        additive = str(candidate.get("Additive", ""))
        if not additive:
            return -float("inf")
        score = 0.0
        for pattern in self.config.candidate_probe_additive_axis_prefer_patterns or ():
            pattern_text = str(pattern)
            if pattern_text and pattern_text in additive:
                score += 100.0
        for pattern in self.config.candidate_probe_additive_axis_bonus_patterns or ():
            pattern_text = str(pattern)
            if pattern_text and pattern_text in additive:
                score += 20.0
        for pattern in self.config.candidate_probe_additive_axis_avoid_patterns or ():
            pattern_text = str(pattern)
            if pattern_text and pattern_text in additive:
                score -= 40.0
        score -= max(0.0, float(len(additive) - 36)) * 0.2
        return float(score)

    def _rank_candidate_probe_items_with_additive_axis_guarded(
        self,
        *,
        shortlist_candidates: list[dict[str, Any]],
        probe_items: list[dict[str, Any]],
        candidate_probe_trace: dict[str, Any],
    ) -> dict[str, Any] | None:
        bo_top1_item = self._main_bo_top1_item(shortlist_candidates)
        scored_probe_items: list[tuple[float, dict[str, Any]]] = []
        details: list[dict[str, Any]] = []
        for item in probe_items:
            if not self._candidate_probe_has_core_anchor_support(item):
                continue
            score = self._candidate_probe_additive_axis_score(item)
            scored_probe_items.append((score, item))
            details.append(
                {
                    "candidate_index": item.get("candidate_index"),
                    "candidate_probe_rank": item.get("candidate_probe_rank"),
                    "candidate_id": item.get("candidate_id"),
                    "additive_axis_score": round(float(score), 6),
                    "additive": (item.get("candidate") or {}).get("Additive")
                    if isinstance(item.get("candidate"), dict)
                    else None,
                }
            )
        min_score = float(self.config.candidate_probe_additive_axis_min_score or 0.0)
        strong_items = [
            (score, item)
            for score, item in scored_probe_items
            if float(score) >= min_score
        ]
        candidate_probe_trace["additive_axis_guarded"] = {
            "enabled": True,
            "min_score": min_score,
            "prefer_patterns": list(self.config.candidate_probe_additive_axis_prefer_patterns or ()),
            "bonus_patterns": list(self.config.candidate_probe_additive_axis_bonus_patterns or ()),
            "avoid_patterns": list(self.config.candidate_probe_additive_axis_avoid_patterns or ()),
            "scores": details,
            "strong_candidate_count": len(strong_items),
        }
        if not strong_items:
            candidate_probe_trace["additive_axis_guarded"]["selection"] = "bo_top1_fallback"
            return bo_top1_item
        strong_items.sort(
            key=lambda pair: (
                float(pair[0]),
                -int(pair[1].get("candidate_probe_rank", 999) or 999),
            ),
            reverse=True,
        )
        chosen_score, chosen_item = strong_items[0]
        candidate_probe_trace["additive_axis_guarded"]["selection"] = "probe"
        candidate_probe_trace["additive_axis_guarded"]["selected_score"] = round(float(chosen_score), 6)
        candidate_probe_trace["additive_axis_guarded"]["selected_candidate_index"] = chosen_item.get("candidate_index")
        return dict(chosen_item)

    def _rank_candidate_probe_items_with_gp_mean_guarded(
        self,
        *,
        shortlist_candidates: list[dict[str, Any]],
        probe_items: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        feature_columns = self._feature_columns()
        probe_indices = {
            int(item.get("candidate_index", -1) or -1)
            for item in probe_items
            if item.get("candidate_index") is not None
        }
        selector_items: list[dict[str, Any]] = []
        for item in shortlist_candidates:
            candidate = item.get("candidate")
            if not isinstance(candidate, dict):
                continue
            if str(item.get("shortlist_source", "")) == "candidate_probe_injected":
                candidate_index = int(item.get("candidate_index", -1) or -1)
                if candidate_index not in probe_indices:
                    continue
                if not self._candidate_probe_has_core_anchor_support(item):
                    continue
            selector_items.append(item)
        if not selector_items:
            return None
        train_candidates = [
            dict(row["candidate"])
            for row in history
            if isinstance(row.get("candidate"), dict) and row.get("result") is not None
        ]
        train_values = np.asarray(
            [
                float(row["result"])
                for row in history
                if isinstance(row.get("candidate"), dict) and row.get("result") is not None
            ],
            dtype=float,
        )
        legal_candidates = [
            dict(item["candidate"])
            for item in selector_items
            if isinstance(item.get("candidate"), dict)
        ]
        if len(train_candidates) < 2 or len(np.unique(train_values)) < 2:
            return selector_items[0]
        try:
            layout = DiscreteBOPlanner._feature_layout(self.env.param_space)  # noqa: SLF001
            train_x = DiscreteBOPlanner._encode_candidates(train_candidates, layout)  # noqa: SLF001
            cand_x = DiscreteBOPlanner._encode_candidates(legal_candidates, layout)  # noqa: SLF001
            dim = max(1, train_x.shape[1])
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2))
                + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                random_state=int(self.config.seed or 7),
                n_restarts_optimizer=1,
            )
            gp.fit(train_x, train_values)
            mu = gp.predict(cand_x, return_std=False)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("candidate_probe GP mean selector failed: %s", exc)
            return None
        def adjusted_score(mean_value: float, item: dict[str, Any]) -> float:
            score = float(mean_value)
            if str(item.get("shortlist_source", "")) != "candidate_probe_injected":
                return score
            candidate = item.get("candidate")
            if not isinstance(candidate, dict):
                return score
            key = tuple(str(candidate.get(name)) for name in feature_columns)
            recurrence_count = max(0, int(self._candidate_probe_seen_counts.get(key, 0)))
            if recurrence_count >= 2 and self._candidate_probe_has_core_anchor_support(item):
                score += float(self.config.candidate_probe_recurrence_bonus or 0.0) * min(
                    recurrence_count - 1,
                    3,
                )
                evidence = item.get("probe_evidence")
                dim_support = evidence.get("dim_support", {}) if isinstance(evidence, dict) else {}
                additive_support = (
                    float(dim_support.get("Additive", 0.0) or 0.0)
                    if isinstance(dim_support, dict)
                    else 0.0
                )
                if additive_support < 0.5:
                    score += float(self.config.candidate_probe_additive_novelty_bonus or 0.0)
            return score

        ranked = sorted(
            zip(mu.tolist(), selector_items, strict=False),
            key=lambda pair: (
                adjusted_score(float(pair[0]), pair[1]),
                float(pair[0]),
                1 if str(pair[1].get("shortlist_source", "")) == "candidate_probe_injected" else 0,
                -int(pair[1].get("candidate_index", 999) or 999),
            ),
            reverse=True,
        )
        if not ranked:
            return None
        chosen_mu, chosen_item = ranked[0]
        bo_top1_item = self._main_bo_top1_item(shortlist_candidates)
        if (
            bo_top1_item is not None
            and bool(self.config.enable_late_stage_incumbent_protection)
        ):
            current_iteration = max(1, len(history) - int(self.config.init_budget or 0) + 1)
            current_best = float(np.max(train_values)) if len(train_values) else None
            current_best_percentile = self._current_best_percentile(current_best)
            strong_incumbent_present = (
                current_best_percentile is not None
                and current_best_percentile
                >= float(self.config.late_stage_strong_incumbent_percentile)
            )
            if self._late_stage_active(current_iteration) and strong_incumbent_present:
                bo_key = tuple(str(bo_top1_item.get("candidate", {}).get(name)) for name in feature_columns)
                chosen_key = tuple(str(chosen_item.get("candidate", {}).get(name)) for name in feature_columns)
                if chosen_key != bo_key:
                    bo_mu = None
                    for item_mu, item in ranked:
                        item_key = tuple(str(item.get("candidate", {}).get(name)) for name in feature_columns)
                        if item_key == bo_key:
                            bo_mu = float(item_mu)
                            break
                    chosen_is_probe = str(chosen_item.get("shortlist_source", "")) == "candidate_probe_injected"
                    margin = float(self.config.candidate_probe_late_stage_min_gp_mean_margin or 0.0)
                    probe_has_clear_advantage = (
                        chosen_is_probe
                        and bo_mu is not None
                        and float(chosen_mu) >= bo_mu + margin
                    )
                    if not probe_has_clear_advantage:
                        return bo_top1_item
        return dict(chosen_item)

    def _rank_candidate_probe_items_with_discrete_ei(
        self,
        *,
        probe_items: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not probe_items:
            return None
        feature_columns = self._feature_columns()
        train_candidates = [
            dict(row["candidate"])
            for row in history
            if isinstance(row.get("candidate"), dict) and row.get("result") is not None
        ]
        train_values = np.asarray(
            [
                float(row["result"])
                for row in history
                if isinstance(row.get("candidate"), dict) and row.get("result") is not None
            ],
            dtype=float,
        )
        legal_candidates = [
            dict(item["candidate"])
            for item in probe_items
            if isinstance(item.get("candidate"), dict)
        ]
        if not train_candidates or len(legal_candidates) <= 1:
            return probe_items[0]
        planner = DiscreteBOPlanner(
            seed=int(self.config.seed or 7),
            goal=self.env.goal,
            num_init_design=int(self.config.init_budget or 5),
        )
        try:
            ranked = planner._score_candidates(  # noqa: SLF001
                train_candidates=train_candidates,
                train_values=train_values,
                legal_candidates=legal_candidates,
                subspace=self.env.param_space,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("candidate_probe discrete EI selector failed: %s", exc)
            return probe_items[0]
        if not ranked:
            return probe_items[0]
        chosen_key = tuple(str(ranked[0][1].get(name)) for name in feature_columns)
        for item in probe_items:
            candidate = item.get("candidate")
            if not isinstance(candidate, dict):
                continue
            key = tuple(str(candidate.get(name)) for name in feature_columns)
            if key == chosen_key:
                return item
        return probe_items[0]

    def _shape_shortlist_preferred_item(
        self,
        *,
        shortlist_candidates: list[dict[str, Any]],
        history: list[dict[str, Any]],
        iteration: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        mode = str(self.config.shape_shortlist_selection_mode or "bo_top1").strip().lower()
        trace: dict[str, Any] = {
            "enabled": mode not in {"", "off", "none", "bo_top1"},
            "mode": mode,
        }
        if not trace["enabled"]:
            trace["reason"] = "disabled"
            return None, trace
        max_iteration = int(self.config.shape_shortlist_selection_max_iteration or 0)
        if max_iteration > 0 and int(iteration) > max_iteration:
            trace["reason"] = "after_max_iteration"
            trace["iteration"] = int(iteration)
            trace["max_iteration"] = max_iteration
            return None, trace
        top_k = max(1, int(self.config.shape_shortlist_selection_top_k or 3))
        selector_items = [
            item
            for item in shortlist_candidates[:top_k]
            if isinstance(item, dict) and isinstance(item.get("candidate"), dict)
        ]
        trace["top_k"] = top_k
        trace["candidate_indices"] = [
            int(item.get("candidate_index", idx) or idx)
            for idx, item in enumerate(selector_items)
        ]
        if not selector_items:
            trace["reason"] = "empty_selector_items"
            return None, trace
        if mode in {
            "topk_discrete_ei",
            "top3_discrete_ei",
            "shaped_topk_discrete_ei",
            "shaped_top3_discrete_ei",
        }:
            chosen_item = self._rank_candidate_probe_items_with_discrete_ei(
                probe_items=selector_items,
                history=history,
            )
            if chosen_item is None:
                trace["reason"] = "selector_returned_none"
                return None, trace
            trace["reason"] = "selected"
            trace["selected_index"] = int(chosen_item.get("candidate_index", 0) or 0)
            return dict(chosen_item), trace
        trace["reason"] = "unknown_mode"
        return None, trace

    def _apply_initial_candidates(self) -> list[dict[str, Any]]:
        if not self.initial_candidates:
            return []
        names = [param.name for param in self.env.param_space]
        applied: list[dict[str, Any]] = []
        seen = {
            tuple(str(v) for v in row)
            for row in self.campaign.observations.get_params(as_array=True)
        }
        for candidate in self.initial_candidates[: self.init_budget]:
            key = tuple(str(candidate.get(name)) for name in names)
            if key in seen:
                continue
            result = self.execution_adapter.evaluate_and_observe(candidate)
            applied.append({**candidate, self.env.objective_name: float(result)})
            seen.add(key)
        return applied

    def _build_initial_state_history(
        self,
        initial_observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for idx, item in enumerate(initial_observations, start=1):
            candidate = {
                param.name: item.get(param.name)
                for param in self.env.param_space
                if param.name in item
            }
            result = item.get(self.env.objective_name)
            records.append(
                {
                    "iteration": 0,
                    "stage": "init_design",
                    "trigger_reasons": ["init_design"],
                    "controller_mode": "init_design",
                    "intervention_type": "init_design",
                    "subspace_active": False,
                    "active_variables": [param.name for param in self.env.param_space],
                    "candidate": candidate,
                    "result": result,
                    "improved_best": None,
                    "feasibility_action": "accept",
                    "semantic_risk_level": "unknown",
                    "knowledge_used": False,
                    "knowledge_source_types": [],
                    "reflection": {
                        "insight": f"Initial design observation {idx}.",
                    },
                    "candidate_pool_size": self.dataset_meta.get("candidate_count"),
                    "observation_source": "initial_design",
                }
            )
        return records

    def _observation_history_with_initial(
        self,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return the full observed context for local candidate tools."""
        combined: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        feature_columns = self._feature_columns()
        for row in [*getattr(self, "_initial_state_history", []), *history]:
            candidate = row.get("candidate")
            if not isinstance(candidate, dict) or row.get("result") is None:
                continue
            key = tuple(str(candidate.get(name)) for name in feature_columns)
            if key in seen:
                continue
            seen.add(key)
            combined.append(row)
        return combined

    @staticmethod
    def _sample_to_dict(sample, param_space) -> dict[str, Any]:  # noqa: ANN001
        return {param.name: getattr(sample, param.name) for param in param_space}

    def _current_best(self) -> float | None:
        values = self.campaign.observations.get_values(as_array=True)
        if len(values) == 0:
            return None
        flat = values.reshape(-1)
        if self.env.goal == "minimize":
            return float(flat.min())
        return float(flat.max())

    def _known_constraint_bundle(
        self,
        *,
        use_subspace: bool,
        active_variables: list[str],
        best_observation: dict[str, Any] | None,
        iteration: int = 0,
    ) -> tuple[list[Any], dict[str, Any], set[tuple[str, ...]] | None]:
        if not self.env.is_finite_pool:
            return [], {
                "mode": "none",
                "pool_size": None,
                "total_candidate_count": None,
                "focus_variables": [],
                "filter_mode": self.config.filter_mode,
                "pool_size_before": None,
                "pool_size_after": None,
            }, None
        focus = active_variables if use_subspace else []
        allowed_keys, subpool_empty_fallback = self.env.filter_candidates(
            focus_variables=focus,
            best_observation=best_observation,
            filter_mode=self.config.filter_mode,
        )
        allowed_keys = set(allowed_keys or self.env.candidate_keys() or set())
        total_candidate_count = self.env.candidate_count
        summary = {
            "mode": "focused_filter" if focus else "full_pool",
            "filter_mode": self.config.filter_mode,
            "pool_size": len(allowed_keys),
            "pool_size_before": total_candidate_count,
            "pool_size_after": len(allowed_keys),
            "total_candidate_count": total_candidate_count,
            "focus_variables": list(focus),
            "subpool_empty_fallback": subpool_empty_fallback,
            "constraint_signature": self.env.allowed_keys_signature(allowed_keys),
        }

        # Layer LLM search constraints on top (Phase B)
        if (
            self.config.enable_llm_search_constraint
            and self._active_llm_constraints
            and iteration < self._llm_constraint_expires_at
        ):
            llm_allowed_keys, llm_fallback, llm_constraint_summary = self.env.filter_candidate_keys_by_constraint_specs(
                constraint_specs=self._active_llm_constraints,
                min_pool_fraction=self.config.min_constraint_pool_fraction,
            )
            llm_allowed_keys = set(llm_allowed_keys or set())
            combined_allowed_keys = allowed_keys & llm_allowed_keys
            min_pool_size = max(
                1,
                int((total_candidate_count or len(allowed_keys) or 1) * self.config.min_constraint_pool_fraction),
            )
            if not llm_fallback and len(combined_allowed_keys) >= min_pool_size:
                allowed_keys = combined_allowed_keys
                summary["llm_constraint_active"] = True
                summary["llm_constraint_pool_size"] = len(allowed_keys)
                summary["llm_constraint_summary"] = llm_constraint_summary
                summary["pool_size"] = len(allowed_keys)
                summary["pool_size_after"] = len(allowed_keys)
                summary["constraint_signature"] = self.env.allowed_keys_signature(allowed_keys)
            else:
                summary["llm_constraint_active"] = False
                summary["llm_constraint_fallback"] = True
        else:
            summary["llm_constraint_active"] = False

        membership_constraint = self.env.build_membership_constraint(allowed_keys)
        constraints = [membership_constraint] if membership_constraint is not None else []
        return constraints, summary, allowed_keys

    def _update_llm_constraints(
        self,
        iteration: int,
        decision_context: dict[str, Any],
        history_tail: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Call LLM to generate search constraints; update internal state."""
        if self.decision_engine is None or not self.env.is_finite_pool:
            return {"updated": False, "summary": "", "specs": []}
        search_space_meta = self.env.search_space_meta()
        try:
            action = self.decision_engine.generate_search_constraints(
                decision_context=decision_context,
                search_space_meta=search_space_meta,
                history_tail=history_tail,
            )
        except Exception:  # noqa: BLE001
            return {"updated": False, "summary": "", "specs": []}
        specs = action.get("constraints", [])
        if specs:
            max_duration = min(
                self.config.constraint_max_duration_rounds,
                max((s.get("duration_rounds", 5) for s in specs), default=5),
            )
            self._active_llm_constraints = specs
            self._llm_constraint_expires_at = iteration + max_duration
            self._llm_constraint_summary = action.get("constraint_summary", "")
        else:
            self._active_llm_constraints = []
            self._llm_constraint_expires_at = 0
            self._llm_constraint_summary = ""
        self._llm_constraint_last_updated_at = iteration
        return {
            "updated": bool(specs),
            "summary": self._llm_constraint_summary,
            "specs": specs,
        }

    def _completion_fallback_defaults(self) -> dict[str, Any]:
        """Prefer best observed settings for inactive variables, then static defaults."""
        merged = dict(self.defaults)
        best_observation = self._best_observation()
        if best_observation is None:
            return merged
        for param in self.env.param_space:
            if param.name in best_observation:
                merged[param.name] = best_observation[param.name]
        return merged

    def _candidate_seen_before(self, candidate: dict[str, Any], history: list[dict[str, Any]]) -> bool:
        for row in history:
            prior = row.get("candidate")
            if isinstance(prior, dict) and prior == candidate:
                return True
        return False

    def _should_trigger_sparse_intervention(
        self,
        decision_context: dict[str, Any],
        iteration: int,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        best_impr = decision_context.get("best_improvement_last_3")
        no_improvement_rounds = int(decision_context.get("no_improvement_rounds", 0) or 0)
        if no_improvement_rounds >= 4 or (
            best_impr is not None
            and float(best_impr) <= self.config.stagnation_threshold
            and no_improvement_rounds >= 3
        ):
            reasons.append("stagnation")
        if float(decision_context.get("recent_duplicate_ratio", 0.0)) >= self.config.duplicate_threshold:
            reasons.append("duplicate_high")
        coverage_gate_ratio = self._coverage_gate_ratio(decision_context)
        if (
            iteration > 8
            and coverage_gate_ratio <= self.config.coverage_threshold
        ):
            reasons.append("coverage_low")
        if (
            self.config.enable_multigranularity_lock_signals
            and self.env.is_finite_pool
        ):
            if (
                iteration > 5
                and no_improvement_rounds >= 3
                and bool(decision_context.get("one_axis_sweep_detected", False))
            ):
                reasons.append("one_axis_local_lock")
            if (
                iteration > 5
                and no_improvement_rounds >= 4
                and self._local_lock_trigger_active(decision_context)
            ):
                reasons.append("local_lock_stall")
            if (
                iteration > 5
                and float(decision_context.get("coverage_min_key_dim_ratio", 1.0) or 1.0)
                <= self.config.coverage_threshold
            ):
                reasons.append("key_dim_coverage_low")
        return (len(reasons) > 0, reasons)

    def _controller_trigger_reasons(
        self,
        decision_context: dict[str, Any],
        trigger_reasons: list[str],
    ) -> list[str]:
        reasons = list(trigger_reasons)
        scaffold_concentration = float(
            decision_context.get("recent_scaffold_concentration", 0.0) or 0.0
        )
        if scaffold_concentration >= 0.8:
            reasons.append("scaffold_concentration_high")
        no_improvement_rounds = int(decision_context.get("no_improvement_rounds", 0) or 0)
        stalled = no_improvement_rounds >= 4
        if (
            stalled
            and scaffold_concentration >= 0.5
            and decision_context.get("best_observation") is not None
        ):
            reasons.append("post_breakthrough_stall")
        if (
            self.config.enable_multigranularity_lock_signals
            and self.env.is_finite_pool
        ):
            if self._local_lock_trigger_active(decision_context):
                reasons.append("key_dim_concentration_high")
            if bool(decision_context.get("one_axis_sweep_detected", False)):
                reasons.append("one_axis_local_lock")
            if int(decision_context.get("anchor_repeat_count", 0) or 0) >= 4:
                reasons.append("anchor_repeat")
            if (
                float(decision_context.get("coverage_min_key_dim_ratio", 1.0) or 1.0)
                <= self.config.coverage_threshold
            ):
                reasons.append("key_dim_underexplored")
        return list(dict.fromkeys(reasons))

    def _full_campaign_for_space(self) -> Campaign:
        campaign = Campaign()
        campaign.set_param_space(self.env.param_space)
        campaign.set_value_space(self.env.value_space)
        params = self.campaign.observations.get_params(as_array=True)
        values = self.campaign.observations.get_values(as_array=True)
        if len(values) == 0:
            return campaign
        for row, value in zip(params, values):
            full_dict = {
                param.name: row[idx]
                for idx, param in enumerate(self.env.param_space)
            }
            param_vec = ParameterVector().from_dict(full_dict, param_space=self.env.param_space)
            if hasattr(value, "reshape"):
                cast_value = float(value.reshape(-1)[0])
            else:
                cast_value = float(value)
            campaign.add_observation(param_vec, cast_value)
        return campaign

    @staticmethod
    def _compact_knowledge_units(
        units: list[dict[str, Any]],
        *,
        max_items: int = 8,
        max_chars: int = 280,
    ) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for item in units[:max_items]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", ""))
            compact.append(
                {
                    "id": item.get("id"),
                    "source_type": item.get("source_type", "unknown"),
                    "knowledge_type": item.get("knowledge_type", "unknown"),
                    "confidence": item.get("confidence"),
                    "score": item.get("score"),
                    "content": (content[:max_chars] + "...") if len(content) > max_chars else content,
                }
            )
        return compact

    @staticmethod
    def _decision_context_snapshot(decision_context: dict[str, Any]) -> dict[str, Any]:
        return build_decision_context_snapshot(decision_context)

    def _reviewed_knowledge_enabled_for(self, node_name: str) -> bool:
        if not self.config.enable_reviewed_knowledge:
            return False
        if self.reviewed_knowledge_store is None or self.reviewed_knowledge_query_builder is None:
            return False
        target_nodes = {str(item).strip() for item in self.config.reviewed_knowledge_target_nodes}
        return node_name in target_nodes

    @staticmethod
    def _normalize_mode(value: Any) -> str:
        if isinstance(value, bool):
            return "decision_active" if value else "off"
        normalized = str(value).strip().lower()
        if normalized in {"false", "no", "0", ""}:
            return "off"
        if normalized in {"true", "yes", "1"}:
            return "decision_active"
        return normalized

    def _knowledge_mode_for(self, node_name: str) -> str:
        if node_name == "design_init_experiments":
            return self._normalize_mode(self.config.init_reviewed_knowledge_mode)
        if node_name in {"stagnation_diagnosis", "hypothesis_action"}:
            return self._normalize_mode(self.config.diagnosis_reviewed_knowledge_mode)
        if node_name in {"semantic_assessment", "reflection_action"}:
            return self._normalize_mode(self.config.semantic_reviewed_knowledge_mode)
        if node_name == "verification_pass":
            return self._normalize_mode(self.config.verification_mode)
        return "off"

    def _allowed_reviewed_content_statuses(self, node_name: str) -> list[str]:
        mode = self._knowledge_mode_for(node_name)
        if mode == "decision_active":
            return [str(item) for item in self.config.reviewed_knowledge_decision_content_statuses]
        if mode == "advisory":
            return [str(item) for item in self.config.reviewed_knowledge_advisory_content_statuses]
        return []

    def _augment_context_with_reviewed_knowledge(
        self,
        *,
        node_name: str,
        decision_context: dict[str, Any],
        candidate: dict[str, Any] | None = None,
        result: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self._reviewed_knowledge_enabled_for(node_name):
            return decision_context, {
                "node_name": node_name,
                "enabled": False,
                "mode": self._knowledge_mode_for(node_name),
                "query": "",
                "filters": {},
                "retrieved_count": 0,
                "retrieved_units": [],
                "retrieved_by_source_type": {},
                "protocol_mode": self.config.protocol_mode,
            }
        mode = self._knowledge_mode_for(node_name)
        if mode == "off":
            return decision_context, {
                "node_name": node_name,
                "enabled": False,
                "mode": mode,
                "query": "",
                "filters": {},
                "retrieved_count": 0,
                "retrieved_units": [],
                "retrieved_by_source_type": {},
                "protocol_mode": self.config.protocol_mode,
            }
        node_views = decision_context.get("node_state_views", {})
        node_view = node_views.get(node_name, decision_context) if isinstance(node_views, dict) else decision_context
        request = self.reviewed_knowledge_query_builder.build(
            node_name=node_name,
            node_state_view=node_view,
            reaction_context=self.reaction_context,
            candidate=candidate,
            result=result,
        )
        pinned_snippets, snippets = self.reviewed_knowledge_store.search_partitioned(
            node_name=node_name,
            query=str(request.get("query", "")),
            dataset=str(request.get("dataset", "")),
            reaction_type=str(request.get("reaction_type", "")),
            variable_tags=list(request.get("variable_tags", [])),
            trigger_tags=list(request.get("trigger_tags", [])),
            top_k=int(self.config.reviewed_knowledge_top_k),
            allowed_content_statuses=self._allowed_reviewed_content_statuses(node_name),
        )
        experience_snippets = []
        if (
            node_name == "stagnation_diagnosis"
            and self.config.enable_reviewed_experience
            and self.reviewed_experience_store is not None
            and mode in {"advisory", "decision_active"}
        ):
            experience_snippets = self.reviewed_experience_store.search(
                node_name=node_name,
                dataset=str(request.get("dataset", "")),
                trigger_tags=list(request.get("trigger_tags", [])),
                node_state_view=node_view,
                top_k=int(self.config.reviewed_experience_top_k),
            )
        prompt_items = [item.to_prompt_item() for item in [*pinned_snippets, *snippets, *experience_snippets]]
        existing_units = list(decision_context.get("knowledge_units", []))
        merged_units = [*prompt_items, *existing_units]
        merged_meta = dict(decision_context.get("knowledge_meta", {}))
        merged_meta["reviewed_knowledge"] = {
            "enabled": True,
            "mode": mode,
            "node_name": node_name,
            "query": request.get("query", ""),
            "dataset": request.get("dataset", ""),
            "reaction_type": request.get("reaction_type", ""),
            "variable_tags": list(request.get("variable_tags", [])),
            "trigger_tags": list(request.get("trigger_tags", [])),
            "retrieved_count": len(prompt_items),
            "retrieved_by_source_type": {
                "reviewed_entry_pinned": len(pinned_snippets),
                "reviewed_entry": len(snippets),
                "reviewed_experience": len(experience_snippets),
            },
            "source_type": "reviewed_entry",
            "protocol_mode": self.config.protocol_mode,
        }
        augmented = deepcopy(decision_context)
        augmented["knowledge_units"] = merged_units
        augmented["knowledge_meta"] = merged_meta
        if isinstance(augmented.get("node_state_views"), dict):
            augmented_views = dict(augmented["node_state_views"])
            node_augmented = dict(augmented_views.get(node_name, node_view))
            node_augmented["knowledge_units"] = merged_units
            node_augmented["knowledge_meta"] = merged_meta
            augmented_views[node_name] = node_augmented
            augmented["node_state_views"] = augmented_views
        trace_payload = {
            "node_name": node_name,
            "enabled": True,
            "mode": mode,
            "query": request.get("query", ""),
            "filters": {
                "dataset": request.get("dataset", ""),
                "reaction_type": request.get("reaction_type", ""),
                "variable_tags": list(request.get("variable_tags", [])),
                "trigger_tags": list(request.get("trigger_tags", [])),
            },
            "retrieved_count": len(prompt_items),
            "retrieved_units": prompt_items,
            "retrieved_by_source_type": {
                "reviewed_entry_pinned": len(pinned_snippets),
                "reviewed_entry": len(snippets),
                "reviewed_experience": len(experience_snippets),
            },
            "protocol_mode": self.config.protocol_mode,
        }
        self._append_knowledge_hit_ledger(
            iteration=(
                node_view.get("iteration")
                if isinstance(node_view, dict)
                else decision_context.get("iteration")
            ),
            node_name=node_name,
            mode=mode,
            request=request,
            retrieved_units=prompt_items,
            node_state_view=node_view if isinstance(node_view, dict) else decision_context,
        )
        return augmented, trace_payload

    def _scaffold_dimensions(self) -> list[str]:
        scaffold_dims = list(self.dataset_meta.get("scaffold_dims", []))
        dims = [name for name in scaffold_dims if isinstance(name, str)]
        if not dims:
            feature_columns = list(self.dataset_meta.get("feature_columns", []))
            dims = [name for name in feature_columns[:2] if isinstance(name, str)]
        if not dims:
            dims = [param.name for param in self.env.param_space[:2]]
        return dims

    def _key_dimensions(self) -> list[str]:
        key_dims = list(self.dataset_meta.get("key_dimensions", []))
        dims = [name for name in key_dims if isinstance(name, str)]
        if dims:
            return dims
        return self._scaffold_dimensions()

    def _candidate_scaffold_key(self, candidate: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(candidate.get(name)) for name in self._scaffold_dimensions())

    def _feature_columns(self) -> list[str]:
        feature_columns = list(self.dataset_meta.get("feature_columns", []))
        dims = [name for name in feature_columns if isinstance(name, str)]
        if dims:
            return dims
        return [param.name for param in self.env.param_space]

    def _runtime_contrast_adapter(self):
        return build_dataset_contrast_adapter(
            dataset=str(self.env.dataset),
            reaction_type=str(self.dataset_meta.get("reaction_type", "") or ""),
            feature_columns=self._feature_columns(),
            scaffold_dims=self._scaffold_dimensions(),
        )

    def _dominant_scaffold_key(
        self,
        dominant_scaffold: dict[str, Any] | None,
    ) -> tuple[str, ...]:
        if not isinstance(dominant_scaffold, dict):
            return ()
        return tuple(str(dominant_scaffold.get(name)) for name in self._scaffold_dimensions())

    def _scaffold_shift(
        self,
        left: tuple[str, ...],
        right: tuple[str, ...],
    ) -> bool:
        if not left or not right or len(left) != len(right):
            return False
        return any(str(a) != str(b) for a, b in zip(left, right))

    @staticmethod
    def _score_by_index(
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

    def _annotate_structural_shift(
        self,
        shortlist_candidates: list[dict[str, Any]],
        *,
        bo_top1_candidate: dict[str, Any] | None,
        dominant_scaffold: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        bo_top1_scaffold = (
            self._candidate_scaffold_key(bo_top1_candidate)
            if isinstance(bo_top1_candidate, dict)
            else ()
        )
        dominant_scaffold_key = self._dominant_scaffold_key(dominant_scaffold)
        enriched: list[dict[str, Any]] = []
        shifted_from_bo = 0
        shifted_from_dominant = 0
        diversity_pool_shifted = 0
        for item in shortlist_candidates:
            candidate = item.get("candidate")
            scaffold_key = (
                self._candidate_scaffold_key(candidate)
                if isinstance(candidate, dict)
                else ()
            )
            shift_from_bo = self._scaffold_shift(scaffold_key, bo_top1_scaffold)
            shift_from_dominant = self._scaffold_shift(scaffold_key, dominant_scaffold_key)
            if shift_from_bo:
                shifted_from_bo += 1
            if shift_from_dominant:
                shifted_from_dominant += 1
            if item.get("pool_source") == "diversity_pool" and (
                shift_from_bo or shift_from_dominant
            ):
                diversity_pool_shifted += 1
            enriched.append(
                {
                    **item,
                    "candidate_scaffold_key": list(scaffold_key),
                    "bo_top1_scaffold_key": list(bo_top1_scaffold),
                    "dominant_scaffold_key": list(dominant_scaffold_key),
                    "structural_shift_from_bo_top1": shift_from_bo,
                    "structural_shift_from_dominant": shift_from_dominant,
                }
            )
        diversity_pool_items = [
            item for item in enriched if item.get("pool_source") == "diversity_pool"
        ]
        quality = "not_used"
        if diversity_pool_items:
            quality = "structural_shift" if diversity_pool_shifted > 0 else "weak_diversity_pool"
        return enriched, {
            "bo_top1_scaffold_key": list(bo_top1_scaffold),
            "dominant_scaffold_key": list(dominant_scaffold_key),
            "structural_shift_candidate_count": shifted_from_bo,
            "dominant_shift_candidate_count": shifted_from_dominant,
            "diversity_pool_quality": quality,
        }

    def _default_action_package(self, *, schema_version: str = "compat_v1") -> dict[str, Any]:
        return {
            "schema_version": schema_version,
            "intent": "exploit",
            "shortlist_policy": "plain",
            "repeat_policy": "allow",
            "selection_policy": "bo_top1",
            "verification_policy": "normal",
            "focus_policy": "full_space",
            "focus_variables": [],
            "window_rounds": 0,
            "candidate_probe_direction": {},
            "requested_execution_action": "direct_bo_pick",
            "admissible_execution_actions": ["direct_bo_pick"],
            "preferred_execution_action": "direct_bo_pick",
            "fallback_reason": None,
            "reasoning": "",
        }

    def _extract_action_package(self, controller_plan: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(controller_plan, dict):
            if bool(self.config.enable_action_package_v06):
                return self._default_action_package(schema_version="v0.6")
            if bool(self.config.enable_action_package_v2):
                return self._default_action_package(schema_version="v2")
            return self._default_action_package()
        action_package = controller_plan.get("action_package")
        if isinstance(action_package, dict):
            schema_version = str(action_package.get("schema_version", "") or "")
            if not schema_version:
                if bool(self.config.enable_action_package_v06):
                    schema_version = "v0.6"
                elif bool(self.config.enable_action_package_v2):
                    schema_version = "v2"
                else:
                    schema_version = "compat_v1"
            base = self._default_action_package(schema_version=schema_version)
            base.update(
                {
                    key: value
                    for key, value in action_package.items()
                    if key in base
                }
            )
            base["focus_variables"] = list(base.get("focus_variables", []) or [])
            base["window_rounds"] = int(base.get("window_rounds", 0) or 0)
            return base
        if bool(self.config.enable_action_package_v06):
            return self._default_action_package(schema_version="v0.6")
        mode = str(controller_plan.get("intervention_type", "bo_direct"))
        if mode == "bo_focus_then_rerank":
            return {
                "schema_version": "compat_v1",
                "intent": "probe",
                "shortlist_policy": "coverage_shape",
                "repeat_policy": "avoid_anchor_repeat",
                "selection_policy": "select_from_shaped_shortlist",
                "verification_policy": "normal",
                "focus_policy": "temporary_focus",
                "focus_variables": list(controller_plan.get("focus_variables", []) or []),
                "window_rounds": int(controller_plan.get("window_rounds", 0) or 0),
                "reasoning": str(controller_plan.get("reasoning", "")),
            }
        if mode == "bo_rerank_topk":
            return {
                "schema_version": "compat_v1",
                "intent": "balance",
                "shortlist_policy": "plain",
                "repeat_policy": "allow",
                "selection_policy": "select_from_shaped_shortlist",
                "verification_policy": "normal",
                "focus_policy": "full_space",
                "focus_variables": [],
                "window_rounds": 0,
                "reasoning": str(controller_plan.get("reasoning", "")),
            }
        return self._default_action_package()

    @staticmethod
    def _sync_action_package_with_controller_mode(
        action_package: dict[str, Any],
        *,
        controller_mode: str,
        focus_variables: list[str],
        window_rounds: int,
        reasoning: str,
    ) -> dict[str, Any]:
        synced = dict(action_package or {})
        synced["reasoning"] = reasoning
        selection_policy = str(synced.get("selection_policy", "bo_top1") or "bo_top1")
        if controller_mode == "bo_focus_then_rerank":
            if selection_policy == "bo_top1":
                selection_policy = "select_from_shaped_shortlist"
            synced["selection_policy"] = selection_policy
            synced["focus_policy"] = "temporary_focus"
            synced["focus_variables"] = list(focus_variables)
            synced["window_rounds"] = int(window_rounds)
        elif controller_mode == "bo_rerank_topk":
            if selection_policy == "bo_top1":
                selection_policy = "bo_top1_from_shaped_shortlist"
            synced["selection_policy"] = selection_policy
            synced["focus_policy"] = "full_space"
            synced["focus_variables"] = []
            synced["window_rounds"] = 0
        else:
            synced["selection_policy"] = "bo_top1"
            synced["focus_policy"] = "full_space"
            synced["focus_variables"] = []
            synced["window_rounds"] = 0
        return synced

    @staticmethod
    def _requested_execution_action(action_package: dict[str, Any] | None) -> str:
        package = dict(action_package or {})
        requested = str(
            package.get("requested_execution_action")
            or package.get("executed_execution_action")
            or "direct_bo_pick"
        ).strip()
        if requested:
            return requested
        return "direct_bo_pick"

    @staticmethod
    def _candidate_probe_direction_from_action_package(
        action_package: dict[str, Any] | None,
    ) -> dict[str, Any]:
        package = dict(action_package or {})
        direction = package.get("candidate_probe_direction")
        if isinstance(direction, dict):
            return dict(direction)
        include_map: dict[str, list[str]] = {}
        for item in list(package.get("candidate_probe_include") or []):
            if "=" not in str(item):
                continue
            name, value = str(item).split("=", 1)
            name = name.strip()
            value = value.strip()
            if not name or not value:
                continue
            include_map.setdefault(name, [])
            if value not in include_map[name]:
                include_map[name].append(value)
        if not include_map:
            return {}
        return {
            "include_map": include_map,
            "reasoning": str(package.get("candidate_probe_reasoning") or "").strip(),
        }

    @staticmethod
    def _execution_action_from_legacy_mode(controller_mode: str) -> str:
        mode = str(controller_mode or "bo_direct")
        if mode == "bo_focus_then_rerank":
            return "focused_shortlist_alt_pick"
        if mode == "bo_rerank_topk":
            return "shortlist_alt_pick"
        return "direct_bo_pick"

    @staticmethod
    def _execution_action_requires_shortlist(execution_action: str) -> bool:
        return str(execution_action or "").strip() in {
            "shape_only_bo_pick",
            "shape_then_probe_topk",
            "shortlist_alt_pick",
            "focused_shortlist_alt_pick",
            "finite_pool_candidate_probe",
        }

    @staticmethod
    def _execution_action_uses_focus(execution_action: str) -> bool:
        return str(execution_action or "").strip() == "focused_shortlist_alt_pick"

    @staticmethod
    def _shape_contract_satisfied(
        shortlist_shaping_trace: dict[str, Any] | None,
        *,
        shortlist_candidates: list[dict[str, Any]] | None = None,
    ) -> bool:
        trace = dict(shortlist_shaping_trace or {})
        if not bool(trace.get("enabled")):
            return False
        if bool(trace.get("order_changed")):
            return True
        retained = list(trace.get("retained_candidate_indices", []) or [])
        dropped = list(trace.get("dropped_candidate_indices", []) or [])
        if dropped:
            return True
        if shortlist_candidates and retained and len(retained) < len(shortlist_candidates):
            return True
        return False

    @staticmethod
    def _build_v06_execution_contract(
        *,
        requested_execution_action: str,
        selected_differs_from_bo_top1: bool,
        shortlist_shaping_trace: dict[str, Any] | None,
        resuggest_trace: dict[str, Any] | None = None,
        candidate_probe_trace: dict[str, Any] | None = None,
        shortlist_candidates: list[dict[str, Any]] | None = None,
        rerank_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        requested = str(requested_execution_action or "direct_bo_pick")
        shape_ok = Orchestrator._shape_contract_satisfied(
            shortlist_shaping_trace,
            shortlist_candidates=shortlist_candidates,
        )
        alt_ok = bool(selected_differs_from_bo_top1)
        resuggest_ok = bool((resuggest_trace or {}).get("mask_applied", False))
        candidate_probe_ok = bool((candidate_probe_trace or {}).get("applied", False))
        shape_probe_ok = bool(
            shape_ok
            and str((rerank_action or {}).get("prompt_style", "") or "") == "shape_probe_topk"
            and list((rerank_action or {}).get("candidate_scores", []) or [])
        )
        executed = requested
        contract_satisfied = True
        fallback_reason = None
        if requested == "direct_bo_pick":
            contract_satisfied = True
            shape_ok = False
            alt_ok = False
            resuggest_ok = False
            shape_probe_ok = False
        elif requested in {
            "mask_scaffold_corridor_resuggest",
            "mask_dominant_resuggest",
            "mask_low_repeat_resuggest",
        }:
            contract_satisfied = resuggest_ok
            shape_ok = False
            alt_ok = False
            shape_probe_ok = False
            if not contract_satisfied:
                executed = "direct_bo_pick"
                fallback_reason = (
                    (resuggest_trace or {}).get("fallback_reason")
                    or "resuggest_contract_not_satisfied"
                )
        elif requested == "finite_pool_candidate_probe":
            contract_satisfied = candidate_probe_ok
            alt_ok = False
            shape_probe_ok = False
            if not contract_satisfied:
                executed = "shape_only_bo_pick" if shape_ok else "direct_bo_pick"
                fallback_reason = (
                    (candidate_probe_trace or {}).get("reason")
                    or "candidate_probe_contract_not_satisfied"
                )
        elif requested == "shape_only_bo_pick":
            contract_satisfied = shape_ok
            shape_probe_ok = False
            if not contract_satisfied:
                executed = "direct_bo_pick"
                fallback_reason = "shape_only_contract_not_satisfied"
        elif requested == "shape_then_probe_topk":
            contract_satisfied = shape_probe_ok
            alt_ok = bool(selected_differs_from_bo_top1)
            if not shape_probe_ok:
                executed = "shape_only_bo_pick" if shape_ok else "direct_bo_pick"
                fallback_reason = "shape_probe_topk_without_candidate_level_comparison"
        elif requested == "shortlist_alt_pick":
            contract_satisfied = shape_ok and alt_ok
            shape_probe_ok = False
            if not alt_ok:
                executed = "shape_only_bo_pick" if shape_ok else "direct_bo_pick"
                fallback_reason = "alt_pick_returned_bo_top_ranked_candidate"
            elif not shape_ok:
                executed = "direct_bo_pick"
                fallback_reason = "alt_pick_without_effective_shortlist_shaping"
        elif requested == "focused_shortlist_alt_pick":
            contract_satisfied = shape_ok and alt_ok
            shape_probe_ok = False
            if not alt_ok:
                executed = "shape_only_bo_pick" if shape_ok else "direct_bo_pick"
                fallback_reason = "focused_alt_pick_returned_bo_top_ranked_candidate"
            elif not shape_ok:
                executed = "direct_bo_pick"
                fallback_reason = "focused_alt_pick_without_effective_shortlist_shaping"
        return {
            "requested_execution_action": requested,
            "executed_execution_action": executed,
            "contract_satisfied": bool(contract_satisfied),
            "fallback_reason": fallback_reason,
            "shape_contract_satisfied": bool(shape_ok),
            "alt_pick_contract_satisfied": bool(alt_ok),
            "shape_probe_contract_satisfied": bool(shape_probe_ok),
            "resuggest_contract_satisfied": bool(resuggest_ok),
            "candidate_probe_contract_satisfied": bool(candidate_probe_ok),
        }

    @staticmethod
    def _candidate_matches_anchor_repeat(
        item: dict[str, Any],
        decision_context: dict[str, Any],
    ) -> bool:
        candidate = item.get("candidate")
        if not isinstance(candidate, dict):
            return False
        anchor_dims = list(decision_context.get("one_axis_sweep_anchor_dims", []) or [])
        anchor_values = dict(decision_context.get("one_axis_sweep_anchor_values", {}) or {})
        if not anchor_dims or not anchor_values:
            return False
        return all(str(candidate.get(name)) == str(anchor_values.get(name)) for name in anchor_dims)

    @staticmethod
    def _main_bo_top1_item(shortlist_candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not shortlist_candidates:
            return None
        tagged = [
            item
            for item in shortlist_candidates
            if bool(item.get("is_main_bo_top1"))
        ]
        if tagged:
            return dict(tagged[0])
        main_pool_sorted = sorted(
            (
                item
                for item in shortlist_candidates
                if item.get("main_pool_rank") is not None
            ),
            key=lambda item: int(item.get("main_pool_rank", 999) or 999),
        )
        if main_pool_sorted:
            return dict(main_pool_sorted[0])
        return dict(shortlist_candidates[0])

    @staticmethod
    def _bo_preferred_item_from_shaped_shortlist(
        shortlist_candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not shortlist_candidates:
            return None
        main_pool_sorted = sorted(
            (
                item
                for item in shortlist_candidates
                if item.get("main_pool_rank") is not None
            ),
            key=lambda item: (
                int(item.get("main_pool_rank", 999) or 999),
                int(item.get("candidate_index", 999) or 999),
            ),
        )
        if main_pool_sorted:
            return dict(main_pool_sorted[0])
        diversity_sorted = sorted(
            shortlist_candidates,
            key=lambda item: (
                int(item.get("diversity_pool_rank", 999) or 999),
                int(item.get("candidate_index", 999) or 999),
            ),
        )
        if diversity_sorted:
            return dict(diversity_sorted[0])
        return dict(shortlist_candidates[0])

    def _shape_shortlist_candidates(
        self,
        *,
        shortlist_candidates: list[dict[str, Any]],
        history: list[dict[str, Any]],
        decision_context: dict[str, Any],
        action_package: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not shortlist_candidates:
            return [], {
                "enabled": False,
                "reason": "empty_shortlist",
                "summary": "No shortlist candidates were available for shaping.",
                "candidate_traces": [],
            }
        if not (self.config.enable_action_package_v2 or self.config.enable_action_package_v06):
            return shortlist_candidates, {
                "enabled": False,
                "reason": "action_package_disabled",
                "summary": "Action package shaping disabled; shortlist order preserved.",
                "candidate_traces": [],
            }

        intent = str(action_package.get("intent", "balance"))
        shortlist_policy = str(action_package.get("shortlist_policy", "plain"))
        repeat_policy = str(action_package.get("repeat_policy", "allow"))
        verification_policy = str(action_package.get("verification_policy", "normal"))
        selection_policy = str(action_package.get("selection_policy", "bo_top1"))
        key_dims = self._key_dimensions()
        underexplored_dims = set(decision_context.get("underexplored_dimensions", []) or [])
        recent_dim_counts = self._recent_dimension_value_counts(history, dims=key_dims)
        dominant_scaffold_key = self._dominant_scaffold_key(decision_context.get("dominant_scaffold"))
        verification_warns_extension = bool(
            decision_context.get("verification_warns_extension", False)
        )
        verification_identity_uncertain = bool(
            decision_context.get("verification_identity_uncertain", False)
        )

        intent_weights = {
            "exploit": {
                "bo_rank": 1.35,
                "coverage": 0.30,
                "contrast": 0.25,
                "support": 0.95,
                "value_risk_penalty": 1.20,
                "repeat_penalty": 1.00,
                "verification_penalty": 1.00,
            },
            "probe": {
                "bo_rank": 0.85,
                "coverage": 1.00,
                "contrast": 1.00,
                "support": 0.35,
                "value_risk_penalty": 0.75,
                "repeat_penalty": 1.35,
                "verification_penalty": 1.20,
            },
            "balance": {
                "bo_rank": 1.10,
                "coverage": 0.75,
                "contrast": 0.70,
                "support": 0.55,
                "value_risk_penalty": 0.95,
                "repeat_penalty": 1.15,
                "verification_penalty": 1.10,
            },
        }
        policy_weights = {
            "plain": {"coverage": 0.0, "contrast": 0.0, "repeat": 0.0},
            "diversity_shape": {"coverage": 0.25, "contrast": 0.75, "repeat": 1.00},
            "coverage_shape": {"coverage": 1.00, "contrast": 0.20, "repeat": 0.60},
            "contrast_shape": {"coverage": 0.15, "contrast": 1.10, "repeat": 0.40},
        }
        repeat_multipliers = {
            "allow": 0.0,
            "avoid_near_duplicate": 1.0,
            "avoid_anchor_repeat": 1.35,
        }

        weights = intent_weights.get(intent, intent_weights["balance"])
        policy = policy_weights.get(shortlist_policy, policy_weights["plain"])
        repeat_multiplier = repeat_multipliers.get(repeat_policy, 0.0)
        current_best = decision_context.get("best_value")
        try:
            current_best = float(current_best) if current_best is not None else None
        except (TypeError, ValueError):
            current_best = None
        best_by_dim_value: dict[tuple[str, str], float] = {}
        best_by_scaffold_key: dict[tuple[str, ...], float] = {}
        for row in history:
            try:
                observed_value = float(row.get("result"))
            except (TypeError, ValueError):
                continue
            if current_best is None:
                current_best = observed_value
            else:
                current_best = max(current_best, observed_value)
            row_candidate = dict(row.get("candidate", {}) or {})
            scaffold_key = self._candidate_scaffold_key(row_candidate)
            if scaffold_key:
                previous = best_by_scaffold_key.get(scaffold_key)
                if previous is None or observed_value > previous:
                    best_by_scaffold_key[scaffold_key] = observed_value
            for dim in key_dims:
                dim_key = (dim, str(row_candidate.get(dim)))
                previous = best_by_dim_value.get(dim_key)
                if previous is None or observed_value > previous:
                    best_by_dim_value[dim_key] = observed_value

        shaped_candidates: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        for current_rank, item in enumerate(shortlist_candidates):
            candidate = dict(item.get("candidate", {}) or {})
            pool_rank = int(item.get("bo_rank", current_rank + 1) or (current_rank + 1))
            main_pool_rank = item.get("main_pool_rank")
            diversity_pool_rank = item.get("diversity_pool_rank")
            if main_pool_rank is not None:
                bo_priority = max(0.0, 1.0 - 0.08 * max(0, int(main_pool_rank) - 1))
            else:
                diversity_rank = (
                    int(diversity_pool_rank)
                    if diversity_pool_rank is not None
                    else pool_rank
                )
                bo_priority = max(0.2, 0.55 - 0.05 * max(0, diversity_rank - 1))

            coverage_signal = 0.0
            for dim in key_dims:
                value = str(candidate.get(dim))
                seen_count = int(recent_dim_counts.get(dim, {}).get(value, 0) or 0)
                dim_bonus = max(0.0, 1.0 - 0.25 * seen_count)
                if dim in underexplored_dims:
                    dim_bonus += 0.35
                coverage_signal += dim_bonus
            if key_dims:
                coverage_signal /= len(key_dims)

            contrast_signal = 0.0
            if bool(item.get("structural_shift_from_bo_top1")):
                contrast_signal += 0.75
            if bool(item.get("structural_shift_from_dominant")):
                contrast_signal += 0.65
            if str(item.get("pool_source")) == "diversity_pool":
                contrast_signal += 0.30

            dim_support_signal = 0.0
            weak_dim_hits = 0
            for dim in key_dims:
                dim_key = (dim, str(candidate.get(dim)))
                seen_best = best_by_dim_value.get(dim_key)
                if seen_best is None:
                    dim_support = 0.58 if dim in underexplored_dims else 0.45
                elif current_best is not None and current_best > 0:
                    dim_support = min(1.0, max(0.0, seen_best / current_best))
                    if seen_best < 0.65 * current_best and dim not in underexplored_dims:
                        weak_dim_hits += 1
                else:
                    dim_support = 0.60
                dim_support_signal += dim_support
            if key_dims:
                dim_support_signal /= len(key_dims)
            else:
                dim_support_signal = 0.50

            candidate_scaffold_key = tuple(str(v) for v in item.get("candidate_scaffold_key", []))
            scaffold_best = best_by_scaffold_key.get(candidate_scaffold_key)
            if scaffold_best is None:
                scaffold_support_signal = (
                    0.56 if str(item.get("pool_source")) == "diversity_pool" else 0.45
                )
            elif current_best is not None and current_best > 0:
                scaffold_support_signal = min(1.0, max(0.0, scaffold_best / current_best))
            else:
                scaffold_support_signal = 0.60
            support_signal = 0.65 * dim_support_signal + 0.35 * scaffold_support_signal

            value_risk_penalty = 0.0
            if current_best is not None and current_best > 0:
                if scaffold_best is not None and scaffold_best < 0.65 * current_best:
                    value_risk_penalty += 0.25
                weak_dim_threshold = max(1, len(key_dims) - 1) if key_dims else 1
                if weak_dim_hits >= weak_dim_threshold:
                    value_risk_penalty += 0.18
                if support_signal < 0.42 and contrast_signal >= 0.65:
                    value_risk_penalty += 0.16
                if (
                    str(item.get("pool_source")) == "diversity_pool"
                    and support_signal < 0.38
                    and coverage_signal < 0.85
                ):
                    value_risk_penalty += 0.10

            repeat_penalty = 0.0
            recent_scaffold_hits = int(item.get("recent_scaffold_hits", 0) or 0)
            recent_primary_hits = int(item.get("recent_primary_dim_hits", 0) or 0)
            recent_secondary_hits = int(item.get("recent_secondary_dim_hits", 0) or 0)
            repeat_penalty += 0.20 * max(0, recent_scaffold_hits - 1)
            repeat_penalty += 0.10 * max(0, recent_primary_hits - 1)
            repeat_penalty += 0.08 * max(0, recent_secondary_hits - 1)
            if tuple(str(v) for v in item.get("candidate_scaffold_key", [])) == dominant_scaffold_key:
                repeat_penalty += 0.30
            if repeat_policy == "avoid_anchor_repeat" and self._candidate_matches_anchor_repeat(
                item,
                decision_context,
            ):
                repeat_penalty += 0.50
            repeat_penalty *= repeat_multiplier

            verification_penalty = 0.0
            if verification_warns_extension and not bool(item.get("structural_shift_from_dominant")):
                verification_penalty += 0.35
            if verification_identity_uncertain and str(item.get("pool_source")) != "diversity_pool":
                verification_penalty += 0.20
            if verification_policy == "strict":
                verification_penalty *= 1.25
            verification_penalty *= weights["verification_penalty"]

            probe_candidate = bool(
                contrast_signal >= 0.65
                or coverage_signal >= 0.75
                or (
                    str(item.get("pool_source")) == "diversity_pool"
                    and support_signal >= 0.40
                )
            )
            hard_repeat_risk = bool(
                repeat_penalty >= 0.55
                or (
                    repeat_policy == "avoid_anchor_repeat"
                    and self._candidate_matches_anchor_repeat(item, decision_context)
                )
            )
            shape_score = (
                weights["bo_rank"] * bo_priority
                + weights["support"] * support_signal
                + weights["coverage"] * policy["coverage"] * coverage_signal
                + weights["contrast"] * policy["contrast"] * contrast_signal
                - weights["repeat_penalty"] * policy["repeat"] * repeat_penalty
                - weights["value_risk_penalty"] * value_risk_penalty
                - verification_penalty
            )
            shaped_item = {
                **item,
                "shape_score": round(float(shape_score), 6),
                "shape_score_components": {
                    "bo_priority": round(float(bo_priority), 6),
                    "coverage_signal": round(float(coverage_signal), 6),
                    "contrast_signal": round(float(contrast_signal), 6),
                    "support_signal": round(float(support_signal), 6),
                    "value_risk_penalty": round(float(value_risk_penalty), 6),
                    "repeat_penalty": round(float(repeat_penalty), 6),
                    "verification_penalty": round(float(verification_penalty), 6),
                },
                "probe_candidate": probe_candidate,
                "hard_repeat_risk": hard_repeat_risk,
            }
            shaped_candidates.append(shaped_item)
            traces.append(
                {
                    "candidate_index": item.get("candidate_index"),
                    "bo_rank": pool_rank,
                    "main_pool_rank": main_pool_rank,
                    "diversity_pool_rank": diversity_pool_rank,
                    "pool_source": item.get("pool_source"),
                    "shortlist_source": item.get("shortlist_source"),
                    "candidate_scaffold_key": item.get("candidate_scaffold_key", []),
                    "shape_score": round(float(shape_score), 6),
                    "probe_candidate": probe_candidate,
                    "hard_repeat_risk": hard_repeat_risk,
                    "components": shaped_item["shape_score_components"],
                }
            )

        ranked_candidates = sorted(
            shaped_candidates,
            key=lambda item: (
                -float(item.get("shape_score", 0.0) or 0.0),
                int(item.get("main_pool_rank", 999) or 999),
                int(item.get("diversity_pool_rank", 999) or 999),
                int(item.get("candidate_index", 999) or 999),
            )
        )
        for shaped_rank, item in enumerate(ranked_candidates):
            item["shaped_rank"] = shaped_rank

        retain_target = len(ranked_candidates)
        if len(ranked_candidates) >= 4:
            retain_target -= 1
        if intent == "probe" and len(ranked_candidates) >= 5:
            retain_target -= 1
        minimum_retained = 1 if len(ranked_candidates) == 1 else 2
        retain_target = max(minimum_retained, min(len(ranked_candidates), retain_target))

        retained_indices: list[int] = []
        retention_reasons: dict[int, str] = {}

        def _candidate_index(item: dict[str, Any]) -> int:
            value = item.get("candidate_index", -1)
            try:
                return int(value)
            except (TypeError, ValueError):
                return -1

        def _retain(item: dict[str, Any] | None, reason: str) -> None:
            if not isinstance(item, dict):
                return
            idx = _candidate_index(item)
            if idx < 0 or idx in retained_indices:
                return
            retained_indices.append(idx)
            retention_reasons[idx] = reason

        best_main = next(
            (
                item
                for item in sorted(
                    ranked_candidates,
                    key=lambda candidate: (
                        int(candidate.get("main_pool_rank", 999) or 999),
                        -float(candidate.get("shape_score", 0.0) or 0.0),
                    ),
                )
                if item.get("main_pool_rank") is not None
            ),
            None,
        )
        best_probe = next((item for item in ranked_candidates if bool(item.get("probe_candidate"))), None)
        best_contrast = next(
            (
                item
                for item in ranked_candidates
                if float(
                    ((item.get("shape_score_components") or {}).get("contrast_signal", 0.0) or 0.0)
                ) >= 0.65
            ),
            None,
        )
        best_coverage = next(
            (
                item
                for item in ranked_candidates
                if float(
                    ((item.get("shape_score_components") or {}).get("coverage_signal", 0.0) or 0.0)
                ) >= 0.75
            ),
            None,
        )

        if intent == "exploit":
            _retain(best_main, "main_bo_anchor")
            if verification_warns_extension or verification_identity_uncertain:
                _retain(best_probe or best_contrast or best_coverage, "verification_probe_slot")
        elif intent == "balance":
            _retain(best_main, "main_bo_anchor")
            _retain(best_probe or best_contrast or best_coverage, "balance_probe_slot")
        else:
            _retain(best_probe or best_contrast or best_coverage, "probe_anchor")
            _retain(best_contrast or best_coverage or best_main, "probe_support")

        def _should_skip_strict(item: dict[str, Any]) -> bool:
            if repeat_policy != "allow" and bool(item.get("hard_repeat_risk")):
                return True
            if (
                verification_warns_extension
                and bool(item.get("is_main_bo_top1"))
                and not bool(item.get("structural_shift_from_dominant"))
                and intent in {"probe", "balance"}
            ):
                return True
            return False

        for strict_pass in (True, False):
            for item in ranked_candidates:
                if len(retained_indices) >= retain_target:
                    break
                idx = _candidate_index(item)
                if idx in retained_indices:
                    continue
                if strict_pass and _should_skip_strict(item):
                    continue
                retention_reasons[idx] = (
                    "shape_score_fill"
                    if strict_pass
                    else "fallback_fill"
                )
                retained_indices.append(idx)
            if len(retained_indices) >= retain_target:
                break

        retained_candidates = [
            item for item in ranked_candidates if _candidate_index(item) in retained_indices
        ]
        if not retained_candidates:
            retained_candidates = [ranked_candidates[0]]
            retained_indices = [_candidate_index(ranked_candidates[0])]
            retention_reasons[retained_indices[0]] = "fallback_top_ranked"

        for shaped_rank, item in enumerate(retained_candidates):
            item["shaped_rank"] = shaped_rank
            item["retained_for_selection"] = True
            item["retention_reason"] = retention_reasons.get(_candidate_index(item), "shape_score_fill")
        retained_lookup = {idx for idx in retained_indices}
        for trace in traces:
            try:
                trace_index = int(trace.get("candidate_index", -1))
            except (TypeError, ValueError):
                trace_index = -1
            trace["retained_for_selection"] = trace_index in retained_lookup
            trace["retention_reason"] = retention_reasons.get(trace_index)

        order_changed = [
            int(item.get("candidate_index", -1))
            for item in retained_candidates
        ] != [
            int(item.get("candidate_index", -1))
            for item in shortlist_candidates[: len(retained_candidates)]
        ] or len(retained_candidates) != len(shortlist_candidates)
        summary = (
            f"intent={intent}; shortlist_policy={shortlist_policy}; repeat_policy={repeat_policy}; "
            f"selection_policy={selection_policy}; verification_policy={verification_policy}; "
            f"retained={len(retained_candidates)}/{len(ranked_candidates)}; order_changed={order_changed}"
        )
        return retained_candidates, {
            "enabled": True,
            "reason": "action_package_v2",
            "summary": summary,
            "intent": intent,
            "shortlist_policy": shortlist_policy,
            "repeat_policy": repeat_policy,
            "selection_policy": selection_policy,
            "verification_policy": verification_policy,
            "order_changed": order_changed,
            "retained_candidate_indices": retained_indices,
            "dropped_candidate_indices": [
                _candidate_index(item)
                for item in ranked_candidates
                if _candidate_index(item) not in retained_lookup
            ],
            "candidate_traces": traces,
        }

    def _apply_override_guardrail(
        self,
        *,
        shortlist_candidates: list[dict[str, Any]],
        chosen_item: dict[str, Any],
        rerank_action: dict[str, Any] | None,
        candidate_contrastive_evidence: dict[str, Any] | None,
        enabled: bool,
        trusted_planner_mode_active: bool,
        trusted_planner_override_allowed: bool,
        trusted_planner_block_reason: str | None,
        trusted_main_pool_soft_override_allowed: bool,
        late_stage_active: bool,
        current_best_percentile: float | None,
        strong_incumbent_present: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return apply_override_guardrail(
            shortlist_candidates=shortlist_candidates,
            chosen_item=chosen_item,
            rerank_action=rerank_action,
            candidate_contrastive_evidence=candidate_contrastive_evidence,
            enabled=enabled,
            score_margin_threshold=self.config.override_score_margin,
            min_transfer_score=self.config.override_min_transfer_score,
            require_structural_shift=self.config.override_require_structural_shift,
            trusted_planner_mode_active=trusted_planner_mode_active,
            trusted_planner_override_allowed=trusted_planner_override_allowed,
            trusted_planner_block_reason=trusted_planner_block_reason,
            trusted_main_pool_soft_override_allowed=trusted_main_pool_soft_override_allowed,
            trusted_planner_score_margin_threshold=self.config.trusted_planner_override_score_margin,
            trusted_planner_min_transfer_score=self.config.trusted_planner_min_transfer_score,
            trusted_planner_min_hypothesis_score=self.config.trusted_planner_min_hypothesis_score,
            late_stage_active=late_stage_active,
            strong_incumbent_present=strong_incumbent_present,
            current_best_percentile=current_best_percentile,
            late_stage_incumbent_protection_enabled=self.config.enable_late_stage_incumbent_protection,
            late_stage_override_score_margin_threshold=self.config.late_stage_override_score_margin,
            late_stage_min_transfer_score=self.config.late_stage_min_transfer_score,
            late_stage_min_hypothesis_score=self.config.late_stage_min_hypothesis_score,
        )

    def _trusted_planner_mode_active(self) -> bool:
        policy = str(self.config.planner_trust_policy or "auto").strip().lower()
        if policy == "balanced":
            return False
        if policy == "conservative":
            return True
        trusted = {str(item).strip().lower() for item in self.config.trusted_planner_names}
        return str(self.config.planner_name).strip().lower() in trusted

    def _current_best_percentile(self, current_best: float | None) -> float | None:
        if current_best is None or not self.env.is_finite_pool:
            return None
        pool = getattr(self.env, "_finite_pool_table", None)
        if pool is None:
            return None
        try:
            series = pool._df[pool.target_column].astype(float)  # noqa: SLF001
        except Exception:  # noqa: BLE001
            return None
        if len(series) == 0:
            return None
        if self.env.goal == "minimize":
            pct = float((series >= float(current_best)).mean())
        else:
            pct = float((series <= float(current_best)).mean())
        return max(0.0, min(1.0, pct))

    def _late_stage_active(self, iteration: int) -> bool:
        start_iter = max(
            int(self.config.late_stage_start_iteration),
            int(self.total_budget * float(self.config.late_stage_start_fraction)),
        )
        return int(iteration) >= max(1, start_iter)

    def _rerank_policy_state(
        self,
        *,
        iteration: int,
        trusted_planner_mode_active: bool,
        trusted_planner_override_allowed: bool,
        trusted_pre_gate_skip: bool,
    ) -> dict[str, Any]:
        current_best = self._current_best()
        current_best_percentile = self._current_best_percentile(current_best)
        late_stage_active = self._late_stage_active(iteration)
        strong_incumbent_present = (
            current_best_percentile is not None
            and current_best_percentile >= float(self.config.late_stage_strong_incumbent_percentile)
        )
        planner_name = str(self.config.planner_name).strip().lower()
        prompt_style = "challenger_with_incumbent" if planner_name == "atlas" else "default"
        return {
            "planner_name": planner_name,
            "prompt_style": prompt_style,
            "trusted_planner_mode_active": trusted_planner_mode_active,
            "trusted_planner_override_allowed": trusted_planner_override_allowed,
            "trusted_pre_gate_skip": bool(trusted_pre_gate_skip),
            "late_stage_active": late_stage_active,
            "current_best": current_best,
            "current_best_percentile": current_best_percentile,
            "strong_incumbent_present": bool(strong_incumbent_present),
            "near_best_margin": 0.08 if prompt_style == "challenger_with_incumbent" else 0.03,
        }

    def _trusted_planner_override_gate(
        self,
        *,
        decision_context: dict[str, Any],
        diagnosis: dict[str, Any],
        trigger_reasons: list[str],
    ) -> tuple[bool, str | None]:
        no_improvement_rounds = int(decision_context.get("no_improvement_rounds", 0) or 0)
        diagnosis_text = " ".join(
            [
                str(diagnosis.get("stagnation_type", "")),
                str(diagnosis.get("recommended_intervention", "")),
                " ".join(str(item) for item in diagnosis.get("suspected_causes", []) or []),
                " ".join(str(item) for item in trigger_reasons or []),
            ]
        ).lower()
        has_diagnosis = bool(diagnosis.get("is_stagnating")) or any(
            keyword in diagnosis_text
            for keyword in ("stagnation", "stagnating", "local_overfocus", "overfocus", "scaffold_concentration")
        )
        if no_improvement_rounds < int(self.config.trusted_planner_min_no_improvement_rounds):
            return False, "trusted_planner_wait_for_sustained_stagnation"
        if not has_diagnosis:
            return False, "trusted_planner_requires_stagnation_or_local_overfocus_diagnosis"
        return True, None

    def _shortlist_scaffold_diversity(self, shortlist_candidates: list[dict[str, Any]]) -> int:
        if not shortlist_candidates:
            return 0
        return len(
            {
                self._candidate_scaffold_key(item.get("candidate", {}))
                for item in shortlist_candidates
                if isinstance(item.get("candidate"), dict)
            }
        )

    def _recent_scaffold_counts(self, history: list[dict[str, Any]], tail_size: int = 6) -> dict[tuple[str, ...], int]:
        counts: dict[tuple[str, ...], int] = {}
        for row in history[-tail_size:]:
            candidate = row.get("candidate")
            if not isinstance(candidate, dict):
                continue
            key = self._candidate_scaffold_key(candidate)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _recent_dimension_value_counts(
        self,
        history: list[dict[str, Any]],
        *,
        dims: list[str],
        tail_size: int = 6,
    ) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {name: {} for name in dims}
        for row in history[-tail_size:]:
            candidate = row.get("candidate")
            if not isinstance(candidate, dict):
                continue
            for name in dims:
                value = str(candidate.get(name))
                dim_counts = counts.setdefault(name, {})
                dim_counts[value] = dim_counts.get(value, 0) + 1
        return counts

    def _coverage_gate_ratio(self, decision_context: dict[str, Any]) -> float:
        overall_ratio = float(decision_context.get("coverage_overall_ratio", 1.0) or 1.0)
        min_key_ratio = float(decision_context.get("coverage_min_key_dim_ratio", overall_ratio) or overall_ratio)
        weighted_key_ratio = float(
            decision_context.get("coverage_weighted_key_dim_ratio", overall_ratio) or overall_ratio
        )
        mode = str(self.config.key_dim_coverage_mode or "mean").strip().lower()
        if mode == "min_key_dim":
            return min_key_ratio
        if mode == "weighted_key_dim":
            return weighted_key_ratio
        return overall_ratio

    def _local_lock_trigger_active(
        self,
        decision_context: dict[str, Any],
        *,
        threshold: float | None = None,
    ) -> bool:
        if not self.config.enable_multigranularity_lock_signals:
            return False
        local_lock_score = float(decision_context.get("local_lock_score", 0.0) or 0.0)
        primary_concentration = float(
            decision_context.get("recent_primary_dim_concentration", 0.0) or 0.0
        )
        anchor_repeat_count = int(decision_context.get("anchor_repeat_count", 0) or 0)
        trigger_threshold = float(
            threshold
            if threshold is not None
            else self.config.shortlist_early_diversity_scaffold_threshold
        )
        return bool(
            decision_context.get("one_axis_sweep_detected", False)
            or local_lock_score >= trigger_threshold
            or primary_concentration >= trigger_threshold
            or anchor_repeat_count >= 4
        )

    def _controller_soft_rerank_authorized(
        self,
        *,
        controller_mode: str,
        diagnosis_intervention: str,
        trigger_reasons: list[str],
    ) -> bool:
        if controller_mode not in {"bo_rerank_topk", "shortlist_alt_pick", "focused_shortlist_alt_pick"}:
            return False
        if not self.env.is_finite_pool:
            return False
        if not self.config.controller_allow_rerank_without_space_narrowing:
            return False
        if not self.config.finite_pool_sparse_coverage_not_veto_rerank:
            return False
        if diagnosis_intervention not in {
            "keep_full_space",
            "keep_full_space_with_caution",
            "observe_more",
            "",
        }:
            return False
        actionable_reasons = {
            "local_lock_stall",
            "one_axis_local_lock",
            "key_dim_underexplored",
            "key_dim_concentration_high",
            "anchor_repeat",
        }
        return any(str(reason) in actionable_reasons for reason in (trigger_reasons or []))

    def _annotate_shortlist_candidates(
        self,
        candidates: list[dict[str, Any]],
        history: list[dict[str, Any]],
        *,
        pool_source: str,
        shortlist_source: str,
    ) -> list[dict[str, Any]]:
        recent_scaffold_counts = self._recent_scaffold_counts(history)
        key_dims = self._key_dimensions()
        recent_dim_counts = self._recent_dimension_value_counts(history, dims=key_dims)
        annotated: list[dict[str, Any]] = []
        for item in candidates:
            candidate = item.get("candidate")
            if not isinstance(candidate, dict):
                continue
            pool_rank = item.get("bo_rank")
            main_pool_rank = int(pool_rank) if pool_source == "main_pool" and pool_rank is not None else None
            diversity_pool_rank = (
                int(pool_rank)
                if pool_source == "diversity_pool" and pool_rank is not None
                else None
            )
            scaffold_key = self._candidate_scaffold_key(candidate)
            primary_dim = key_dims[0] if key_dims else None
            secondary_dim = key_dims[1] if len(key_dims) > 1 else None
            primary_value = str(candidate.get(primary_dim)) if primary_dim is not None else None
            secondary_value = str(candidate.get(secondary_dim)) if secondary_dim is not None else None
            annotated.append(
                {
                    **item,
                    "scaffold_key": list(scaffold_key),
                    "recent_scaffold_hits": int(recent_scaffold_counts.get(scaffold_key, 0)),
                    "recent_primary_dim_hits": (
                        int(recent_dim_counts.get(primary_dim, {}).get(primary_value, 0))
                        if primary_dim is not None and primary_value is not None
                        else 0
                    ),
                    "recent_secondary_dim_hits": (
                        int(recent_dim_counts.get(secondary_dim, {}).get(secondary_value, 0))
                        if secondary_dim is not None and secondary_value is not None
                        else 0
                    ),
                    "main_pool_rank": main_pool_rank,
                    "diversity_pool_rank": diversity_pool_rank,
                    "is_main_bo_top1": bool(pool_source == "main_pool" and main_pool_rank == 1),
                    "pool_source": pool_source,
                    "shortlist_source": shortlist_source,
                }
            )
        return annotated

    def _should_use_diversity_pool(self, decision_context: dict[str, Any], controller_mode: str) -> bool:
        if controller_mode not in {
            "bo_rerank_topk",
            "shape_only_bo_pick",
            "shortlist_alt_pick",
            "focused_shortlist_alt_pick",
        }:
            return False
        if not self.env.is_finite_pool:
            return False
        scaffold_concentration = float(decision_context.get("recent_scaffold_concentration", 0.0) or 0.0)
        no_improvement_rounds = int(decision_context.get("no_improvement_rounds", 0) or 0)
        if self._local_lock_trigger_active(decision_context):
            return True
        return (
            scaffold_concentration >= self.config.shortlist_early_diversity_scaffold_threshold
            or no_improvement_rounds >= self.config.shortlist_early_diversity_no_improvement_rounds
        )

    def _infer_visible_evidence_state(
        self,
        *,
        decision_context: dict[str, Any],
        shortlist_scaffold_diversity: int,
        diversity_pool_candidate_count: int,
    ) -> tuple[str, list[str]]:
        no_improvement_rounds = int(decision_context.get("no_improvement_rounds", 0) or 0)
        recent_duplicate_ratio = float(decision_context.get("recent_duplicate_ratio", 0.0) or 0.0)
        recent_scaffold_concentration = float(
            decision_context.get("recent_scaffold_concentration", 0.0) or 0.0
        )
        remaining_budget = decision_context.get("remaining_budget")
        remaining_budget_int = None if remaining_budget is None else int(remaining_budget)
        signals: list[str] = []
        if no_improvement_rounds >= 3:
            signals.append("stall>=3")
        if recent_duplicate_ratio >= 0.35:
            signals.append("dup_ratio>=0.35")
        if recent_scaffold_concentration >= 0.55:
            signals.append("scaffold_concentration>=0.55")
        if shortlist_scaffold_diversity <= 1:
            signals.append("shortlist_diversity<=1")
        if diversity_pool_candidate_count > 0:
            signals.append("diversity_pool_available")
        if remaining_budget_int is not None and remaining_budget_int <= 3:
            signals.append("late_budget<=3")
        if remaining_budget_int is not None and remaining_budget_int <= 3 and no_improvement_rounds <= 1:
            return "strong_exploit", signals
        if no_improvement_rounds >= 3 and (
            recent_scaffold_concentration >= 0.55
            or shortlist_scaffold_diversity <= 1
            or diversity_pool_candidate_count > 0
        ):
            return "local_lock", signals
        if recent_duplicate_ratio >= 0.35:
            return "repeat_pressure", signals
        if (
            no_improvement_rounds <= 1
            and recent_duplicate_ratio < 0.2
            and recent_scaffold_concentration < 0.45
        ):
            return "evidence_sparse", signals
        return "rank_uncertain", signals

    @staticmethod
    def _dedupe_preserve_order(indices: list[int]) -> list[int]:
        seen: set[int] = set()
        ordered: list[int] = []
        for value in indices:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _build_state_router_guidance(
        self,
        *,
        decision_context: dict[str, Any],
        shortlist_candidates: list[dict[str, Any]],
        shortlist_scaffold_diversity: int,
        diversity_pool_candidate_count: int,
    ) -> dict[str, Any]:
        if not shortlist_candidates:
            return {
                "visible_evidence_state": "unknown",
                "visible_state_signals": [],
                "admissible_candidate_indices": [0],
                "preferred_candidate_indices": [0],
                "preferred_selection_policy": "bo_direct",
                "fallback_candidate_index": 0,
                "rationale": "Shortlist unavailable; keep BO top-1.",
            }
        dataset = str(getattr(self.env, "dataset", "") or "").strip().lower()
        bo_index = int(shortlist_candidates[0].get("candidate_index", 0) or 0)
        no_improvement_rounds = int(decision_context.get("no_improvement_rounds", 0) or 0)
        failed_action_family_rounds = int(
            decision_context.get("consecutive_failed_action_family_rounds", 0) or 0
        )
        last_action_family = str(decision_context.get("last_action_family", "") or "").strip()
        last_action_effective = bool(decision_context.get("last_action_effective", False))
        non_top_indices = [
            int(item.get("candidate_index", idx))
            for idx, item in enumerate(shortlist_candidates)
            if int(item.get("candidate_index", idx)) != bo_index
        ]
        top23_indices = [
            int(item.get("candidate_index", idx))
            for idx, item in enumerate(shortlist_candidates)
            if int(item.get("candidate_index", idx)) != bo_index
            and int(item.get("bo_rank", idx + 1) or (idx + 1)) <= 3
        ]
        top45_indices = [
            int(item.get("candidate_index", idx))
            for idx, item in enumerate(shortlist_candidates)
            if int(item.get("candidate_index", idx)) != bo_index
            and 4 <= int(item.get("bo_rank", idx + 1) or (idx + 1)) <= 5
        ]
        structural_indices = [
            int(item.get("candidate_index", idx))
            for idx, item in enumerate(shortlist_candidates)
            if int(item.get("candidate_index", idx)) != bo_index
            and (
                bool(item.get("structural_shift_from_bo_top1"))
                or bool(item.get("structural_shift_from_dominant"))
                or str(item.get("shortlist_source", "")) == "diversity_injected"
            )
        ]
        structural_top23_indices = [
            int(item.get("candidate_index", idx))
            for idx, item in enumerate(shortlist_candidates)
            if int(item.get("candidate_index", idx)) in set(top23_indices)
            and int(item.get("candidate_index", idx)) in set(structural_indices)
        ]
        structural_top45_indices = [
            int(item.get("candidate_index", idx))
            for idx, item in enumerate(shortlist_candidates)
            if int(item.get("candidate_index", idx)) in set(top45_indices)
            and int(item.get("candidate_index", idx)) in set(structural_indices)
        ]
        low_repeat_indices: list[int] = []
        non_top_items = [
            dict(item)
            for idx, item in enumerate(shortlist_candidates)
            if int(item.get("candidate_index", idx)) != bo_index
        ]
        low_repeat_top23_indices: list[int] = []
        low_repeat_top45_indices: list[int] = []
        if non_top_items:
            min_repeat = min(int(item.get("recent_scaffold_hits", 0) or 0) for item in non_top_items)
            low_repeat_indices = [
                int(item.get("candidate_index", 0))
                for item in non_top_items
                if int(item.get("recent_scaffold_hits", 0) or 0) == min_repeat
            ]
        top23_items = [
            dict(item)
            for idx, item in enumerate(shortlist_candidates)
            if int(item.get("candidate_index", idx)) in set(top23_indices)
        ]
        if top23_items:
            min_repeat_top23 = min(
                int(item.get("recent_scaffold_hits", 0) or 0) for item in top23_items
            )
            low_repeat_top23_indices = [
                int(item.get("candidate_index", 0))
                for item in top23_items
                if int(item.get("recent_scaffold_hits", 0) or 0) == min_repeat_top23
            ]
        top45_items = [
            dict(item)
            for idx, item in enumerate(shortlist_candidates)
            if int(item.get("candidate_index", idx)) in set(top45_indices)
        ]
        if top45_items:
            min_repeat_top45 = min(
                int(item.get("recent_scaffold_hits", 0) or 0) for item in top45_items
            )
            low_repeat_top45_indices = [
                int(item.get("candidate_index", 0))
                for item in top45_items
                if int(item.get("recent_scaffold_hits", 0) or 0) == min_repeat_top45
            ]
        visible_evidence_state, visible_state_signals = self._infer_visible_evidence_state(
            decision_context=decision_context,
            shortlist_scaffold_diversity=shortlist_scaffold_diversity,
            diversity_pool_candidate_count=diversity_pool_candidate_count,
        )
        admissible = [bo_index]
        preferred = [bo_index]
        preferred_policy = "bo_direct"
        failure_mode = "none"
        headroom_bucket = "top1"
        planner_rank_topk = 1
        rationale = "Default to BO top-1 when visible evidence does not justify intervention."
        if visible_evidence_state in {"evidence_sparse", "strong_exploit", "repeat_pressure"}:
            return {
                "visible_evidence_state": visible_evidence_state,
                "visible_state_signals": visible_state_signals,
                "admissible_candidate_indices": [bo_index],
                "preferred_candidate_indices": [bo_index],
                "preferred_selection_policy": preferred_policy,
                "failure_mode": failure_mode,
                "headroom_bucket": headroom_bucket,
                "planner_rank_topk": planner_rank_topk,
                "fallback_candidate_index": bo_index,
                "rationale": rationale,
            }
        if dataset == "arylation":
            failed_shape_only_streak = bool(
                last_action_family == "shape_only"
                and failed_action_family_rounds >= 1
                and not last_action_effective
            )
            shortlist_selection_failure = bool(
                visible_evidence_state in {"local_lock", "rank_uncertain"}
                and top23_indices
                and failed_shape_only_streak
            )
            trajectory_shaping_failure = bool(
                visible_evidence_state == "local_lock"
                and diversity_pool_candidate_count > 0
                and structural_top45_indices
                and failed_shape_only_streak
                and no_improvement_rounds >= 5
            )
            if visible_evidence_state in {"local_lock", "rank_uncertain"}:
                admissible = [bo_index, *top23_indices]
                preferred = low_repeat_top23_indices or structural_top23_indices or top23_indices or [bo_index]
                preferred_policy = "shape_probe_topk"
                failure_mode = (
                    "shortlist_selection_failure"
                    if shortlist_selection_failure
                    else "none"
                )
                headroom_bucket = "top2_3" if len(preferred) > 0 and preferred != [bo_index] else "top1"
                planner_rank_topk = 3 if headroom_bucket == "top2_3" else 1
                rationale = (
                    "Arylation discrete cases often need bounded shortlist comparison inside the BO near-top slice once shortlist structure alone stops helping."
                )
                if trajectory_shaping_failure:
                    admissible = [bo_index, *top23_indices, *top45_indices]
                    preferred = (
                        structural_top45_indices
                        or low_repeat_top45_indices
                        or structural_top23_indices
                        or low_repeat_top23_indices
                        or top23_indices
                        or [bo_index]
                    )
                    preferred_policy = "deeper_diversity_probe"
                    failure_mode = "trajectory_shaping_failure"
                    headroom_bucket = "top4_5"
                    planner_rank_topk = 5
                    rationale = (
                        "Arylation discrete local-lock cases can require a deeper diversity challenger because the current trajectory is confined to the wrong scaffold corridor."
                    )
        elif dataset == "buchwald_task_1":
            if visible_evidence_state == "local_lock":
                admissible = [bo_index, *non_top_indices]
                preferred = non_top_indices or [bo_index]
                preferred_policy = "challenger_pick"
                rationale = (
                    "Buchwald task 1 local-lock cases are compatible with shortlist challenger intervention."
                )
        elif dataset == "buchwald_task_2":
            admissible = [bo_index]
            preferred = [bo_index]
            preferred_policy = "bo_direct"
            rationale = (
                "Buchwald task 2 discrete cases penalize aggressive intervention; keep BO top-1 unless much stronger evidence appears."
            )
        elif dataset == "suzuki_hte_full":
            if visible_evidence_state == "local_lock":
                admissible = [bo_index, *structural_indices, *low_repeat_indices]
                preferred = low_repeat_indices or structural_indices or [bo_index]
                preferred_policy = "low_repeat_probe"
                rationale = (
                    "Suzuki local-lock cases benefit most from escaping repeated local sampling; prefer low-repeat or structural shortlist picks."
                )
        admissible = self._dedupe_preserve_order(admissible) or [bo_index]
        preferred = self._dedupe_preserve_order(preferred) or [bo_index]
        preferred = [idx for idx in preferred if idx in admissible] or [bo_index]
        return {
            "visible_evidence_state": visible_evidence_state,
            "visible_state_signals": visible_state_signals,
            "admissible_candidate_indices": admissible,
            "preferred_candidate_indices": preferred,
            "preferred_selection_policy": preferred_policy,
            "failure_mode": failure_mode,
            "headroom_bucket": headroom_bucket,
            "planner_rank_topk": planner_rank_topk,
            "fallback_candidate_index": bo_index,
            "rationale": rationale,
        }

    @staticmethod
    def _post_resuggest_selection_mode(
        *,
        execution_action: str,
        state_router_guidance: dict[str, Any] | None,
        resuggest_trace: dict[str, Any] | None,
    ) -> str:
        if str(execution_action or "").strip() not in {
            "mask_scaffold_corridor_resuggest",
            "mask_dominant_resuggest",
            "mask_low_repeat_resuggest",
        }:
            return "bo_top1"
        if not bool((resuggest_trace or {}).get("mask_applied", False)):
            return "bo_top1"
        guidance = dict(state_router_guidance or {})
        visible_evidence_state = str(guidance.get("visible_evidence_state", "") or "")
        preferred_selection_policy = str(
            guidance.get("preferred_selection_policy", "") or ""
        )
        preferred_candidate_indices = list(guidance.get("preferred_candidate_indices", []) or [])
        if visible_evidence_state not in {"local_lock", "rank_uncertain"}:
            return "bo_top1"
        if preferred_selection_policy not in {
            "low_repeat_probe",
            "challenger_pick",
            "shape_probe_topk",
            "deeper_diversity_probe",
        }:
            return "bo_top1"
        non_top_preferred = [
            int(idx)
            for idx in preferred_candidate_indices
            if int(idx) != int(guidance.get("fallback_candidate_index", 0) or 0)
        ]
        if not non_top_preferred:
            return "bo_top1"
        return "probe_topk"

    def _dominant_scaffold_constraint(
        self,
        dominant_scaffold: dict[str, Any] | None,
    ):
        if not dominant_scaffold:
            return None
        dims = [name for name in self._scaffold_dimensions() if name in dominant_scaffold]
        if not dims:
            return None

        def _avoid_dominant(values: Any, *, _dims=dims, _dominant=dict(dominant_scaffold)) -> bool:
            if self.env.is_finite_pool and getattr(self.env, "_finite_pool_table", None) is not None:
                try:
                    key_values = values.tolist() if hasattr(values, "tolist") else values
                    key = self.env._finite_pool_table.sample_to_key(key_values)
                    idx_map = {
                        name: idx
                        for idx, name in enumerate(self.env._finite_pool_table.feature_columns)
                    }
                    return any(key[idx_map[name]] != str(_dominant.get(name)) for name in _dims)
                except Exception:  # noqa: BLE001
                    pass
            if isinstance(values, dict):
                data = values
            elif hasattr(values, "to_dict"):
                data = values.to_dict()
            elif hasattr(values, "tolist"):
                arr = values.tolist()
                if isinstance(arr, list):
                    data = {
                        name: arr[idx]
                        for idx, name in enumerate(self._scaffold_dimensions())
                        if idx < len(arr)
                    }
                else:
                    data = {}
            else:
                data = {name: getattr(values, name) for name in _dims if hasattr(values, name)}
            return any(str(data.get(name)) != str(_dominant.get(name)) for name in _dims)

        return _avoid_dominant

    def _apply_pre_shortlist_resuggest(
        self,
        *,
        execution_action: str,
        decision_context: dict[str, Any],
        history: list[dict[str, Any]],
        allowed_keys: set[tuple[str, ...]] | None,
        effective_constraint_summary: dict[str, Any],
    ) -> tuple[list[Any], dict[str, Any], set[tuple[str, ...]] | None, dict[str, Any]]:
        trace = {
            "enabled": execution_action in {
                "mask_scaffold_corridor_resuggest",
                "mask_dominant_resuggest",
                "mask_low_repeat_resuggest",
            },
            "requested_execution_action": execution_action,
            "mask_applied": False,
            "fallback_reason": "not_requested",
            "pool_size_before": len(allowed_keys or []),
            "pool_size_after": len(allowed_keys or []),
            "excluded_candidate_count": 0,
            "mask_basis": {},
        }
        if not trace["enabled"] or not self.env.is_finite_pool or not allowed_keys:
            return [], effective_constraint_summary, allowed_keys, trace
        feature_columns = self._feature_columns()
        scaffold_dims = self._scaffold_dimensions()
        if execution_action == "mask_scaffold_corridor_resuggest":
            masked_keys, mask_summary = scaffold_corridor_mask_keys(
                allowed_keys=set(allowed_keys),
                feature_columns=feature_columns,
                scaffold_dims=scaffold_dims,
                dominant_values_by_dim=dict(
                    decision_context.get("dominant_values_by_dim", {}) or {}
                ),
                history=history,
            )
        elif execution_action == "mask_dominant_resuggest":
            masked_keys, mask_summary = dominant_mask_keys(
                allowed_keys=set(allowed_keys),
                feature_columns=feature_columns,
                scaffold_dims=scaffold_dims,
                dominant_scaffold=decision_context.get("dominant_scaffold"),
                history=history,
            )
        else:
            masked_keys, mask_summary = low_repeat_mask_keys(
                allowed_keys=set(allowed_keys),
                feature_columns=feature_columns,
                anchor_dims=list(decision_context.get("one_axis_sweep_anchor_dims", []) or []),
                anchor_values=dict(decision_context.get("one_axis_sweep_anchor_values", {}) or {}),
                dominant_scaffold=decision_context.get("dominant_scaffold"),
                scaffold_dims=scaffold_dims,
                history=history,
            )
        masked_keys = set(masked_keys or set())
        trace.update(mask_summary)
        min_pool_size = max(
            1,
            int((self.env.candidate_count or len(allowed_keys) or 1) * self.config.min_constraint_pool_fraction),
        )
        if not masked_keys or len(masked_keys) >= len(allowed_keys):
            trace["fallback_reason"] = trace.get("fallback_reason") or "mask_did_not_change_pool"
            return [], effective_constraint_summary, allowed_keys, trace
        if len(masked_keys) < min_pool_size:
            trace["mask_applied"] = False
            trace["fallback_reason"] = "mask_pool_below_min_fraction"
            trace["pool_size_after"] = len(allowed_keys)
            trace["excluded_candidate_count"] = 0
            return [], effective_constraint_summary, allowed_keys, trace
        membership_constraint = self.env.build_membership_constraint(masked_keys)
        if membership_constraint is None:
            trace["mask_applied"] = False
            trace["fallback_reason"] = "membership_constraint_unavailable"
            return [], effective_constraint_summary, allowed_keys, trace
        updated_summary = {
            **dict(effective_constraint_summary),
            "mode": "pre_shortlist_resuggest",
            "resuggest_action": execution_action,
            "resuggest_mask_applied": True,
            "resuggest_mask_basis": trace.get("mask_basis"),
            "pool_size_before": len(allowed_keys),
            "pool_size_after": len(masked_keys),
            "pool_size": len(masked_keys),
            "constraint_signature": self.env.allowed_keys_signature(masked_keys),
        }
        trace["mask_applied"] = True
        trace["fallback_reason"] = None
        trace["constraint_signature"] = updated_summary.get("constraint_signature")
        trace["base_constraint_signature"] = effective_constraint_summary.get("constraint_signature")
        return [membership_constraint], updated_summary, masked_keys, trace

    @staticmethod
    def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, Any], ...]] = set()
        for item in candidates:
            candidate = item.get("candidate")
            if not isinstance(candidate, dict):
                continue
            key = tuple(sorted(candidate.items()))
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def _build_dual_pool_shortlist(
        self,
        *,
        main_pool: list[dict[str, Any]],
        diversity_pool: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        available_pool = self._dedupe_candidates([*main_pool, *diversity_pool])
        if not available_pool:
            return [], {
                "diversity_injection_count": 0,
                "shortlist_target_diversity_met": False,
                "shortlist_available_scaffold_diversity": 0,
                "shortlist_source_mix": [],
                "shortlist_pool_strategy": "empty",
                "pool_limited_diversity": True,
                "diversity_pool_used": False,
                "diversity_pool_unique_scaffolds": 0,
            }

        final_size = min(max(1, int(self.config.shortlist_size)), len(available_pool))
        keep_count = min(
            max(1, int(self.config.shortlist_top_rank_keep)),
            final_size,
            len(main_pool),
        )
        selected: list[dict[str, Any]] = []
        selected_scaffolds: set[tuple[str, ...]] = set()
        selected_keys: set[tuple[tuple[str, Any], ...]] = set()

        for item in main_pool[:keep_count]:
            candidate = item.get("candidate", {})
            candidate_key = tuple(sorted(candidate.items()))
            scaffold_key = tuple(str(v) for v in item.get("scaffold_key", []))
            selected.append({**item, "shortlist_source": "bo_top_ranked"})
            selected_scaffolds.add(scaffold_key)
            selected_keys.add(candidate_key)

        available_scaffolds = {
            tuple(str(v) for v in item.get("scaffold_key", []))
            for item in available_pool
            if item.get("scaffold_key")
        }
        desired_diversity = min(
            max(1, int(self.config.shortlist_target_scaffold_diversity)),
            max(1, len(available_scaffolds)),
            final_size,
        )
        diversity_injection_count = 0

        for item in diversity_pool:
            if len(selected) >= final_size:
                break
            scaffold_key = tuple(str(v) for v in item.get("scaffold_key", []))
            candidate_key = tuple(sorted(item.get("candidate", {}).items()))
            if candidate_key in selected_keys:
                continue
            if len(selected_scaffolds) < desired_diversity and scaffold_key in selected_scaffolds:
                continue
            selected.append({**item, "shortlist_source": "diversity_injected"})
            selected_scaffolds.add(scaffold_key)
            selected_keys.add(candidate_key)
            diversity_injection_count += 1

        for item in main_pool[keep_count:]:
            if len(selected) >= final_size:
                break
            candidate_key = tuple(sorted(item.get("candidate", {}).items()))
            if candidate_key in selected_keys:
                continue
            selected.append({**item, "shortlist_source": "bo_top_ranked"})
            selected_keys.add(candidate_key)
            selected_scaffolds.add(tuple(str(v) for v in item.get("scaffold_key", [])))

        for item in diversity_pool:
            if len(selected) >= final_size:
                break
            candidate_key = tuple(sorted(item.get("candidate", {}).items()))
            if candidate_key in selected_keys:
                continue
            selected.append({**item, "shortlist_source": "diversity_injected"})
            selected_keys.add(candidate_key)
            selected_scaffolds.add(tuple(str(v) for v in item.get("scaffold_key", [])))
            diversity_injection_count += 1

        for idx, item in enumerate(selected):
            item["candidate_index"] = idx

        source_mix = [str(item.get("shortlist_source", "bo_top_ranked")) for item in selected]
        actual_diversity = self._shortlist_scaffold_diversity(selected)
        global_diversity_target = min(
            max(1, int(self.config.shortlist_min_scaffold_diversity)),
            final_size,
        )
        diversity_pool_scaffolds = {
            tuple(str(v) for v in item.get("scaffold_key", []))
            for item in diversity_pool
            if item.get("scaffold_key")
        }
        return selected, {
            "diversity_injection_count": diversity_injection_count,
            "shortlist_target_diversity_met": actual_diversity >= global_diversity_target,
            "shortlist_available_scaffold_diversity": len(available_scaffolds),
            "shortlist_source_mix": source_mix,
            "shortlist_pool_strategy": (
                "dual_pool_mixed"
                if diversity_pool
                else "main_pool_only"
            ),
            "pool_limited_diversity": len(available_scaffolds) < global_diversity_target,
            "diversity_pool_used": bool(diversity_pool),
            "diversity_pool_unique_scaffolds": len(diversity_pool_scaffolds),
        }

    def _is_improvement(self, result: float, best_before: float | None) -> bool:
        if best_before is None:
            return True
        if self.env.goal == "minimize":
            return float(result) < float(best_before)
        return float(result) > float(best_before)

    def _write_audit_artifact(
        self,
        *,
        iteration: int,
        name: str,
        payload: dict[str, Any],
    ) -> str | None:
        if self.audit_artifact_root is None:
            return None
        file_path = self.audit_artifact_root / f"iter_{iteration:03d}_{name}.json"
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        return str(file_path)

    @staticmethod
    def _knowledge_state_bucket(node_state_view: dict[str, Any]) -> dict[str, Any]:
        """Compact state features for offline knowledge governance.

        This intentionally avoids full prompts, full state views, and node outputs.
        Runtime only records enough metadata to let an offline auditor decide what
        evidence to inspect later.
        """
        bucket: dict[str, Any] = {}
        for key in (
            "iteration",
            "observations",
            "remaining_budget",
            "no_improvement_rounds",
            "best_improvement_last_3",
            "recent_duplicate_ratio",
            "coverage_overall_ratio",
            "recent_scaffold_concentration",
        ):
            value = node_state_view.get(key)
            if value is not None:
                bucket[key] = value
        if isinstance(node_state_view.get("semantic_assessment"), dict):
            risk_level = node_state_view["semantic_assessment"].get("risk_level")
            if risk_level is not None:
                bucket["risk_level"] = risk_level
        return bucket

    def _append_knowledge_hit_ledger(
        self,
        *,
        iteration: int | None,
        node_name: str,
        mode: str,
        request: dict[str, Any],
        retrieved_units: list[dict[str, Any]],
        node_state_view: dict[str, Any],
    ) -> None:
        if self.knowledge_hit_ledger_path is None or not retrieved_units:
            return
        state_bucket = self._knowledge_state_bucket(node_state_view)
        trigger_tags = list(request.get("trigger_tags", []))
        variable_tags = list(request.get("variable_tags", []))
        with self.knowledge_hit_ledger_path.open("a", encoding="utf-8") as handle:
            for rank, unit in enumerate(retrieved_units, start=1):
                if not isinstance(unit, dict):
                    continue
                metadata = unit.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                source_ref = unit.get("source_ref", {})
                if not isinstance(source_ref, dict):
                    source_ref = {}
                record = {
                    "run_id": self._run_id(),
                    "dataset": self.env.dataset,
                    "iteration": iteration,
                    "node": node_name,
                    "mode": mode,
                    "knowledge_id": unit.get("id"),
                    "source_type": unit.get("source_type"),
                    "knowledge_type": unit.get("knowledge_type"),
                    "rank": rank,
                    "score": unit.get("score"),
                    "confidence": unit.get("confidence"),
                    "content_status": metadata.get("content_status"),
                    "review_status": metadata.get("review_status"),
                    "benchmark_safety": metadata.get("benchmark_safety"),
                    "retrieval_reason": unit.get("retrieval_reason"),
                    "trigger_tags": trigger_tags,
                    "variable_tags": variable_tags,
                    "source_file": source_ref.get("source_file"),
                    "state_bucket": state_bucket,
                }
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _run_id(self) -> str:
        return (
            f"{self.env.dataset}__{self.config.method_family}__{self.config.planner_name}"
            f"__seed{self.config.seed}__tb{self.total_budget}__ib{self.init_budget}"
        )

    def _progress_enabled(self) -> bool:
        return str(self.config.terminal_verbosity).strip().lower() != "quiet"

    def _debug_progress_enabled(self) -> bool:
        return str(self.config.terminal_verbosity).strip().lower() == "debug"

    def _emit_progress(self, event: str, **payload: Any) -> None:
        record = {
            "event": event,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **payload,
        }
        if self.progress_jsonl_path is not None:
            with self.progress_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if self.status_json_path is not None:
            with self.status_json_path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, ensure_ascii=False, default=str)

        if not self._progress_enabled():
            return
        if event == "run_start":
            print(
                "[RUN] "
                f"dataset={payload.get('dataset')} planner={payload.get('planner_name')} "
                f"seed={payload.get('seed')} budget={payload.get('total_budget')} "
                f"init={payload.get('init_budget')}"
            )
        elif event == "iteration_start":
            print(
                "[PROGRESS] "
                f"iter={payload.get('iteration')}/{payload.get('bo_iterations')} "
                f"obs={payload.get('observations')}/{payload.get('total_budget')} "
                f"best={payload.get('best_result')} elapsed={payload.get('elapsed_sec')}s"
            )
        elif event == "phase_start":
            print(
                "[PHASE] "
                f"iter={payload.get('iteration')} phase={payload.get('phase')}"
            )
        elif event == "iteration_end":
            print(
                "[ITER] "
                f"{payload.get('iteration'):03d}/{payload.get('bo_iterations')} "
                f"mode={payload.get('controller_mode')} result={payload.get('result')} "
                f"best={payload.get('best_result')} improved={payload.get('improved_best')} "
                f"elapsed={payload.get('elapsed_sec')}s"
            )
        elif event == "run_end":
            print(
                "[DONE] "
                f"best={payload.get('best_result')} runtime={payload.get('runtime_sec')}s"
            )
        elif self._debug_progress_enabled():
            print(f"[DEBUG] {event}: {payload}")

    def _call_planner(self, label: str, func, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN001, ANN401
        with capture_third_party_output(
            enabled=bool(self.config.suppress_third_party_output),
            log_path=self.third_party_log_path,
            label=label,
        ):
            return func(*args, **kwargs)

    def run_optimization(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        start_time = time.time()
        iteration = 0
        last_intervention_type = "none"
        consecutive_focus_rounds = 0
        previous_focus_improved: bool | None = None
        initial_observations = self._apply_initial_candidates()
        self._initial_state_history = self._build_initial_state_history(initial_observations)
        bo_iterations = max(
            0,
            self.total_budget - len(self.campaign.observations.get_values(as_array=True)),
        )
        self._emit_progress(
            "run_start",
            dataset=self.env.dataset,
            planner_name=self.config.planner_name,
            seed=self.config.seed,
            total_budget=self.total_budget,
            init_budget=self.init_budget,
            initial_observations=len(initial_observations),
            bo_iterations=bo_iterations,
        )
        while len(self.campaign.observations.get_values(as_array=True)) < self.total_budget:
            iteration += 1
            if hasattr(self.decision_engine, "reset_skill_trace"):
                self.decision_engine.reset_skill_trace()
            history = self.memory.get_history()
            full_observation_history = self._observation_history_with_initial(history)
            best_observation = self._best_observation()
            observations = len(self.campaign.observations.get_values(as_array=True))
            self.online_decision_state.refresh_from_history(
                bootstrap_history=self._initial_state_history,
                history=history,
                best_observation=best_observation,
                current_best=self._current_best(),
                iteration=iteration,
                observations=observations,
                total_budget=self.total_budget,
                search_space=self.env.param_space,
                search_space_meta=self.dataset_meta,
                goal=self.env.goal,
            )
            self.working_memory.replace_state(self.online_decision_state.working_memory_summary())
            working_memory_summary = self.working_memory.summarize()
            self._emit_progress(
                "iteration_start",
                iteration=iteration,
                bo_iterations=bo_iterations,
                observations=observations,
                total_budget=self.total_budget,
                best_result=self._current_best(),
                elapsed_sec=round(time.time() - start_time, 1),
            )
            decision_context = self._attach_planner_policy_context(
                self.online_decision_state.build_decision_context(
                    search_space_meta=self.dataset_meta,
                )
            )
            trigger, trigger_reasons = self._should_trigger_sparse_intervention(
                decision_context=decision_context,
                iteration=iteration,
            )

            diagnosis: dict[str, Any] | None = None
            hypothesis_action: dict[str, Any] | None = None
            coverage_insight: dict[str, Any] | None = None
            intervention_plan: dict[str, Any] | None = None
            controller_plan: dict[str, Any] | None = None
            knowledge_query_rule = ""
            knowledge_query_document = ""
            knowledge_filters: dict[str, Any] = {}
            knowledge_units: list[dict[str, Any]] = []
            knowledge_source_types: list[str] = []
            knowledge_route = "auto"
            knowledge_retrieved_by_source: dict[str, int] = {
                "rule": 0,
                "document": 0,
                "memory": 0,
            }
            knowledge_injected_by_source: dict[str, int] = {}
            knowledge_retrieved_units_by_source: dict[str, list[dict[str, Any]]] = {}
            reviewed_knowledge_trace: dict[str, Any] = {}
            enable_document_retrieval = False
            retrieval_bundle: dict[str, Any] = {}

            sparse_cycle = trigger or (iteration % 5 == 0)
            if (
                sparse_cycle
                and self.knowledge_provider is not None
                and self.knowledge_query_builder is not None
            ):
                try:
                    retrieval_request = self.knowledge_query_builder.build(
                        dataset=self.env.dataset,
                        best_observation=best_observation,
                        underexplored_dimensions=decision_context.get(
                            "underexplored_dimensions",
                            [],
                        ),
                        trigger_reasons=trigger_reasons,
                        working_memory_summary=working_memory_summary,
                        iteration=iteration,
                    )
                    knowledge_query_rule = str(retrieval_request.get("rule_query", ""))
                    knowledge_query_document = str(retrieval_request.get("document_query", ""))
                    knowledge_filters = dict(retrieval_request.get("filters", {}))
                    knowledge_route = str(retrieval_request.get("route", "auto"))
                    enable_document_retrieval = bool(
                        retrieval_request.get("enable_document_retrieval", True)
                    )
                    retrieval_bundle = self.knowledge_provider.get_context_snippets(
                        rule_query=knowledge_query_rule,
                        document_query=knowledge_query_document,
                        filters=knowledge_filters,
                        top_k=self.config.knowledge_top_k,
                        route=knowledge_route,
                        enable_document_retrieval=enable_document_retrieval,
                    )
                    knowledge_units = list(retrieval_bundle.get("injected_units", []))
                    knowledge_retrieved_by_source = dict(
                        retrieval_bundle.get("retrieved_by_source", knowledge_retrieved_by_source)
                    )
                    knowledge_injected_by_source = dict(
                        retrieval_bundle.get("injected_by_source", {})
                    )
                    knowledge_retrieved_units_by_source = dict(
                        retrieval_bundle.get("retrieved_units_by_source", {})
                    )
                    knowledge_source_types = sorted(
                        {
                            str(item.get("source_type", "unknown"))
                            for item in knowledge_units
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Knowledge retrieval failed, continue without snippets: %s", exc)
                    knowledge_query_rule = ""
                    knowledge_query_document = ""
                    knowledge_filters = {}
                    knowledge_units = []
                    knowledge_source_types = []
                    knowledge_route = "auto"
                    knowledge_retrieved_by_source = {"rule": 0, "document": 0, "memory": 0}
                    knowledge_injected_by_source = {}
                    knowledge_retrieved_units_by_source = {}
                    enable_document_retrieval = False
                    retrieval_bundle = {}

            self.online_decision_state.refresh_from_history(
                bootstrap_history=self._initial_state_history,
                history=history,
                best_observation=best_observation,
                current_best=self._current_best(),
                iteration=iteration,
                observations=observations,
                total_budget=self.total_budget,
                search_space=self.env.param_space,
                search_space_meta=self.dataset_meta,
                goal=self.env.goal,
                knowledge_units=knowledge_units,
                knowledge_query=knowledge_query_rule,
                knowledge_meta={
                    "filters": knowledge_filters,
                    "source_types": knowledge_source_types,
                    "route": knowledge_route,
                    "rule_query": knowledge_query_rule,
                    "document_query": knowledge_query_document,
                    "retrieved_by_source": knowledge_retrieved_by_source,
                    "injected_by_source": knowledge_injected_by_source,
                    "enable_document_retrieval": enable_document_retrieval,
                },
            )
            self.working_memory.replace_state(self.online_decision_state.working_memory_summary())
            working_memory_summary = self.working_memory.summarize()
            decision_context = self._attach_planner_policy_context(
                self.online_decision_state.build_decision_context(
                    search_space_meta=self.dataset_meta,
                )
            )
            if sparse_cycle:
                self._emit_progress(
                    "phase_start",
                    iteration=iteration,
                    phase="llm_diagnostics",
                    elapsed_sec=round(time.time() - start_time, 1),
                )
                diagnosis_context, diagnosis_reviewed_trace = self._augment_context_with_reviewed_knowledge(
                    node_name="stagnation_diagnosis",
                    decision_context=decision_context,
                )
                reviewed_knowledge_trace["stagnation_diagnosis"] = diagnosis_reviewed_trace
                diagnosis = self.decision_engine.diagnose_stagnation(
                    decision_context=diagnosis_context,
                    history_tail=decision_context.get("recent_history_tail", []),
                )
                hypothesis_context, hypothesis_reviewed_trace = self._augment_context_with_reviewed_knowledge(
                    node_name="hypothesis_action",
                    decision_context=decision_context,
                )
                reviewed_knowledge_trace["hypothesis_action"] = hypothesis_reviewed_trace
                hypothesis_action = self.decision_engine.generate_hypothesis(
                    decision_context=hypothesis_context,
                    reaction_context=self.reaction_context,
                    search_space=self.env.param_space,
                )
                coverage_insight = self.decision_engine.analyze_coverage(
                    decision_context=decision_context,
                    search_space=self.env.param_space,
                )

            # Phase B: update LLM search constraints when scheduled
            llm_constraint_update_record: dict[str, Any] | None = None
            if self.config.enable_llm_search_constraint and self.env.is_finite_pool:
                bo_start_iter = self.init_budget + 1
                stagnation_detected = bool(
                    diagnosis and diagnosis.get("is_stagnating")
                )
                since_last = iteration - self._llm_constraint_last_updated_at
                should_update_constraint = (
                    self.decision_engine is not None
                    and (
                        iteration == bo_start_iter
                        or since_last >= self.config.constraint_update_freq
                        or (stagnation_detected and since_last >= 2)
                    )
                )
                if should_update_constraint:
                    self._emit_progress(
                        "phase_start",
                        iteration=iteration,
                        phase="llm_search_constraint",
                        elapsed_sec=round(time.time() - start_time, 1),
                    )
                    llm_constraint_update_record = self._update_llm_constraints(
                        iteration=iteration,
                        decision_context=decision_context,
                        history_tail=decision_context.get("recent_history_tail", []),
                    )

            controller_trigger_reasons = self._controller_trigger_reasons(
                decision_context=decision_context,
                trigger_reasons=trigger_reasons,
            )
            controller_should_consider = (
                self.config.enable_auto_subspace_trigger
                and len(controller_trigger_reasons) > 0
                and len(self.campaign.observations.get_values(as_array=True))
                >= getattr(self.bo_tool, "_num_init_design", self.init_budget)
            )
            if controller_should_consider and (
                diagnosis is None or hypothesis_action is None or coverage_insight is None
            ):
                self._emit_progress(
                    "phase_start",
                    iteration=iteration,
                    phase="llm_controller_context",
                    elapsed_sec=round(time.time() - start_time, 1),
                )
                diagnosis_context, diagnosis_reviewed_trace = self._augment_context_with_reviewed_knowledge(
                    node_name="stagnation_diagnosis",
                    decision_context=decision_context,
                )
                reviewed_knowledge_trace["stagnation_diagnosis"] = diagnosis_reviewed_trace
                diagnosis = self.decision_engine.diagnose_stagnation(
                    decision_context=diagnosis_context,
                    history_tail=decision_context.get("recent_history_tail", []),
                )
                hypothesis_context, hypothesis_reviewed_trace = self._augment_context_with_reviewed_knowledge(
                    node_name="hypothesis_action",
                    decision_context=decision_context,
                )
                reviewed_knowledge_trace["hypothesis_action"] = hypothesis_reviewed_trace
                hypothesis_action = self.decision_engine.generate_hypothesis(
                    decision_context=hypothesis_context,
                    reaction_context=self.reaction_context,
                    search_space=self.env.param_space,
                )
                coverage_insight = self.decision_engine.analyze_coverage(
                    decision_context=decision_context,
                    search_space=self.env.param_space,
                )
            if (
                controller_should_consider
                and len(controller_trigger_reasons) > 0
                and diagnosis is not None
                and hypothesis_action is not None
                and coverage_insight is not None
            ):
                self._emit_progress(
                    "phase_start",
                    iteration=iteration,
                    phase="llm_controller_plan",
                    elapsed_sec=round(time.time() - start_time, 1),
                )
                controller_plan = self.decision_engine.choose_controller_plan(
                    decision_context=decision_context,
                    diagnosis=diagnosis,
                    hypothesis_action=hypothesis_action,
                    coverage_insight=coverage_insight,
                    controller_trigger_reasons=controller_trigger_reasons,
                    search_space=self.env.param_space,
                    enable_action_package_v2=bool(self.config.enable_action_package_v2),
                    enable_action_package_v06=bool(self.config.enable_action_package_v06),
                )
            else:
                controller_plan = {
                    "intervention_type": "bo_direct",
                    "action_schema_version": (
                        "v0.6"
                        if self.config.enable_action_package_v06
                        else ("v2" if self.config.enable_action_package_v2 else "compat_v1")
                    ),
                    "requested_execution_action": "direct_bo_pick",
                    "admissible_execution_actions": ["direct_bo_pick"],
                    "preferred_execution_action": "direct_bo_pick",
                    "fallback_reason": None,
                    "intent": "exploit",
                    "shortlist_policy": "plain",
                    "repeat_policy": "allow",
                    "selection_policy": "bo_top1",
                    "verification_policy": "normal",
                    "focus_policy": "full_space",
                    "use_subspace": False,
                    "focus_variables": [],
                    "window_rounds": 0,
                    "reasoning": "No controller intervention triggered.",
                    "action_package": {
                        **self._default_action_package(
                            schema_version=(
                                "v0.6" if self.config.enable_action_package_v06 else (
                                    "v2" if self.config.enable_action_package_v2 else "compat_v1"
                                )
                            )
                        ),
                        "schema_version": (
                            "v0.6" if self.config.enable_action_package_v06 else (
                                "v2" if self.config.enable_action_package_v2 else "compat_v1"
                            )
                        ),
                    },
                }

            intervention_plan = controller_plan
            action_package = self._extract_action_package(controller_plan)
            v06_enabled = bool(self.config.enable_action_package_v06)
            focus = list(controller_plan.get("focus_variables", []))
            controller_mode = str(controller_plan.get("intervention_type", "bo_direct"))
            execution_action_requested = (
                self._requested_execution_action(action_package)
                if v06_enabled
                else self._execution_action_from_legacy_mode(controller_mode)
            )
            execution_action_current = execution_action_requested
            execution_action_fallback_reason: str | None = None
            diagnosis_type = str((diagnosis or {}).get("stagnation_type", "")).strip().lower()
            diagnosis_intervention = str(
                (diagnosis or {}).get("recommended_intervention", "")
            ).strip().lower()
            no_improvement_rounds = int(decision_context.get("no_improvement_rounds", 0) or 0)
            local_lock_actionable = (
                self.env.is_finite_pool
                and no_improvement_rounds >= 4
                and self._local_lock_trigger_active(decision_context)
            )
            trusted_planner_mode_active = self._trusted_planner_mode_active()
            if trusted_planner_mode_active:
                trusted_planner_override_allowed, trusted_planner_block_reason = self._trusted_planner_override_gate(
                    decision_context=decision_context,
                    diagnosis=diagnosis or {},
                    trigger_reasons=controller_trigger_reasons,
                )
            else:
                trusted_planner_override_allowed = True
                trusted_planner_block_reason = None

            if v06_enabled:
                if execution_action_current == "focused_shortlist_alt_pick" and not self.config.allow_focus_then_rerank:
                    execution_action_current = "shortlist_alt_pick"
                    execution_action_fallback_reason = "focused_execution_disabled_by_config"
                shape_only_min_iteration = int(self.config.shape_only_min_iteration or 0)
                if (
                    shape_only_min_iteration > 0
                    and execution_action_current in {"shape_only_bo_pick", "shape_then_probe_topk"}
                    and int(iteration) < shape_only_min_iteration
                ):
                    execution_action_current = "direct_bo_pick"
                    execution_action_fallback_reason = (
                        f"shape_action_delayed_until_iteration_{shape_only_min_iteration}"
                    )
                use_subspace = (
                    execution_action_current == "focused_shortlist_alt_pick"
                    and 0 < len(focus) < len(self.env.param_space)
                )
                if execution_action_current == "focused_shortlist_alt_pick" and not use_subspace:
                    execution_action_current = "shortlist_alt_pick"
                    execution_action_fallback_reason = "focused_execution_missing_valid_focus_variables"
                subspace_window_remaining = 1 if use_subspace else 0
                subspace_focus_variables = list(focus) if use_subspace else []
                controller_soft_rerank_authorized = False
                trusted_pre_gate_forced_direct = False
                controller_mode = execution_action_current
            else:
                if controller_mode == "bo_focus_then_rerank" and not self.config.allow_focus_then_rerank:
                    controller_mode = "bo_rerank_topk"
                    controller_plan = {
                        **controller_plan,
                        "intervention_type": "bo_rerank_topk",
                        "use_subspace": False,
                        "focus_variables": [],
                        "window_rounds": 0,
                        "reasoning": (
                            f"{controller_plan.get('reasoning', '')} [v1.2 config: focus_then_rerank disabled for this run, use full-space rerank.]"
                        ).strip(),
                    }
                if (
                    controller_mode == "bo_rerank_topk"
                    and diagnosis_type in {"insufficient_data", "insufficient_data_early_stage", "init_artifact"}
                    and diagnosis_intervention in {"keep_full_space", "keep_full_space_with_caution", "observe_more"}
                    and no_improvement_rounds <= 3
                ):
                    controller_mode = "bo_direct"
                    controller_plan = {
                        **controller_plan,
                        "intervention_type": "bo_direct",
                        "use_subspace": False,
                        "focus_variables": [],
                        "window_rounds": 0,
                        "reasoning": (
                            f"{controller_plan.get('reasoning', '')} [controller guardrail: diagnosis remains evidence-limited, so downgrade shortlist rerank to bo_direct.]"
                        ).strip(),
                    }
                if (
                    controller_mode == "bo_direct"
                    and self.config.controller_allow_rerank_without_space_narrowing
                    and self.config.finite_pool_sparse_coverage_not_veto_rerank
                    and local_lock_actionable
                    and diagnosis_intervention in {
                        "keep_full_space",
                        "keep_full_space_with_caution",
                        "observe_more",
                        "",
                    }
                ):
                    controller_mode = "bo_rerank_topk"
                    controller_plan = {
                        **controller_plan,
                        "intervention_type": "bo_rerank_topk",
                        "use_subspace": False,
                        "focus_variables": [],
                        "window_rounds": 0,
                        "reasoning": (
                            f"{controller_plan.get('reasoning', '')} "
                            "[controller auto-nudge: sustained finite-pool local lock is actionable even without space narrowing, so upgrade bo_direct to full-space shortlist rerank.]"
                        ).strip(),
                    }
                controller_soft_rerank_authorized = self._controller_soft_rerank_authorized(
                    controller_mode=controller_mode,
                    diagnosis_intervention=diagnosis_intervention,
                    trigger_reasons=controller_trigger_reasons,
                )
                trusted_pre_gate_forced_direct = False
                if (
                    controller_mode == "bo_rerank_topk"
                    and trusted_planner_mode_active
                    and self.config.enable_trusted_planner_pre_rerank_skip
                    and not trusted_planner_override_allowed
                    and not controller_soft_rerank_authorized
                    and diagnosis_type == "borderline_stagnation"
                ):
                    controller_mode = "bo_direct"
                    trusted_pre_gate_forced_direct = True
                    controller_plan = {
                        **controller_plan,
                        "intervention_type": "bo_direct",
                        "use_subspace": False,
                        "focus_variables": [],
                        "window_rounds": 0,
                        "reasoning": (
                            f"{controller_plan.get('reasoning', '')} "
                            "[trusted planner pre-gate: keep BO top-1 and skip shortlist execution.]"
                        ).strip(),
                    }
                if controller_mode == "bo_focus_then_rerank" and previous_focus_improved is False:
                    controller_mode = "bo_rerank_topk"
                    controller_plan = {
                        **controller_plan,
                        "intervention_type": "bo_rerank_topk",
                        "use_subspace": False,
                        "focus_variables": [],
                        "window_rounds": 0,
                        "reasoning": (
                            f"{controller_plan.get('reasoning', '')} [focus exit: previous focused round did not improve best, return to full-space reevaluation.]"
                        ).strip(),
                    }
                if controller_mode == "bo_focus_then_rerank" and consecutive_focus_rounds >= 2:
                    controller_mode = "bo_rerank_topk"
                    controller_plan = {
                        **controller_plan,
                        "intervention_type": "bo_rerank_topk",
                        "use_subspace": False,
                        "focus_variables": [],
                        "window_rounds": 0,
                        "reasoning": (
                            f"{controller_plan.get('reasoning', '')} [focus exit: max consecutive focus rounds reached, return to full-space reevaluation.]"
                        ).strip(),
                    }
                focus = list(controller_plan.get("focus_variables", []))
                action_package = self._sync_action_package_with_controller_mode(
                    action_package,
                    controller_mode=controller_mode,
                    focus_variables=focus,
                    window_rounds=int(controller_plan.get("window_rounds", 0) or 0),
                    reasoning=str(controller_plan.get("reasoning", "")),
                )
                controller_plan = {
                    **controller_plan,
                    "action_package": action_package,
                }
                execution_action_current = self._execution_action_from_legacy_mode(controller_mode)
                use_subspace = (
                    controller_mode == "bo_focus_then_rerank"
                    and 0 < len(focus) < len(self.env.param_space)
                )
                subspace_window_remaining = 1 if use_subspace else 0
                subspace_focus_variables = list(focus) if use_subspace else []
            if use_subspace and not self.env.is_finite_pool:
                stage = "targeted_subspace_refinement"
                active_variables = list(subspace_focus_variables)
                sub_campaign = build_subspace_campaign(
                    full_campaign=self.campaign,
                    original_space=self.env.param_space,
                    active_variables=active_variables,
                )
                suggestion_campaign = sub_campaign
                suggestion_space = sub_campaign.param_space
            else:
                stage = (
                    "targeted_subspace_refinement"
                    if use_subspace and self.env.is_finite_pool
                    else "full_space_search"
                )
                active_variables = (
                    list(subspace_focus_variables)
                    if use_subspace
                    else [param.name for param in self.env.param_space]
                )
                suggestion_campaign = self._full_campaign_for_space()
                suggestion_space = self.env.param_space

            selected_candidate: dict[str, Any] | None = None
            semantic_assessment: dict[str, Any] | None = None
            verification_pass: dict[str, Any] | None = None
            completion_action: dict[str, Any] | None = None
            bo_candidate_origin = "full_space_direct"
            if not self.env.is_finite_pool and execution_action_current != "direct_bo_pick":
                if v06_enabled:
                    execution_action_current = "direct_bo_pick"
                    controller_mode = execution_action_current
                    execution_action_fallback_reason = (
                        execution_action_fallback_reason
                        or "non_finite_pool_does_not_support_shortlist_actions"
                    )
                else:
                    controller_mode = "bo_direct"
                    execution_action_current = "direct_bo_pick"
                    controller_plan = {
                        **controller_plan,
                        "intervention_type": "bo_direct",
                        "use_subspace": False,
                        "focus_variables": [],
                        "window_rounds": 0,
                        "reasoning": "Non-finite-pool path falls back to bo_direct in controller v1.",
                    }
                    action_package = self._sync_action_package_with_controller_mode(
                        action_package,
                        controller_mode=controller_mode,
                        focus_variables=[],
                        window_rounds=0,
                        reasoning=str(controller_plan.get("reasoning", "")),
                    )
                    controller_plan = {
                        **controller_plan,
                        "action_package": action_package,
                    }
            known_constraints, subspace_filter_summary, known_allowed_keys = self._known_constraint_bundle(
                use_subspace=use_subspace,
                active_variables=active_variables,
                best_observation=best_observation,
                iteration=iteration,
            )
            effective_constraint_summary = dict(subspace_filter_summary)
            shortlist_candidates: list[dict[str, Any]] = []
            rerank_action: dict[str, Any] | None = None
            rerank_policy: dict[str, Any] = {
                "planner_name": str(self.config.planner_name).strip().lower(),
                "prompt_style": "default",
                "trusted_pre_gate_skip": False,
                "late_stage_active": False,
                "current_best": self._current_best(),
                "current_best_percentile": self._current_best_percentile(self._current_best()),
                "strong_incumbent_present": False,
                "near_best_margin": 0.03,
            }
            rerank_skipped_by_planner_policy = bool(trusted_pre_gate_forced_direct)
            bo_top1_candidate: dict[str, Any] | None = None
            selected_candidate_rank_in_shortlist: int | None = None
            selected_differs_from_bo_top1 = False
            focus_filter_applied = False
            focus_fallback_reason: str | None = None
            shortlist_scaffold_diversity = 0
            shortlist_source_mix: list[str] = []
            shortlist_target_diversity_met = False
            shortlist_available_scaffold_diversity = 0
            diversity_injection_count = 0
            selected_candidate_source = "bo_top_ranked"
            selected_candidate_pool_source = "main_pool"
            shortlist_pool_strategy = "main_pool_only"
            main_pool_candidates: list[dict[str, Any]] = []
            diversity_pool_candidates: list[dict[str, Any]] = []
            diversity_pool_used = False
            diversity_pool_unique_scaffolds = 0
            pool_limited_diversity = False
            structural_meta: dict[str, Any] = {
                "bo_top1_scaffold_key": [],
                "dominant_scaffold_key": [],
                "structural_shift_candidate_count": 0,
                "dominant_shift_candidate_count": 0,
                "diversity_pool_quality": "not_used",
            }
            shortlist_shaping_trace: dict[str, Any] = {
                "enabled": False,
                "reason": "not_applicable",
                "summary": "Shortlist shaping not used.",
                "candidate_traces": [],
            }
            resuggest_trace: dict[str, Any] = {
                "enabled": False,
                "requested_execution_action": execution_action_current,
                "mask_applied": False,
                "fallback_reason": "not_requested",
                "pool_size_before": None,
                "pool_size_after": None,
                "excluded_candidate_count": 0,
                "mask_basis": {},
            }
            state_router_guidance: dict[str, Any] | None = None
            candidate_contrastive_evidence: dict[str, Any] = {}
            shortlist_value_audit_payload: dict[str, Any] = {}
            shaped_shortlist_candidates: list[dict[str, Any]] = []
            post_resuggest_selection_mode = "not_applicable"
            planner_action_policy = self._planner_action_policy()
            selection_authority: dict[str, Any] = {
                "planner_action_policy_name": planner_action_policy.get("planner_policy_name"),
                "selection_authority_level": planner_action_policy.get(
                    "default_selection_authority_level",
                    "planner_only",
                ),
                "authority_source": "planner_policy_default",
                "evidence_sufficiency_passed": False,
                "evidence_failure_reasons": ["no_shortlist_probe_evaluation"],
                "shortlist_probe_authorized": False,
            }
            candidate_direction_review: dict[str, Any] = {
                "enabled": str(self.config.candidate_direction_review_mode or "off").strip().lower() != "off",
                "mode": str(self.config.candidate_direction_review_mode or "off").strip().lower(),
                "eligible": False,
                "applied": False,
                "reason": "no_shortlist_review_evaluation",
            }
            candidate_probe_trace: dict[str, Any] = {
                "enabled": bool(self.config.candidate_probe_enabled),
                "eligible": False,
                "applied": False,
                "reason": "no_candidate_probe_evaluation",
            }
            candidate_direction_review_applied = False
            planner_preferred_candidate_index: int | None = None
            planner_preferred_after_shaping = False
            planner_preferred_after_resuggest = False
            final_selection_source = "planner_direct"
            guardrail_veto_applied = False
            override_guardrail: dict[str, Any] = {
                "enabled": self.config.enable_override_guardrail,
                "passed": True,
                "action": "not_applicable",
                "reason": "no_shortlist_override",
                "score_margin": None,
                "structural_shift": False,
                "llm_requested_override": False,
                "trusted_planner_mode_active": trusted_planner_mode_active,
                "trusted_planner_override_allowed": trusted_planner_override_allowed,
                "trusted_planner_block_reason": trusted_planner_block_reason,
                "blocked_by_trusted_planner_policy": False,
                "late_stage_active": False,
                "strong_incumbent_present": False,
                "current_best_percentile": rerank_policy.get("current_best_percentile"),
                "blocked_by_late_stage_incumbent_protection": False,
            }
            llm_selected_candidate: dict[str, Any] | None = None
            llm_selected_candidate_rank_in_shortlist: int | None = None

            if execution_action_current == "direct_bo_pick":
                self._emit_progress(
                    "phase_start",
                    iteration=iteration,
                    phase="planner_suggest",
                    elapsed_sec=round(time.time() - start_time, 1),
                )
                samples = self._call_planner(
                    "planner_suggest",
                    self.bo_tool.suggest,
                    observations=suggestion_campaign.observations,
                    subspace=suggestion_space,
                    known_constraints=known_constraints,
                    known_constraints_signature=subspace_filter_summary.get("constraint_signature"),
                )
                partial_candidate = self._sample_to_dict(samples[0], suggestion_space)
                if use_subspace and not self.env.is_finite_pool:
                    completion_action = self.decision_engine.complete_candidate_action(
                        partial_candidate=partial_candidate,
                        decision_action={
                            "stage": stage,
                            "active_variables": active_variables,
                            "decision_mode": "mixed",
                        },
                        decision_context=decision_context,
                        search_space=self.env.param_space,
                    )
                    selected_candidate = complete_candidate(
                        partial_candidate=partial_candidate,
                        completion_overrides=completion_action.get("full_candidate", {}),
                        fallback_defaults=self._completion_fallback_defaults(),
                        param_space=self.env.param_space,
                    )
                    bo_candidate_origin = "subspace_plus_completion"
                else:
                    selected_candidate = partial_candidate
                    if self.env.is_finite_pool:
                        bo_candidate_origin = (
                            "finite_pool_subspace_filter"
                            if use_subspace
                            else "finite_pool_full_pool"
                        )
                bo_top1_candidate = dict(selected_candidate)
                selected_candidate_rank_in_shortlist = 0
                llm_selected_candidate = dict(selected_candidate)
                llm_selected_candidate_rank_in_shortlist = 0
                if self.env.is_finite_pool:
                    candidate_probe_trace = self._candidate_probe_state(
                        execution_action=execution_action_current,
                        decision_context=decision_context,
                        iteration=iteration,
                    )
                    candidate_probe_direction = self._candidate_probe_direction_from_action_package(
                        action_package
                    )
                    if candidate_probe_direction:
                        candidate_probe_trace["direction"] = candidate_probe_direction
                    if bool(candidate_probe_trace.get("eligible", False)):
                        direct_shortlist = [
                            {
                                "candidate_index": 0,
                                "bo_rank": 1,
                                "candidate": dict(selected_candidate),
                                "pool_source": "main_pool",
                                "shortlist_source": "bo_top_ranked",
                                "is_main_bo_top1": True,
                            }
                        ]
                        shortlist_candidates, candidate_probe_trace = self._apply_candidate_probe(
                            history=full_observation_history,
                            shortlist_candidates=direct_shortlist,
                            allowed_keys=set(known_allowed_keys or set()),
                            candidate_probe=candidate_probe_trace,
                        )
                        candidate_probe_preferred_item = self._candidate_probe_preferred_item(
                            shortlist_candidates=shortlist_candidates,
                            candidate_probe_trace=candidate_probe_trace,
                            history=full_observation_history,
                        )
                        if candidate_probe_preferred_item is not None:
                            selected_candidate = dict(candidate_probe_preferred_item["candidate"])
                            llm_selected_candidate = dict(selected_candidate)
                            selected_candidate_rank_in_shortlist = next(
                                (
                                    idx
                                    for idx, item in enumerate(shortlist_candidates)
                                    if int(item.get("candidate_index", idx))
                                    == int(candidate_probe_preferred_item.get("candidate_index", -1))
                                ),
                                0,
                            )
                            llm_selected_candidate_rank_in_shortlist = selected_candidate_rank_in_shortlist
                            selected_differs_from_bo_top1 = True
                            selected_candidate_source = str(
                                candidate_probe_preferred_item.get("shortlist_source", "bo_top_ranked")
                            )
                            selected_candidate_pool_source = str(
                                candidate_probe_preferred_item.get("pool_source", "main_pool")
                            )
                            candidate_probe_selection_mode = str(
                                candidate_probe_trace.get("effective_selection_mode")
                                or self.config.candidate_probe_selection_mode
                                or "merge_only"
                            )
                            rerank_action = {
                                "selected_index": int(
                                    candidate_probe_preferred_item.get("candidate_index", 0) or 0
                                ),
                                "candidate_scores": [],
                                "reasoning": (
                                    "Direct BO pick was replaced by candidate_probe using "
                                    + candidate_probe_selection_mode
                                    + "."
                                ),
                                "prompt_style": "direct_candidate_probe",
                                "candidate_probe_selection_mode": candidate_probe_selection_mode,
                                "selection_status": f"candidate_probe_{candidate_probe_selection_mode}",
                            }
            else:
                shortlist_size = min(
                    max(1, int(self.config.shortlist_size)),
                    max(
                        1,
                        int(
                            subspace_filter_summary.get("pool_size")
                            or self.config.shortlist_size
                        ),
                    ),
                )
                shortlist_main_pool_size = min(
                    max(shortlist_size, int(self.config.shortlist_main_pool_size)),
                    max(
                        shortlist_size,
                        int(
                            subspace_filter_summary.get("pool_size")
                            or self.config.shortlist_main_pool_size
                        ),
                    ),
                )
                shortlist_constraints = known_constraints
                shortlist_allowed_keys = set(known_allowed_keys or set())
                shortlist_constraint_signature = subspace_filter_summary.get("constraint_signature")
                shortlist_mode = execution_action_current if v06_enabled else controller_mode
                if v06_enabled and execution_action_current in {
                    "mask_scaffold_corridor_resuggest",
                    "mask_dominant_resuggest",
                    "mask_low_repeat_resuggest",
                }:
                    (
                        shortlist_constraints,
                        effective_constraint_summary,
                        shortlist_allowed_keys,
                        resuggest_trace,
                    ) = self._apply_pre_shortlist_resuggest(
                        execution_action=execution_action_current,
                        decision_context=decision_context,
                        history=history,
                        allowed_keys=shortlist_allowed_keys,
                        effective_constraint_summary=effective_constraint_summary,
                    )
                    shortlist_constraint_signature = effective_constraint_summary.get("constraint_signature")
                    if not shortlist_constraints:
                        shortlist_constraints = known_constraints
                        shortlist_allowed_keys = set(known_allowed_keys or set())
                if (
                    shortlist_mode in {"bo_focus_then_rerank", "focused_shortlist_alt_pick"}
                    and (
                        not isinstance(subspace_filter_summary.get("pool_size"), (int, float))
                        or int(subspace_filter_summary.get("pool_size") or 0) < shortlist_size + 1
                    )
                ):
                    shortlist_mode = "shortlist_alt_pick" if v06_enabled else "bo_rerank_topk"
                    focus_fallback_reason = "focused_pool_too_small"
                    if v06_enabled:
                        execution_action_current = shortlist_mode
                        controller_mode = shortlist_mode
                        execution_action_fallback_reason = (
                            execution_action_fallback_reason or focus_fallback_reason
                        )
                    shortlist_constraints, fallback_summary, shortlist_allowed_keys = self._known_constraint_bundle(
                        use_subspace=False,
                        active_variables=[param.name for param in self.env.param_space],
                        best_observation=best_observation,
                        iteration=iteration,
                    )
                    effective_constraint_summary = dict(fallback_summary)
                    shortlist_constraint_signature = fallback_summary.get("constraint_signature")
                self._emit_progress(
                    "phase_start",
                    iteration=iteration,
                    phase="planner_shortlist_main",
                    elapsed_sec=round(time.time() - start_time, 1),
                )
                samples = self._call_planner(
                    "planner_shortlist_main",
                    self.bo_tool.suggest_shortlist,
                    observations=suggestion_campaign.observations,
                    subspace=suggestion_space,
                    shortlist_size=shortlist_main_pool_size,
                    known_constraints=shortlist_constraints,
                    known_constraints_signature=shortlist_constraint_signature,
                )
                raw_shortlist: list[dict[str, Any]] = []
                for idx, sample in enumerate(samples):
                    raw_shortlist.append(
                        {
                            "candidate_index": idx,
                            "bo_rank": idx + 1,
                            "candidate": self._sample_to_dict(sample, suggestion_space),
                        }
                    )
                main_pool_candidates = self._annotate_shortlist_candidates(
                    raw_shortlist,
                    history,
                    pool_source="main_pool",
                    shortlist_source="bo_top_ranked",
                )
                use_diversity_pool = self._should_use_diversity_pool(
                    decision_context=decision_context,
                    controller_mode=execution_action_current if v06_enabled else controller_mode,
                )
                if (
                    trusted_planner_mode_active
                    and self.config.trusted_planner_disable_early_diversity
                    and not trusted_planner_override_allowed
                    and not (
                        self.config.controller_allow_rerank_without_space_narrowing
                        and "local_lock_stall" in controller_trigger_reasons
                    )
                ):
                    use_diversity_pool = False
                if execution_action_current in {
                    "mask_scaffold_corridor_resuggest",
                    "mask_dominant_resuggest",
                    "mask_low_repeat_resuggest",
                }:
                    use_diversity_pool = False
                if use_diversity_pool:
                    dominant_constraint = self._dominant_scaffold_constraint(
                        decision_context.get("dominant_scaffold")
                    )
                    if dominant_constraint is not None:
                        diversity_constraints = list(shortlist_constraints) + [dominant_constraint]
                        diversity_signature = json.dumps(
                            [
                                str(shortlist_constraint_signature),
                                "avoid_dominant_scaffold",
                                decision_context.get("dominant_scaffold", {}),
                            ],
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        diversity_pool_size = min(
                            max(1, int(self.config.shortlist_diversity_pool_size)),
                            max(
                                1,
                                int(
                                    subspace_filter_summary.get("pool_size")
                                    or self.config.shortlist_diversity_pool_size
                                ),
                            ),
                        )
                        self._emit_progress(
                            "phase_start",
                            iteration=iteration,
                            phase="planner_shortlist_diversity",
                            elapsed_sec=round(time.time() - start_time, 1),
                        )
                        diversity_samples = self._call_planner(
                            "planner_shortlist_diversity",
                            self.bo_tool.suggest_shortlist,
                            observations=suggestion_campaign.observations,
                            subspace=suggestion_space,
                            shortlist_size=diversity_pool_size,
                            known_constraints=diversity_constraints,
                            known_constraints_signature=diversity_signature,
                        )
                        diversity_raw = []
                        for idx, sample in enumerate(diversity_samples):
                            diversity_raw.append(
                                {
                                    "candidate_index": idx,
                                    "bo_rank": idx + 1,
                                    "candidate": self._sample_to_dict(sample, suggestion_space),
                                }
                            )
                        diversity_pool_candidates = self._annotate_shortlist_candidates(
                            diversity_raw,
                            history,
                            pool_source="diversity_pool",
                            shortlist_source="diversity_injected",
                        )
                shortlist_candidates, shortlist_meta = self._build_dual_pool_shortlist(
                    main_pool=main_pool_candidates,
                    diversity_pool=diversity_pool_candidates,
                )
                shortlist_source_mix = list(shortlist_meta.get("shortlist_source_mix", []))
                shortlist_target_diversity_met = bool(
                    shortlist_meta.get("shortlist_target_diversity_met", False)
                )
                shortlist_available_scaffold_diversity = int(
                    shortlist_meta.get("shortlist_available_scaffold_diversity", 0)
                )
                diversity_injection_count = int(shortlist_meta.get("diversity_injection_count", 0))
                shortlist_pool_strategy = str(shortlist_meta.get("shortlist_pool_strategy", "main_pool_only"))
                pool_limited_diversity = bool(shortlist_meta.get("pool_limited_diversity", False))
                diversity_pool_used = bool(shortlist_meta.get("diversity_pool_used", False))
                diversity_pool_unique_scaffolds = int(
                    shortlist_meta.get("diversity_pool_unique_scaffolds", 0)
                )
                shortlist_scaffold_diversity = self._shortlist_scaffold_diversity(shortlist_candidates)
                if shortlist_mode in {"bo_focus_then_rerank", "focused_shortlist_alt_pick"} and shortlist_scaffold_diversity < 2:
                    shortlist_mode = "shortlist_alt_pick" if v06_enabled else "bo_rerank_topk"
                    focus_fallback_reason = "focused_shortlist_low_diversity"
                    if v06_enabled:
                        execution_action_current = shortlist_mode
                        controller_mode = shortlist_mode
                        execution_action_fallback_reason = (
                            execution_action_fallback_reason or focus_fallback_reason
                        )
                    shortlist_candidates = []
                    shortlist_constraints, fallback_summary, shortlist_allowed_keys = self._known_constraint_bundle(
                        use_subspace=False,
                        active_variables=[param.name for param in self.env.param_space],
                        best_observation=best_observation,
                        iteration=iteration,
                    )
                    effective_constraint_summary = dict(fallback_summary)
                    shortlist_constraint_signature = fallback_summary.get("constraint_signature")
                    self._emit_progress(
                        "phase_start",
                        iteration=iteration,
                        phase="planner_shortlist_fallback",
                        elapsed_sec=round(time.time() - start_time, 1),
                    )
                    samples = self._call_planner(
                        "planner_shortlist_fallback",
                        self.bo_tool.suggest_shortlist,
                        observations=suggestion_campaign.observations,
                        subspace=suggestion_space,
                        shortlist_size=shortlist_main_pool_size,
                        known_constraints=shortlist_constraints,
                        known_constraints_signature=shortlist_constraint_signature,
                    )
                    raw_shortlist = []
                    for idx, sample in enumerate(samples):
                        raw_shortlist.append(
                            {
                                "candidate_index": idx,
                                "bo_rank": idx + 1,
                                "candidate": self._sample_to_dict(sample, suggestion_space),
                            }
                        )
                    main_pool_candidates = self._annotate_shortlist_candidates(
                        raw_shortlist,
                        history,
                        pool_source="main_pool",
                        shortlist_source="bo_top_ranked",
                    )
                    diversity_pool_candidates = []
                    shortlist_candidates, shortlist_meta = self._build_dual_pool_shortlist(
                        main_pool=main_pool_candidates,
                        diversity_pool=diversity_pool_candidates,
                    )
                    shortlist_source_mix = list(shortlist_meta.get("shortlist_source_mix", []))
                    shortlist_target_diversity_met = bool(
                        shortlist_meta.get("shortlist_target_diversity_met", False)
                    )
                    shortlist_available_scaffold_diversity = int(
                        shortlist_meta.get("shortlist_available_scaffold_diversity", 0)
                    )
                    diversity_injection_count = int(
                        shortlist_meta.get("diversity_injection_count", 0)
                    )
                    shortlist_pool_strategy = str(
                        shortlist_meta.get("shortlist_pool_strategy", "main_pool_only")
                    )
                    pool_limited_diversity = bool(
                        shortlist_meta.get("pool_limited_diversity", False)
                    )
                    diversity_pool_used = bool(shortlist_meta.get("diversity_pool_used", False))
                    diversity_pool_unique_scaffolds = int(
                        shortlist_meta.get("diversity_pool_unique_scaffolds", 0)
                    )
                    shortlist_scaffold_diversity = self._shortlist_scaffold_diversity(shortlist_candidates)
                if shortlist_mode != (execution_action_current if v06_enabled else controller_mode):
                    if v06_enabled:
                        execution_action_current = shortlist_mode
                        controller_mode = shortlist_mode
                        use_subspace = False
                        focus_filter_applied = False
                        stage = "full_space_search"
                        active_variables = [param.name for param in self.env.param_space]
                    else:
                        controller_mode = shortlist_mode
                        use_subspace = False
                        focus_filter_applied = False
                        stage = "full_space_search"
                        active_variables = [param.name for param in self.env.param_space]
                        controller_plan = {
                            **controller_plan,
                            "intervention_type": controller_mode,
                            "use_subspace": False,
                            "focus_variables": [],
                            "window_rounds": 0,
                            "reasoning": (
                                f"{controller_plan.get('reasoning', '')} [focus fallback: {focus_fallback_reason}, use full-space rerank instead.]"
                            ).strip(),
                        }
                        action_package = self._sync_action_package_with_controller_mode(
                            action_package,
                            controller_mode=controller_mode,
                            focus_variables=[],
                            window_rounds=0,
                            reasoning=str(controller_plan.get("reasoning", "")),
                        )
                        controller_plan = {
                            **controller_plan,
                            "action_package": action_package,
                        }
                candidate_probe_trace = self._candidate_probe_state(
                    execution_action=execution_action_current,
                    decision_context=decision_context,
                    iteration=iteration,
                )
                candidate_probe_direction = self._candidate_probe_direction_from_action_package(
                    action_package
                )
                if candidate_probe_direction:
                    candidate_probe_trace["direction"] = candidate_probe_direction
                if self.env.is_finite_pool and shortlist_candidates and bool(
                    candidate_probe_trace.get("eligible", False)
                ):
                    shortlist_candidates, candidate_probe_trace = self._apply_candidate_probe(
                        history=full_observation_history,
                        shortlist_candidates=shortlist_candidates,
                        allowed_keys=shortlist_allowed_keys,
                        candidate_probe=candidate_probe_trace,
                    )
                    shortlist_scaffold_diversity = self._shortlist_scaffold_diversity(shortlist_candidates)
                    shortlist_source_mix = [
                        str(item.get("shortlist_source", "bo_top_ranked"))
                        for item in shortlist_candidates
                    ]
                    if bool(candidate_probe_trace.get("applied", False)):
                        shortlist_pool_strategy = (
                            f"{shortlist_pool_strategy}+candidate_probe"
                        )
                if v06_enabled and self.env.is_finite_pool:
                    state_router_guidance = self._build_state_router_guidance(
                        decision_context=decision_context,
                        shortlist_candidates=shortlist_candidates,
                        shortlist_scaffold_diversity=shortlist_scaffold_diversity,
                        diversity_pool_candidate_count=len(diversity_pool_candidates),
                    )
                    shape_only_failure_streak = bool(
                        str(decision_context.get("last_action_family", "") or "").strip() == "shape_only"
                        and not bool(decision_context.get("last_action_effective", False))
                        and int(
                            decision_context.get("consecutive_failed_action_family_rounds", 0) or 0
                        )
                        >= 1
                    )
                    guidance_failure_mode = str(
                        (state_router_guidance or {}).get("failure_mode", "") or ""
                    )
                    guidance_preferred_policy = str(
                        (state_router_guidance or {}).get("preferred_selection_policy", "") or ""
                    )
                    guidance_has_nonfallback_preferred = any(
                        int(idx)
                        != int((state_router_guidance or {}).get("fallback_candidate_index", 0) or 0)
                        for idx in list(
                            (state_router_guidance or {}).get("preferred_candidate_indices", [])
                            or []
                        )
                    )
                    runtime_promotion_allowed = planner_policy_allows_runtime_promotion(
                        self._planner_action_policy()
                    )
                    if (
                        runtime_promotion_allowed
                        and
                        execution_action_current == "shape_only_bo_pick"
                        and shape_only_failure_streak
                        and guidance_failure_mode == "shortlist_selection_failure"
                        and guidance_preferred_policy == "shape_probe_topk"
                        and guidance_has_nonfallback_preferred
                    ):
                        promotion_reason = guidance_failure_mode or "shortlist_selection_failure"
                        execution_action_requested = "shape_then_probe_topk"
                        execution_action_current = "shape_then_probe_topk"
                        controller_mode = execution_action_current
                        promoted_admissible = list(
                            dict.fromkeys(
                                [
                                    *list(
                                        (action_package or {}).get(
                                            "admissible_execution_actions",
                                            [],
                                        )
                                        or []
                                    ),
                                    "shape_then_probe_topk",
                                ]
                            )
                        )
                        action_package = {
                            **action_package,
                            "requested_execution_action": "shape_then_probe_topk",
                            "preferred_execution_action": "shape_then_probe_topk",
                            "admissible_execution_actions": promoted_admissible,
                            "selection_policy": "select_from_shaped_shortlist",
                            "reasoning": (
                                f"{action_package.get('reasoning', '')} "
                                f"[runtime promotion: {promotion_reason}_promoted_shape_probe_topk]"
                            ).strip(),
                        }
                        controller_plan = {
                            **(controller_plan or {}),
                            "requested_execution_action": "shape_then_probe_topk",
                            "selection_policy": "select_from_shaped_shortlist",
                            "action_package": action_package,
                        }
                    elif (
                        runtime_promotion_allowed
                        and
                        execution_action_current == "shape_only_bo_pick"
                        and shape_only_failure_streak
                        and guidance_failure_mode == "trajectory_shaping_failure"
                        and guidance_preferred_policy == "deeper_diversity_probe"
                        and guidance_has_nonfallback_preferred
                        and int(decision_context.get("no_improvement_rounds", 0) or 0) >= 8
                    ):
                        promotion_reason = guidance_failure_mode
                        execution_action_requested = "mask_scaffold_corridor_resuggest"
                        execution_action_current = "mask_scaffold_corridor_resuggest"
                        controller_mode = execution_action_current
                        promoted_admissible = list(
                            dict.fromkeys(
                                [
                                    *list(
                                        (action_package or {}).get(
                                            "admissible_execution_actions",
                                            [],
                                        )
                                        or []
                                    ),
                                    "mask_scaffold_corridor_resuggest",
                                ]
                            )
                        )
                        action_package = {
                            **action_package,
                            "requested_execution_action": "mask_scaffold_corridor_resuggest",
                            "preferred_execution_action": "mask_scaffold_corridor_resuggest",
                            "admissible_execution_actions": promoted_admissible,
                            "selection_policy": "bo_top1",
                            "reasoning": (
                                f"{action_package.get('reasoning', '')} "
                                f"[runtime promotion: {promotion_reason}_promoted_resuggest]"
                            ).strip(),
                        }
                        controller_plan = {
                            **(controller_plan or {}),
                            "requested_execution_action": "mask_scaffold_corridor_resuggest",
                            "selection_policy": "bo_top1",
                            "action_package": action_package,
                        }
                reference_bo_top1_item = self._main_bo_top1_item(shortlist_candidates)
                bo_top1_candidate = (
                    dict(reference_bo_top1_item.get("candidate", {}) or {})
                    if isinstance(reference_bo_top1_item, dict)
                    else dict(shortlist_candidates[0]["candidate"])
                )
                shortlist_candidates, structural_meta = self._annotate_structural_shift(
                    shortlist_candidates,
                    bo_top1_candidate=bo_top1_candidate,
                    dominant_scaffold=decision_context.get("dominant_scaffold"),
                )
                if self.env.is_finite_pool and shortlist_candidates:
                    candidate_contrastive_evidence = build_candidate_contrastive_evidence(
                        shortlist_candidates=shortlist_candidates,
                        history=history,
                        feature_columns=self._feature_columns(),
                        goal=self.env.goal,
                        adapter=self._runtime_contrast_adapter(),
                    )
                if (
                    (self.config.enable_action_package_v2 or self.config.enable_action_package_v06)
                    and str(action_package.get("selection_policy", "bo_top1"))
                    in {
                        "bo_top1_from_shaped_shortlist",
                        "select_from_shaped_shortlist",
                    }
                    and execution_action_current
                    not in {
                        "mask_scaffold_corridor_resuggest",
                        "mask_dominant_resuggest",
                        "mask_low_repeat_resuggest",
                    }
                ):
                    shaped_shortlist_candidates, shortlist_shaping_trace = self._shape_shortlist_candidates(
                        shortlist_candidates=shortlist_candidates,
                        history=history,
                        decision_context=decision_context,
                        action_package=action_package,
                    )
                    shortlist_candidates = shaped_shortlist_candidates
                else:
                    shaped_shortlist_candidates = list(shortlist_candidates)
                    shortlist_shaping_trace = {
                        "enabled": False,
                        "reason": "selection_policy_bo_top1",
                        "summary": "Controller kept direct BO execution; shortlist shaping skipped.",
                        "candidate_traces": [],
                    }
                selection_phase = "llm_rerank"
                if execution_action_current in {
                    "mask_scaffold_corridor_resuggest",
                    "mask_dominant_resuggest",
                    "mask_low_repeat_resuggest",
                }:
                    selection_phase = "shortlist_select_bo"
                elif str(action_package.get("selection_policy", "bo_top1")) == "bo_top1_from_shaped_shortlist":
                    selection_phase = "shortlist_select_bo"
                self._emit_progress(
                    "phase_start",
                    iteration=iteration,
                    phase=selection_phase,
                    elapsed_sec=round(time.time() - start_time, 1),
                )
                rerank_policy = self._rerank_policy_state(
                    iteration=iteration,
                    trusted_planner_mode_active=trusted_planner_mode_active,
                    trusted_planner_override_allowed=trusted_planner_override_allowed,
                    trusted_pre_gate_skip=bool(
                        (not v06_enabled)
                        and trusted_planner_mode_active
                        and self.config.enable_trusted_planner_pre_rerank_skip
                        and not trusted_planner_override_allowed
                        and not controller_soft_rerank_authorized
                    ),
                )
                if state_router_guidance:
                    rerank_policy = {
                        **rerank_policy,
                        "state_router_guidance": state_router_guidance,
                    }
                if candidate_contrastive_evidence:
                    rerank_policy = {
                        **rerank_policy,
                        "candidate_contrastive_evidence": candidate_contrastive_evidence,
                    }
                planner_action_policy = self._planner_action_policy()
                rerank_skipped_by_planner_policy = bool(
                    (not v06_enabled)
                    and (
                        trusted_pre_gate_forced_direct
                        or rerank_policy.get("trusted_pre_gate_skip", False)
                    )
                )
                selection_policy = str(action_package.get("selection_policy", "bo_top1"))
                selection_authority = self._evaluate_selection_authority(
                    execution_action=execution_action_current,
                    state_router_guidance=state_router_guidance,
                    candidate_contrastive_evidence=candidate_contrastive_evidence,
                    decision_context=decision_context,
                )
                candidate_direction_review = self._candidate_direction_review_state(
                    execution_action=execution_action_current,
                    decision_context=decision_context,
                    shortlist_candidates=shortlist_candidates,
                )
                candidate_direction_review_applied = False
                shape_shortlist_selection_trace: dict[str, Any] = {
                    "enabled": False,
                    "mode": "bo_top1",
                    "reason": "not_shape_selection",
                }
                if execution_action_current in {
                    "mask_scaffold_corridor_resuggest",
                    "mask_dominant_resuggest",
                    "mask_low_repeat_resuggest",
                }:
                    post_resuggest_selection_mode = self._post_resuggest_selection_mode(
                        execution_action=execution_action_current,
                        state_router_guidance=state_router_guidance,
                        resuggest_trace=resuggest_trace,
                    )
                    if (
                        post_resuggest_selection_mode == "probe_topk"
                        and bool(selection_authority.get("shortlist_probe_authorized", False))
                    ):
                        rerank_policy = {
                            **rerank_policy,
                            "prompt_style": "resuggest_probe_topk",
                            "near_best_margin": 0.08,
                            "resuggest_hybrid_policy": {
                                "enabled": True,
                                "selection_mode": post_resuggest_selection_mode,
                                "planner_rank_topk": 3,
                            },
                        }
                        rerank_action = self.decision_engine.rerank_shortlist(
                            decision_context=decision_context,
                            shortlist_candidates=shortlist_candidates,
                            controller_plan=controller_plan,
                            rerank_policy=rerank_policy,
                        )
                    else:
                        if post_resuggest_selection_mode == "probe_topk":
                            post_resuggest_selection_mode = "bo_top1"
                        bo_preferred_item = self._main_bo_top1_item(shortlist_candidates)
                        chosen_item = bo_preferred_item or shortlist_candidates[0]
                        rerank_action = {
                            "selected_index": int(chosen_item.get("candidate_index", 0)),
                            "candidate_scores": [],
                            "reasoning": (
                                "Skipped shortlist rerank; pre-shortlist resuggest changed the legal pool and the final pick follows BO top-1 in the resuggested pool."
                            ),
                            "prompt_style": "resuggest_bo_pick",
                            "state_router_guidance": state_router_guidance,
                            "post_resuggest_selection_mode": post_resuggest_selection_mode,
                        }
                elif (
                    execution_action_current == "shape_then_probe_topk"
                    and bool(selection_authority.get("shortlist_probe_authorized", False))
                ):
                    rerank_policy = {
                        **rerank_policy,
                        "prompt_style": "shape_probe_topk",
                        "near_best_margin": 0.06,
                        "shape_hybrid_policy": {
                            "enabled": True,
                            "selection_mode": "shape_probe_topk",
                            "planner_rank_topk": int(
                                (state_router_guidance or {}).get("planner_rank_topk", 3) or 3
                            ),
                        },
                    }
                    rerank_action = self.decision_engine.rerank_shortlist(
                        decision_context=decision_context,
                        shortlist_candidates=shortlist_candidates,
                        controller_plan=controller_plan,
                        rerank_policy=rerank_policy,
                    )
                elif execution_action_current == "shape_then_probe_topk":
                    bo_preferred_item = self._bo_preferred_item_from_shaped_shortlist(shortlist_candidates)
                    chosen_item = bo_preferred_item or shortlist_candidates[0]
                    rerank_action = {
                        "selected_index": int(chosen_item.get("candidate_index", 0)),
                        "candidate_scores": [],
                        "reasoning": (
                            "Skipped shortlist probe; evidence sufficiency gate kept planner-preferred "
                            "candidate after shortlist shaping."
                        ),
                        "prompt_style": "shape_only",
                        "selection_status": "planner_only_authority_fallback",
                    }
                elif bool(candidate_direction_review.get("eligible", False)):
                    review_shortlist = list(
                        shortlist_candidates[
                            : max(2, int(candidate_direction_review.get("max_candidates", 5) or 5))
                        ]
                    )
                    if execution_action_current in {"shape_only_bo_pick"}:
                        review_fallback_item = (
                            self._bo_preferred_item_from_shaped_shortlist(review_shortlist)
                            or self._bo_preferred_item_from_shaped_shortlist(shortlist_candidates)
                            or review_shortlist[0]
                        )
                    else:
                        review_fallback_item = (
                            self._main_bo_top1_item(review_shortlist)
                            or self._main_bo_top1_item(shortlist_candidates)
                            or review_shortlist[0]
                        )
                    review_fallback_index = int(review_fallback_item.get("candidate_index", 0) or 0)
                    review_indices = [
                        int(item.get("candidate_index", idx) or idx)
                        for idx, item in enumerate(review_shortlist)
                    ]
                    direction_review_policy = {
                        **rerank_policy,
                        "prompt_style": "candidate_direction_review",
                        "near_best_margin": float(
                            self.config.candidate_direction_review_near_best_margin or 0.05
                        ),
                        "candidate_direction_review": {
                            "mode": candidate_direction_review.get("mode"),
                            "allow_final_replacement": (
                                candidate_direction_review.get("mode") == "bounded_pick"
                            ),
                            "max_candidates": candidate_direction_review.get("max_candidates"),
                        },
                        "state_router_guidance": {
                            **dict(state_router_guidance or {}),
                            "visible_evidence_state": (
                                (state_router_guidance or {}).get("visible_evidence_state")
                                if isinstance(state_router_guidance, dict)
                                else "candidate_direction_review"
                            )
                            or "candidate_direction_review",
                            "admissible_candidate_indices": review_indices,
                            "preferred_candidate_indices": [
                                idx for idx in review_indices if idx != review_fallback_index
                            ]
                            or [review_fallback_index],
                            "fallback_candidate_index": review_fallback_index,
                            "preferred_selection_policy": "candidate_direction_review",
                            "rationale": (
                                "Config-gated v0.7.5 candidate direction review: "
                                "BO supplies the shortlist; LLM may choose a bounded near-top "
                                "candidate only when support and downside risk justify it."
                            ),
                        },
                    }
                    direction_rerank_action = self.decision_engine.rerank_shortlist(
                        decision_context=decision_context,
                        shortlist_candidates=review_shortlist,
                        controller_plan=controller_plan,
                        rerank_policy=direction_review_policy,
                    )
                    if candidate_direction_review.get("mode") == "bounded_pick":
                        rerank_action = direction_rerank_action
                        candidate_direction_review_applied = True
                    else:
                        rerank_action = {
                            "selected_index": review_fallback_index,
                            "candidate_scores": direction_rerank_action.get("candidate_scores", []),
                            "reasoning": (
                                "Candidate direction review ran in advisory mode; final pick kept BO-preferred candidate."
                            ),
                            "prompt_style": "candidate_direction_review",
                            "selection_status": "candidate_direction_review_advisory",
                            "advisory_selected_index": direction_rerank_action.get("selected_index"),
                            "requested_selected_index": direction_rerank_action.get(
                                "requested_selected_index"
                            ),
                        }
                    candidate_direction_review = {
                        **candidate_direction_review,
                        "applied": bool(candidate_direction_review_applied),
                        "selected_index": rerank_action.get("selected_index"),
                        "advisory_selected_index": (
                            direction_rerank_action.get("selected_index")
                            if candidate_direction_review.get("mode") == "advisory"
                            else None
                        ),
                        "selection_status": rerank_action.get("selection_status"),
                    }
                    rerank_policy = direction_review_policy
                elif selection_policy == "bo_top1_from_shaped_shortlist":
                    shape_selected_item, shape_shortlist_selection_trace = (
                        self._shape_shortlist_preferred_item(
                            shortlist_candidates=shortlist_candidates,
                            history=full_observation_history,
                            iteration=iteration,
                        )
                    )
                    bo_preferred_item = (
                        shape_selected_item
                        or self._bo_preferred_item_from_shaped_shortlist(shortlist_candidates)
                    )
                    chosen_item = bo_preferred_item or shortlist_candidates[0]
                    rerank_action = {
                        "selected_index": int(chosen_item.get("candidate_index", 0)),
                        "candidate_scores": [],
                        "reasoning": (
                            "Skipped shortlist-internal LLM selection; selected from shaped "
                            "shortlist using configured planner-side selector."
                        ),
                        "prompt_style": "shape_only",
                        "selection_status": (
                            "shape_shortlist_"
                            + str(shape_shortlist_selection_trace.get("mode", "bo_top1"))
                            if bool(shape_shortlist_selection_trace.get("enabled", False))
                            else "shape_shortlist_bo_top1"
                        ),
                        "shape_shortlist_selection": shape_shortlist_selection_trace,
                    }
                elif rerank_skipped_by_planner_policy:
                    bo_top1_item = self._main_bo_top1_item(shortlist_candidates) or shortlist_candidates[0]
                    rerank_action = {
                        "selected_index": int(bo_top1_item.get("candidate_index", 0)),
                        "candidate_scores": [],
                        "reasoning": "Skipped LLM rerank because trusted planner pre-gate kept BO top-1.",
                        "prompt_style": str(rerank_policy.get("prompt_style", "default")),
                    }
                else:
                    rerank_action = self.decision_engine.rerank_shortlist(
                        decision_context=decision_context,
                        shortlist_candidates=shortlist_candidates,
                        controller_plan=controller_plan,
                        rerank_policy=rerank_policy,
                    )
                candidate_probe_preferred_item = self._candidate_probe_preferred_item(
                    shortlist_candidates=shortlist_candidates,
                    candidate_probe_trace=candidate_probe_trace,
                    history=full_observation_history,
                )
                if candidate_probe_preferred_item is not None:
                    candidate_probe_selection_mode = str(
                        candidate_probe_trace.get("effective_selection_mode")
                        or self.config.candidate_probe_selection_mode
                        or "merge_only"
                    )
                    rerank_action = {
                        **dict(rerank_action or {}),
                        "selected_index": int(
                            candidate_probe_preferred_item.get("candidate_index", 0) or 0
                        ),
                        "candidate_probe_selection_mode": candidate_probe_selection_mode,
                        "selection_status": f"candidate_probe_{candidate_probe_selection_mode}",
                        "reasoning": (
                            str((rerank_action or {}).get("reasoning") or "").strip()
                            + " Candidate_probe selected an injected candidate using "
                            + candidate_probe_selection_mode
                            + "."
                        ).strip(),
                    }
                chosen_index = int(rerank_action.get("selected_index", 0))
                shortlist_by_index = {
                    int(item.get("candidate_index", idx)): item
                    for idx, item in enumerate(shortlist_candidates)
                }
                chosen_item = shortlist_by_index.get(chosen_index, shortlist_candidates[0])
                selected_candidate = dict(chosen_item["candidate"])
                selected_candidate_rank_in_shortlist = next(
                    (
                        idx
                        for idx, item in enumerate(shortlist_candidates)
                        if int(item.get("candidate_index", idx)) == int(chosen_item.get("candidate_index", -1))
                    ),
                    0,
                )
                llm_selected_candidate = dict(chosen_item["candidate"])
                llm_selected_candidate_rank_in_shortlist = selected_candidate_rank_in_shortlist
                guardrail_enabled_for_turn = bool(
                    self.config.enable_override_guardrail
                    and str(planner_action_policy.get("guardrail_mode", "veto_only")) == "veto_only"
                    and (
                        bool(selection_authority.get("shortlist_probe_authorized", False))
                        or bool(candidate_direction_review_applied)
                    )
                )
                chosen_item, override_guardrail = self._apply_override_guardrail(
                    shortlist_candidates=shortlist_candidates,
                    chosen_item=chosen_item,
                    rerank_action=rerank_action,
                    candidate_contrastive_evidence=candidate_contrastive_evidence,
                    enabled=guardrail_enabled_for_turn,
                    trusted_planner_mode_active=trusted_planner_mode_active,
                    trusted_planner_override_allowed=trusted_planner_override_allowed,
                    trusted_planner_block_reason=trusted_planner_block_reason,
                    trusted_main_pool_soft_override_allowed=controller_soft_rerank_authorized,
                    late_stage_active=bool(rerank_policy.get("late_stage_active", False)),
                    current_best_percentile=rerank_policy.get("current_best_percentile"),
                    strong_incumbent_present=bool(rerank_policy.get("strong_incumbent_present", False)),
                )
                selected_candidate = dict(chosen_item["candidate"])
                selected_candidate_rank_in_shortlist = next(
                    (
                        idx
                        for idx, item in enumerate(shortlist_candidates)
                        if int(item.get("candidate_index", idx)) == int(chosen_item.get("candidate_index", -1))
                    ),
                    0,
                )
                selected_differs_from_bo_top1 = bool(
                    not bool(chosen_item.get("is_main_bo_top1"))
                )
                selected_candidate_source = str(chosen_item.get("shortlist_source", "bo_top_ranked"))
                selected_candidate_pool_source = str(chosen_item.get("pool_source", "main_pool"))
                if execution_action_current in {
                    "mask_scaffold_corridor_resuggest",
                    "mask_dominant_resuggest",
                    "mask_low_repeat_resuggest",
                }:
                    planner_preferred_item = self._main_bo_top1_item(shortlist_candidates) or shortlist_candidates[0]
                    planner_preferred_after_shaping = False
                    planner_preferred_after_resuggest = True
                elif execution_action_current in {"shape_only_bo_pick", "shape_then_probe_topk"}:
                    planner_preferred_item = (
                        self._bo_preferred_item_from_shaped_shortlist(shortlist_candidates)
                        or shortlist_candidates[0]
                    )
                    planner_preferred_after_shaping = True
                    planner_preferred_after_resuggest = False
                else:
                    planner_preferred_item = self._main_bo_top1_item(shortlist_candidates) or shortlist_candidates[0]
                    planner_preferred_after_shaping = False
                    planner_preferred_after_resuggest = False
                planner_preferred_candidate_index = int(
                    planner_preferred_item.get("candidate_index", 0) or 0
                )
                guardrail_veto_applied = bool(
                    override_guardrail.get("enabled", False)
                    and not bool(override_guardrail.get("passed", True))
                )
                if guardrail_veto_applied:
                    final_selection_source = "guardrail_veto_fallback"
                elif bool(selection_authority.get("shortlist_probe_authorized", False)):
                    final_selection_source = "bounded_probe"
                elif bool(candidate_direction_review_applied):
                    final_selection_source = "candidate_direction_review"
                elif planner_preferred_after_resuggest:
                    final_selection_source = "planner_after_resuggest"
                elif planner_preferred_after_shaping:
                    final_selection_source = "planner_after_shaping"
                else:
                    final_selection_source = "planner_direct"
                if self.env.is_finite_pool:
                    bo_candidate_origin = (
                        "finite_pool_subspace_filter"
                        if use_subspace
                        else "finite_pool_full_pool"
                    )
                    shortlist_candidate_values = self.execution_adapter.candidate_values(
                        shortlist_candidates
                    )
                    bo_top1_index = int(reference_bo_top1_item.get("candidate_index", 0) or 0)
                    llm_selected_index = int(chosen_index) if llm_selected_candidate is not None else None
                    final_selected_index = int(chosen_item.get("candidate_index", 0) or 0)
                    shortlist_value_audit_payload = shortlist_value_audit(
                        shortlist_candidates=shortlist_candidates,
                        candidate_values=shortlist_candidate_values,
                        bo_top1_index=bo_top1_index,
                        llm_selected_index=llm_selected_index,
                        final_selected_index=final_selected_index,
                        goal=self.env.goal,
                    )

            if v06_enabled:
                execution_contract = self._build_v06_execution_contract(
                    requested_execution_action=execution_action_requested,
                    selected_differs_from_bo_top1=selected_differs_from_bo_top1,
                    shortlist_shaping_trace=shortlist_shaping_trace,
                    resuggest_trace=resuggest_trace,
                    candidate_probe_trace=candidate_probe_trace,
                    shortlist_candidates=shortlist_candidates,
                    rerank_action=rerank_action,
                )
                if execution_action_fallback_reason and not execution_contract.get("fallback_reason"):
                    execution_contract["fallback_reason"] = execution_action_fallback_reason
                execution_action_current = str(
                    execution_contract.get("executed_execution_action", execution_action_current)
                )
                controller_mode = execution_action_current
            else:
                execution_contract = {
                    "requested_execution_action": execution_action_requested,
                    "executed_execution_action": execution_action_current,
                    "contract_satisfied": True,
                    "fallback_reason": execution_action_fallback_reason,
                    "shape_contract_satisfied": self._shape_contract_satisfied(
                        shortlist_shaping_trace,
                        shortlist_candidates=shortlist_candidates,
                    ),
                    "alt_pick_contract_satisfied": bool(selected_differs_from_bo_top1),
                    "candidate_probe_contract_satisfied": bool(
                        candidate_probe_trace.get("applied", False)
                    ),
                }
            controller_plan = {
                **(controller_plan or {}),
                "intervention_type": controller_mode,
                "requested_execution_action": execution_contract.get("requested_execution_action"),
                "executed_execution_action": execution_contract.get("executed_execution_action"),
                "contract_satisfied": execution_contract.get("contract_satisfied"),
                "fallback_reason": execution_contract.get("fallback_reason"),
                "action_package": action_package,
            }

            assert selected_candidate is not None
            candidate_seen_before = self._candidate_seen_before(selected_candidate, history)
            focus_filter_applied = effective_constraint_summary.get("mode") not in {"none", "full_pool"}
            last_intervention_type = controller_mode
            self.execution_adapter.validate_candidate(selected_candidate)
            planner_diagnostics = self.bo_tool.planner_diagnostics()

            self._emit_progress(
                "phase_start",
                iteration=iteration,
                phase="llm_semantic_assessment",
                elapsed_sec=round(time.time() - start_time, 1),
            )
            semantic_context, semantic_reviewed_trace = self._augment_context_with_reviewed_knowledge(
                node_name="semantic_assessment",
                decision_context=decision_context,
                candidate=selected_candidate,
            )
            reviewed_knowledge_trace["semantic_assessment"] = semantic_reviewed_trace
            semantic_assessment = self.decision_engine.semantic_assessment(
                candidate=selected_candidate,
                decision_context=semantic_context,
                reaction_context=self.reaction_context,
            )
            verification_mode = str(self.config.verification_mode).strip().lower()
            verification_policy = str(action_package.get("verification_policy", "normal")).strip().lower()
            verification_should_run = (
                verification_mode != "off"
                and (
                    verification_policy == "strict"
                    or semantic_assessment.get("risk_level", "low") in {"medium", "high"}
                )
            )
            if verification_should_run:
                verification_context, verification_reviewed_trace = self._augment_context_with_reviewed_knowledge(
                    node_name="verification_pass",
                    decision_context=decision_context,
                    candidate=selected_candidate,
                )
                reviewed_knowledge_trace["verification_pass"] = verification_reviewed_trace
                verification_pass = self.decision_engine.candidate_verification(
                    candidate=selected_candidate,
                    decision_context=verification_context,
                    reaction_context=self.reaction_context,
                    semantic_assessment=semantic_assessment,
                    controller_plan=controller_plan,
                )
                verification_pass = {
                    **verification_pass,
                    "verification_policy": verification_policy,
                }
                if verification_mode in {"advisory", "decision_active"} and verification_pass.get("status") in {"caution", "fail_soft"}:
                    verification_reason = str(verification_pass.get("reasoning", "")).strip()
                    risk_flags = list(verification_pass.get("risk_flags", []))
                    semantic_assessment = {
                        **semantic_assessment,
                        "soft_comment": (
                            f"{semantic_assessment.get('soft_comment', '')} [verification: {verification_reason}]"
                        ).strip(),
                        "suggested_bias": list(dict.fromkeys([*semantic_assessment.get("suggested_bias", []), *risk_flags])),
                    }
            selected_feasibility_action = {
                "action": "accept",
                "reasoning": semantic_assessment.get("soft_comment", "semantic soft assessment"),
            }

            best_before_iteration = self._current_best()
            self._emit_progress(
                "phase_start",
                iteration=iteration,
                phase="evaluate_candidate",
                elapsed_sec=round(time.time() - start_time, 1),
            )
            result = self.execution_adapter.evaluate_and_observe(selected_candidate)
            improved_best = self._is_improvement(result, best_before_iteration)

            self._emit_progress(
                "phase_start",
                iteration=iteration,
                phase="llm_reflection",
                elapsed_sec=round(time.time() - start_time, 1),
            )
            reflection_context, reflection_reviewed_trace = self._augment_context_with_reviewed_knowledge(
                node_name="reflection_action",
                decision_context=decision_context,
                candidate=selected_candidate,
                result=result,
            )
            reviewed_knowledge_trace["reflection_action"] = reflection_reviewed_trace
            reflection_action = self.decision_engine.reflection_action(
                decision_action={
                    "stage": stage,
                    "active_variables": active_variables,
                    "decision_mode": "mixed",
                    "reasoning": (
                        controller_plan.get("reasoning", "")
                        if controller_plan is not None
                        else "BO full-space default path."
                    ),
                },
                candidate=selected_candidate,
                feasibility_action=selected_feasibility_action,
                result=result,
                decision_context=reflection_context,
            )
            post_round_history = history + [
                {
                    "iteration": iteration,
                    "stage": stage,
                    "trigger_reasons": trigger_reasons,
                    "controller_mode": controller_mode,
                    "intervention_type": last_intervention_type,
                    "requested_execution_action": execution_contract.get("requested_execution_action"),
                    "executed_execution_action": execution_contract.get("executed_execution_action"),
                    "contract_satisfied": execution_contract.get("contract_satisfied"),
                    "execution_fallback_reason": execution_contract.get("fallback_reason"),
                    "shape_contract_satisfied": execution_contract.get("shape_contract_satisfied"),
                    "alt_pick_contract_satisfied": execution_contract.get("alt_pick_contract_satisfied"),
                    "subspace_active": use_subspace,
                    "active_variables": active_variables,
                    "candidate": selected_candidate,
                    "result": result,
                    "improved_best": improved_best,
                    "feasibility_action": selected_feasibility_action.get("action", "accept"),
                    "semantic_risk_level": (
                        semantic_assessment.get("risk_level", "low")
                        if semantic_assessment is not None
                        else "low"
                    ),
                    "knowledge_used": len(knowledge_units) > 0,
                    "knowledge_source_types": knowledge_source_types,
                    "reflection": reflection_action,
                    "action_package": action_package,
                    "verification_pass": verification_pass,
                    "shortlist_shaping_trace": shortlist_shaping_trace,
                    "candidate_probe": candidate_probe_trace,
                    "candidate_pool_size": effective_constraint_summary.get("pool_size"),
                }
            ]
            self.online_decision_state.refresh_from_history(
                bootstrap_history=self._initial_state_history,
                history=post_round_history,
                best_observation=self._best_observation(),
                current_best=self._current_best(),
                iteration=iteration,
                observations=len(self.campaign.observations.get_values(as_array=True)),
                total_budget=self.total_budget,
                search_space=self.env.param_space,
                search_space_meta=self.dataset_meta,
                goal=self.env.goal,
                knowledge_units=knowledge_units,
                knowledge_query=knowledge_query_rule,
                knowledge_meta={
                    "filters": knowledge_filters,
                    "source_types": knowledge_source_types,
                    "route": knowledge_route,
                    "rule_query": knowledge_query_rule,
                    "document_query": knowledge_query_document,
                    "retrieved_by_source": knowledge_retrieved_by_source,
                    "injected_by_source": knowledge_injected_by_source,
                    "enable_document_retrieval": enable_document_retrieval,
                },
            )
            self.online_decision_state.apply_round_outputs(
                diagnosis=diagnosis,
                hypothesis_action=hypothesis_action,
                coverage_insight=coverage_insight,
                semantic_assessment=semantic_assessment,
                verification_pass=verification_pass,
                reflection_action=reflection_action,
                controller_plan=controller_plan,
            )
            self.working_memory.replace_state(self.online_decision_state.working_memory_summary())
            episodic_review_candidate = build_episodic_review_candidate(
                run_id=self._run_id(),
                dataset=self.env.dataset,
                iteration=iteration,
                protocol_mode=self.config.protocol_mode,
                trigger_reasons=trigger_reasons,
                controller_plan=controller_plan,
                diagnosis=diagnosis,
                reflection_action=reflection_action,
                candidate=selected_candidate,
                result=result,
                improved_best=improved_best,
            )
            if (
                self.config.enable_episodic_review_queue
                and episodic_review_candidate is not None
                and self.episodic_review_queue is not None
            ):
                self.episodic_review_queue.append(episodic_review_candidate)
            experience_candidate = build_experience_candidate(
                iteration=iteration,
                dataset=self.env.dataset,
                trigger_reasons=trigger_reasons,
                reflection_action=reflection_action,
                intervention_plan=intervention_plan,
            )
            experience_promoted = False
            if (
                self.config.enable_experience_promotion
                and experience_candidate is not None
                and should_promote_experience(experience_candidate)
                and self.long_term_store is not None
            ):
                self.long_term_store.put(
                    ("promoted_experience",),
                    experience_candidate.id,
                    experience_candidate.to_dict(),
                )
                experience_promoted = True

            if controller_mode in {"bo_focus_then_rerank", "focused_shortlist_alt_pick"}:
                consecutive_focus_rounds += 1
                previous_focus_improved = improved_best
            else:
                consecutive_focus_rounds = 0
                previous_focus_improved = None

            decision_context_snapshot = self._decision_context_snapshot(decision_context)
            skill_trace_snapshot = (
                self.decision_engine.skill_trace_snapshot()
                if hasattr(self.decision_engine, "skill_trace_snapshot")
                else {}
            )
            compact_injected_units = self._compact_knowledge_units(knowledge_units)
            compact_retrieved_units_by_source = {
                source: self._compact_knowledge_units(units)
                for source, units in knowledge_retrieved_units_by_source.items()
            }
            decision_context_artifact_path = self._write_audit_artifact(
                iteration=iteration,
                name="decision_context",
                payload=decision_context_snapshot,
            )
            knowledge_artifact_path = self._write_audit_artifact(
                iteration=iteration,
                name="knowledge_bundle",
                payload={
                    "query_rule": knowledge_query_rule,
                    "query_document": knowledge_query_document,
                    "filters": knowledge_filters,
                    "route": knowledge_route,
                    "enable_document_retrieval": enable_document_retrieval,
                    "retrieved_by_source": knowledge_retrieved_by_source,
                    "injected_by_source": knowledge_injected_by_source,
                    "retrieved_units_by_source": compact_retrieved_units_by_source,
                    "injected_units": compact_injected_units,
                    "raw_bundle_available": bool(retrieval_bundle),
                },
            )
            intervention_summary = {
                "planner_action_policy_name": planner_action_policy.get("planner_policy_name"),
                "problem_shaping_action": execution_action_current,
                "selection_authority_level": selection_authority.get("selection_authority_level"),
                "authority_source": selection_authority.get("authority_source"),
                "evidence_sufficiency_passed": selection_authority.get("evidence_sufficiency_passed"),
                "evidence_failure_reasons": selection_authority.get("evidence_failure_reasons", []),
                "shortlist_probe_authorized": selection_authority.get("shortlist_probe_authorized", False),
                "requested_execution_action": execution_contract.get("requested_execution_action"),
                "executed_execution_action": execution_contract.get("executed_execution_action"),
                "contract_satisfied": execution_contract.get("contract_satisfied"),
                "failure_mode": (
                    state_router_guidance.get("failure_mode")
                    if isinstance(state_router_guidance, dict)
                    else None
                ),
                "headroom_bucket": (
                    state_router_guidance.get("headroom_bucket")
                    if isinstance(state_router_guidance, dict)
                    else None
                ),
                "rerank_state_router_guidance": state_router_guidance,
                "shape_selection_mode": (
                    rerank_action.get("prompt_style")
                    if execution_action_current in {"shape_only_bo_pick", "shape_then_probe_topk"}
                    and rerank_action
                    else None
                ),
                "shape_selection_status": (
                    rerank_action.get("selection_status")
                    if execution_action_current in {"shape_only_bo_pick", "shape_then_probe_topk"}
                    and rerank_action
                    else None
                ),
                "shape_admissible_candidate_indices": (
                    rerank_action.get("admissible_candidate_indices", [])
                    if execution_action_current in {"shape_only_bo_pick", "shape_then_probe_topk"}
                    and rerank_action
                    else []
                ),
                "shape_effective_preferred_candidate_indices": (
                    rerank_action.get("effective_preferred_candidate_indices", [])
                    if execution_action_current in {"shape_only_bo_pick", "shape_then_probe_topk"}
                    and rerank_action
                    else []
                ),
                "shape_candidate_scores": (
                    rerank_action.get("candidate_scores", [])
                    if execution_action_current in {"shape_only_bo_pick", "shape_then_probe_topk"}
                    and rerank_action
                    and rerank_action.get("prompt_style") == "shape_probe_topk"
                    else []
                ),
                "planner_preferred_candidate_index": planner_preferred_candidate_index,
                "planner_preferred_after_shaping": planner_preferred_after_shaping,
                "planner_preferred_after_resuggest": planner_preferred_after_resuggest,
                "final_selection_source": final_selection_source,
                "candidate_probe": candidate_probe_trace,
                "candidate_probe_applied": bool(candidate_probe_trace.get("applied", False)),
                "candidate_direction_review": candidate_direction_review,
                "candidate_direction_review_applied": candidate_direction_review_applied,
                "guardrail_veto_applied": guardrail_veto_applied,
                "downside_risk_level": override_guardrail.get("downside_risk_level"),
                "support_gap_vs_bo_top1": override_guardrail.get("support_gap_vs_bo_top1"),
            }
            intervention_artifact_path = self._write_audit_artifact(
                iteration=iteration,
                name="intervention_plan",
                payload={
                    **(controller_plan or {}),
                    **intervention_summary,
                },
            )
            action_package_artifact_path = self._write_audit_artifact(
                iteration=iteration,
                name="action_package",
                payload={
                    "action_package": action_package,
                    "execution_contract": execution_contract,
                },
            )
            shaped_shortlist_artifact_path = self._write_audit_artifact(
                iteration=iteration,
                name="shaped_shortlist",
                payload={
                    "action_package": action_package,
                    "execution_contract": execution_contract,
                    "shortlist_shaping_trace": shortlist_shaping_trace,
                    "candidate_probe": candidate_probe_trace,
                    "shaped_shortlist_candidates": shortlist_candidates,
                },
            )
            rerank_artifact_path = self._write_audit_artifact(
                iteration=iteration,
                name="rerank_action",
                payload={
                    "rerank_action": rerank_action or {},
                    "rerank_policy": rerank_policy,
                    "rerank_skipped_by_planner_policy": rerank_skipped_by_planner_policy,
                    "execution_contract": execution_contract,
                    **intervention_summary,
                },
            )
            completion_artifact_path = self._write_audit_artifact(
                iteration=iteration,
                name="completion_action",
                payload=completion_action or {},
            )
            llm_skills_artifact_path = self._write_audit_artifact(
                iteration=iteration,
                name="llm_skills",
                payload=skill_trace_snapshot,
            )
            verification_artifact_path = self._write_audit_artifact(
                iteration=iteration,
                name="verification_pass",
                payload=verification_pass or {},
            )
            reviewed_knowledge_artifact_path = self._write_audit_artifact(
                iteration=iteration,
                name="reviewed_knowledge",
                payload=reviewed_knowledge_trace,
            )
            episodic_review_candidate_artifact_path = self._write_audit_artifact(
                iteration=iteration,
                name="episodic_review_candidate",
                payload=(
                    episodic_review_candidate.to_dict()
                    if episodic_review_candidate is not None
                    else {
                        "created": False,
                        "reason": "no_reflection_insight",
                    }
                ),
            )
            online_decision_state_artifact_path = self._write_audit_artifact(
                iteration=iteration,
                name="online_state",
                payload=self.online_decision_state.summarize(),
            )

            record = {
                "iteration": iteration,
                "stage": stage,
                "decision_mode": "mixed",
                "decision_reasoning": (
                    controller_plan.get("reasoning", "")
                    if controller_plan is not None
                    else ""
                ),
                "planner_action_policy_name": planner_action_policy.get("planner_policy_name"),
                "trigger_reasons": trigger_reasons,
                "controller_trigger_reasons": controller_trigger_reasons,
                "intervention_type": last_intervention_type,
                "controller_mode": controller_mode,
                "problem_shaping_action": execution_action_current,
                "selection_authority_level": selection_authority.get("selection_authority_level"),
                "authority_source": selection_authority.get("authority_source"),
                "evidence_sufficiency_passed": selection_authority.get("evidence_sufficiency_passed"),
                "evidence_failure_reasons": selection_authority.get("evidence_failure_reasons", []),
                "shortlist_probe_authorized": selection_authority.get("shortlist_probe_authorized", False),
                "requested_execution_action": execution_contract.get("requested_execution_action"),
                "executed_execution_action": execution_contract.get("executed_execution_action"),
                "contract_satisfied": execution_contract.get("contract_satisfied"),
                "execution_fallback_reason": execution_contract.get("fallback_reason"),
                "shape_contract_satisfied": execution_contract.get("shape_contract_satisfied"),
                "alt_pick_contract_satisfied": execution_contract.get("alt_pick_contract_satisfied"),
                "shape_probe_contract_satisfied": execution_contract.get("shape_probe_contract_satisfied"),
                "resuggest_contract_satisfied": execution_contract.get("resuggest_contract_satisfied"),
                "action_package": action_package,
                "action_package_version": action_package.get("schema_version"),
                "admissible_execution_actions": action_package.get("admissible_execution_actions"),
                "intent_label": action_package.get("intent"),
                "shortlist_policy": action_package.get("shortlist_policy"),
                "repeat_policy": action_package.get("repeat_policy"),
                "selection_policy": action_package.get("selection_policy"),
                "verification_policy": action_package.get("verification_policy"),
                "focus_policy": action_package.get("focus_policy"),
                "subspace_active": use_subspace,
                "subspace_window_remaining": subspace_window_remaining,
                "consecutive_focus_rounds": consecutive_focus_rounds,
                "active_variables": active_variables,
                "candidate": selected_candidate,
                "bo_top1_candidate": bo_top1_candidate,
                "shortlist_candidates": shortlist_candidates,
                "shaped_shortlist_candidates": shortlist_candidates,
                "shortlist_shaping_trace": shortlist_shaping_trace,
                "resuggest_trace": resuggest_trace,
                "candidate_probe": candidate_probe_trace,
                "candidate_probe_applied": bool(candidate_probe_trace.get("applied", False)),
                "candidate_probe_contract_satisfied": execution_contract.get(
                    "candidate_probe_contract_satisfied"
                ),
                "post_resuggest_selection_mode": post_resuggest_selection_mode,
                "shortlist_size": len(shortlist_candidates),
                "selected_candidate_rank_in_shortlist": selected_candidate_rank_in_shortlist,
                "selected_differs_from_bo_top1": selected_differs_from_bo_top1,
                "llm_requested_override": bool(
                    override_guardrail.get("llm_requested_override", False)
                ),
                "llm_selected_candidate": llm_selected_candidate,
                "llm_selected_candidate_rank_in_shortlist": llm_selected_candidate_rank_in_shortlist,
                "planner_preferred_candidate_index": planner_preferred_candidate_index,
                "planner_preferred_after_shaping": planner_preferred_after_shaping,
                "planner_preferred_after_resuggest": planner_preferred_after_resuggest,
                "final_selection_source": final_selection_source,
                "candidate_direction_review": candidate_direction_review,
                "candidate_direction_review_applied": candidate_direction_review_applied,
                "override_guardrail_enabled": bool(override_guardrail.get("enabled", False)),
                "override_guardrail_passed": bool(override_guardrail.get("passed", True)),
                "guardrail_veto_applied": guardrail_veto_applied,
                "override_guardrail_action": override_guardrail.get("action"),
                "override_guardrail_reason": override_guardrail.get("reason"),
                "override_guardrail_score_margin": override_guardrail.get("score_margin"),
                "override_guardrail_structural_shift": override_guardrail.get("structural_shift"),
                "override_guardrail_strong_structural_shift": override_guardrail.get(
                    "strong_structural_shift"
                ),
                "override_guardrail_bo_scaffold_shift_count": override_guardrail.get(
                    "bo_scaffold_shift_count"
                ),
                "override_guardrail_dominant_scaffold_shift_count": override_guardrail.get(
                    "dominant_scaffold_shift_count"
                ),
                "planner_trust_policy": self.config.planner_trust_policy,
                "trusted_planner_mode_active": bool(
                    override_guardrail.get("trusted_planner_mode_active", trusted_planner_mode_active)
                ),
                "trusted_planner_override_allowed": bool(
                    override_guardrail.get("trusted_planner_override_allowed", trusted_planner_override_allowed)
                ),
                "trusted_planner_block_reason": override_guardrail.get(
                    "trusted_planner_block_reason",
                    trusted_planner_block_reason,
                ),
                "blocked_by_trusted_planner_policy": bool(
                    override_guardrail.get("blocked_by_trusted_planner_policy", False)
                ),
                "blocked_by_late_stage_incumbent_protection": bool(
                    override_guardrail.get("blocked_by_late_stage_incumbent_protection", False)
                ),
                "current_best_percentile": override_guardrail.get(
                    "current_best_percentile",
                    rerank_policy.get("current_best_percentile"),
                ),
                "late_stage_active": bool(
                    override_guardrail.get("late_stage_active", rerank_policy.get("late_stage_active", False))
                ),
                "strong_incumbent_present": bool(
                    override_guardrail.get(
                        "strong_incumbent_present",
                        rerank_policy.get("strong_incumbent_present", False),
                    )
                ),
                "guardrail_selected_candidate_before": llm_selected_candidate,
                "guardrail_selected_candidate_after": selected_candidate,
                "rerank_scores": rerank_action.get("candidate_scores", []) if rerank_action else [],
                "rerank_prompt_style": rerank_action.get("prompt_style") if rerank_action else None,
                "rerank_selected_index": (
                    rerank_action.get("selected_index") if rerank_action else None
                ),
                "rerank_requested_selected_index": (
                    rerank_action.get("requested_selected_index") if rerank_action else None
                ),
                "rerank_selection_status": (
                    rerank_action.get("selection_status") if rerank_action else None
                ),
                "rerank_state_router_guidance": (
                    rerank_policy.get("state_router_guidance")
                    if isinstance(rerank_policy, dict)
                    else None
                ),
                "rerank_visible_evidence_state": (
                    state_router_guidance.get("visible_evidence_state")
                    if isinstance(state_router_guidance, dict)
                    else None
                ),
                "rerank_visible_state_signals": (
                    state_router_guidance.get("visible_state_signals")
                    if isinstance(state_router_guidance, dict)
                    else []
                ),
                "failure_mode": (
                    state_router_guidance.get("failure_mode")
                    if isinstance(state_router_guidance, dict)
                    else None
                ),
                "headroom_bucket": (
                    state_router_guidance.get("headroom_bucket")
                    if isinstance(state_router_guidance, dict)
                    else None
                ),
                "rerank_admissible_candidate_indices": (
                    state_router_guidance.get("admissible_candidate_indices")
                    if isinstance(state_router_guidance, dict)
                    else []
                ),
                "rerank_preferred_candidate_indices": (
                    state_router_guidance.get("preferred_candidate_indices")
                    if isinstance(state_router_guidance, dict)
                    else []
                ),
                "rerank_effective_preferred_candidate_indices": (
                    rerank_action.get("effective_preferred_candidate_indices", [])
                    if rerank_action
                    else []
                ),
                "shape_selection_mode": (
                    rerank_action.get("prompt_style")
                    if execution_action_current in {"shape_only_bo_pick", "shape_then_probe_topk"}
                    else None
                ),
                "shape_admissible_candidate_indices": (
                    rerank_action.get("admissible_candidate_indices", [])
                    if execution_action_current in {"shape_only_bo_pick", "shape_then_probe_topk"}
                    and rerank_action
                    else []
                ),
                "shape_effective_preferred_candidate_indices": (
                    rerank_action.get("effective_preferred_candidate_indices", [])
                    if execution_action_current in {"shape_only_bo_pick", "shape_then_probe_topk"}
                    and rerank_action
                    else []
                ),
                "shape_candidate_scores": (
                    rerank_action.get("candidate_scores", [])
                    if execution_action_current in {"shape_only_bo_pick", "shape_then_probe_topk"}
                    and rerank_action
                    and rerank_action.get("prompt_style") == "shape_probe_topk"
                    else []
                ),
                "shape_selection_status": (
                    rerank_action.get("selection_status")
                    if execution_action_current in {"shape_only_bo_pick", "shape_then_probe_topk"}
                    and rerank_action
                    else None
                ),
                "rerank_preferred_selection_policy": (
                    state_router_guidance.get("preferred_selection_policy")
                    if isinstance(state_router_guidance, dict)
                    else None
                ),
                "downside_risk_level": override_guardrail.get("downside_risk_level"),
                "support_gap_vs_bo_top1": override_guardrail.get("support_gap_vs_bo_top1"),
                "candidate_contrastive_evidence": candidate_contrastive_evidence,
                "rerank_skipped_by_planner_policy": bool(rerank_skipped_by_planner_policy),
                "focus_filter_applied": focus_filter_applied,
                "focus_fallback_reason": focus_fallback_reason,
                "focus_filter_pool_size": (
                    effective_constraint_summary.get("pool_size")
                    if focus_filter_applied
                    else None
                ),
                "shortlist_scaffold_diversity": shortlist_scaffold_diversity,
                "shortlist_pool_strategy": shortlist_pool_strategy,
                "shortlist_source_mix": shortlist_source_mix,
                "shortlist_target_diversity_met": shortlist_target_diversity_met,
                "shortlist_available_scaffold_diversity": shortlist_available_scaffold_diversity,
                "pool_limited_diversity": pool_limited_diversity,
                "diversity_injection_count": diversity_injection_count,
                "diversity_pool_used": diversity_pool_used,
                "diversity_pool_unique_scaffolds": diversity_pool_unique_scaffolds,
                "diversity_pool_quality": structural_meta.get("diversity_pool_quality"),
                "bo_top1_scaffold_key": structural_meta.get("bo_top1_scaffold_key"),
                "dominant_scaffold_key": structural_meta.get("dominant_scaffold_key"),
                "structural_shift_candidate_count": structural_meta.get(
                    "structural_shift_candidate_count"
                ),
                "dominant_shift_candidate_count": structural_meta.get(
                    "dominant_shift_candidate_count"
                ),
                "main_pool_candidates": main_pool_candidates,
                "diversity_pool_candidates": diversity_pool_candidates,
                "selected_candidate_source": selected_candidate_source,
                "selected_candidate_pool_source": selected_candidate_pool_source,
                "shortlist_true_values": shortlist_value_audit_payload.get("shortlist_true_values"),
                "llm_selected_index": shortlist_value_audit_payload.get("llm_selected_index"),
                "llm_selected_true_value": shortlist_value_audit_payload.get("llm_selected_true_value"),
                "final_selected_index": shortlist_value_audit_payload.get("final_selected_index"),
                "selected_true_value": shortlist_value_audit_payload.get("selected_true_value"),
                "bo_top1_true_value": shortlist_value_audit_payload.get("bo_top1_true_value"),
                "best_shortlist_index": shortlist_value_audit_payload.get("best_shortlist_index"),
                "best_shortlist_true_value": shortlist_value_audit_payload.get("best_shortlist_true_value"),
                "best_non_top1_index": shortlist_value_audit_payload.get("best_non_top1_index"),
                "best_non_top1_true_value": shortlist_value_audit_payload.get("best_non_top1_true_value"),
                "shortlist_oracle_headroom": shortlist_value_audit_payload.get("shortlist_oracle_headroom"),
                "best_non_top1_headroom": shortlist_value_audit_payload.get("best_non_top1_headroom"),
                "llm_headroom": shortlist_value_audit_payload.get("llm_headroom"),
                "final_headroom": shortlist_value_audit_payload.get("final_headroom"),
                "feasibility_action": selected_feasibility_action.get("action", "accept"),
                "feasibility_reasoning": selected_feasibility_action.get("reasoning", ""),
                "semantic_risk_level": (
                    semantic_assessment.get("risk_level", "low")
                    if semantic_assessment is not None
                    else "low"
                ),
                "semantic_plausibility_score": (
                    semantic_assessment.get("plausibility_score", 0.0)
                    if semantic_assessment is not None
                    else 0.0
                ),
                "semantic_novelty_score": (
                    semantic_assessment.get("novelty_score", 0.0)
                    if semantic_assessment is not None
                    else 0.0
                ),
                "semantic_comment": (
                    semantic_assessment.get("soft_comment", "")
                    if semantic_assessment is not None
                    else ""
                ),
                "verification_pass": verification_pass,
                "hypothesis_action": hypothesis_action,
                "stagnation_diagnosis": diagnosis,
                "coverage_insight": coverage_insight,
                "knowledge_used": len(knowledge_units) > 0,
                "knowledge_snippet_count": len(knowledge_units),  # backward-compatible alias
                "knowledge_injected_total": len(knowledge_units),
                "knowledge_retrieved_total": int(sum(knowledge_retrieved_by_source.values())),
                "knowledge_retrieved_by_source": knowledge_retrieved_by_source,
                "knowledge_injected_by_source": knowledge_injected_by_source,
                "knowledge_source_types": knowledge_source_types,
                "knowledge_injected_units": compact_injected_units,
                "knowledge_query": knowledge_query_rule,  # backward-compatible alias
                "knowledge_query_rule": knowledge_query_rule,
                "knowledge_query_document": knowledge_query_document,
                "knowledge_trigger_reason": trigger_reasons,
                "knowledge_route": knowledge_route,
                "knowledge_enable_document_retrieval": enable_document_retrieval,
                "reviewed_knowledge_trace": reviewed_knowledge_trace,
                "loaded_skills_by_call": skill_trace_snapshot,
                "decision_context_snapshot": decision_context_snapshot,
                "intervention_plan_full": controller_plan,
                "action_package_full": action_package,
                "rerank_action_full": rerank_action,
                "rerank_policy_full": rerank_policy,
                "completion_action_full": completion_action,
                "bo_candidate_origin": bo_candidate_origin,
                "constraint_mode": effective_constraint_summary.get("mode"),
                "constraint_signature": effective_constraint_summary.get("constraint_signature"),
                "filter_mode": effective_constraint_summary.get("filter_mode"),
                "candidate_pool_size": effective_constraint_summary.get("pool_size"),
                "candidate_pool_total": effective_constraint_summary.get("total_candidate_count"),
                "candidate_pool_name": self.env.dataset,
                "pool_size_before": effective_constraint_summary.get("pool_size_before"),
                "pool_size_after": effective_constraint_summary.get("pool_size_after"),
                "subspace_filter_summary": effective_constraint_summary,
                "focus_filter_summary": effective_constraint_summary,
                "focus_variables": effective_constraint_summary.get("focus_variables", []),
                "filter_applied": focus_filter_applied,
                "subpool_filter_fallback": effective_constraint_summary.get(
                    "subpool_empty_fallback"
                ),
                "candidate_seen_before": candidate_seen_before,
                "improved_best": improved_best,
                "best_result": self._current_best(),
                "planner_refreshed": planner_diagnostics.get("planner_refreshed", False),
                "planner_refresh_reason": planner_diagnostics.get("planner_refresh_reason"),
                "planner_refresh_count": planner_diagnostics.get("planner_refresh_count"),
                "planner_constraint_signature": planner_diagnostics.get("constraint_signature"),
                "planner_family": planner_diagnostics.get("planner_family"),
                "planner_supports_shortlist": planner_diagnostics.get("supports_shortlist"),
                "planner_search_space_type": planner_diagnostics.get("search_space_type"),
                "planner_candidate_pool_mode": planner_diagnostics.get("candidate_pool_mode"),
                "planner_surrogate_name": planner_diagnostics.get("surrogate_name"),
                "planner_acquisition_name": planner_diagnostics.get("acquisition_name"),
                "planner_encoding_name": planner_diagnostics.get("encoding_name"),
                "planner_fallback_reason": planner_diagnostics.get("fallback_reason"),
                "planner_runtime_env": planner_diagnostics.get("runtime_env"),
                "planner_runtime_python": planner_diagnostics.get("runtime_python"),
                "finite_pool_backend": self.env.backend if self.env.is_finite_pool else None,
                "decision_context_artifact_path": decision_context_artifact_path,
                "knowledge_retrieval_artifact_path": knowledge_artifact_path,
                "intervention_plan_artifact_path": intervention_artifact_path,
                "action_package_artifact_path": action_package_artifact_path,
                "shaped_shortlist_artifact_path": shaped_shortlist_artifact_path,
                "rerank_action_artifact_path": rerank_artifact_path,
                "completion_action_artifact_path": completion_artifact_path,
                "llm_skills_artifact_path": llm_skills_artifact_path,
                "verification_artifact_path": verification_artifact_path,
                "reviewed_knowledge_artifact_path": reviewed_knowledge_artifact_path,
                "episodic_review_candidate_artifact_path": episodic_review_candidate_artifact_path,
                "online_decision_state_artifact_path": online_decision_state_artifact_path,
                "episodic_review_candidate": (
                    episodic_review_candidate.to_dict()
                    if episodic_review_candidate is not None
                    else None
                ),
                "experience_candidate_created": experience_candidate is not None,
                "experience_promoted": experience_promoted,
                "result": result,
                "reflection": reflection_action,
                "working_memory": self.working_memory.summarize(),
                # Phase B: LLM search constraint trace fields
                "llm_constraint_active": effective_constraint_summary.get("llm_constraint_active", False),
                "llm_constraint_pool_size": effective_constraint_summary.get("llm_constraint_pool_size"),
                "llm_constraint_summary": effective_constraint_summary.get("llm_constraint_summary"),
                "llm_constraint_updated_this_round": llm_constraint_update_record is not None and llm_constraint_update_record.get("updated", False),
                "llm_constraint_expires_at": self._llm_constraint_expires_at if self._active_llm_constraints else None,
            }
            record.update(self._protocol_metadata())
            self.memory.record_decision(record)
            self._emit_progress(
                "iteration_end",
                iteration=iteration,
                bo_iterations=bo_iterations,
                stage=stage,
                controller_mode=controller_mode,
                selected_candidate=selected_candidate,
                result=result,
                best_result=self._current_best(),
                improved_best=improved_best,
                elapsed_sec=round(time.time() - start_time, 1),
            )

        records = self.memory.get_history()
        trace_df = pd.DataFrame(records)
        runtime_sec = time.time() - start_time
        reviewed_hits_by_node: dict[str, int] = {}
        reviewed_hits_by_entry: dict[str, int] = {}
        reviewed_total_retrieved = 0
        verification_count = 0
        for row in records:
            trace = row.get("reviewed_knowledge_trace", {})
            if not isinstance(trace, dict):
                continue
            if row.get("verification_pass"):
                verification_count += 1
            for node_name, node_trace in trace.items():
                if not isinstance(node_trace, dict):
                    continue
                retrieved_count = int(node_trace.get("retrieved_count", 0) or 0)
                reviewed_total_retrieved += retrieved_count
                if retrieved_count > 0:
                    reviewed_hits_by_node[node_name] = (
                        reviewed_hits_by_node.get(node_name, 0) + retrieved_count
                    )
                for unit in node_trace.get("retrieved_units", []) or []:
                    if not isinstance(unit, dict):
                        continue
                    unit_id = str(unit.get("id") or "").strip()
                    if not unit_id:
                        continue
                    reviewed_hits_by_entry[unit_id] = reviewed_hits_by_entry.get(unit_id, 0) + 1
        summary = {
            "dataset": self.env.dataset,
            "dataset_meta": self.dataset_meta,
            "objective_name": self.env.objective_name,
            "goal": self.env.goal,
            "budget": self.total_budget,
            "initial_observations": initial_observations,
            "initial_candidate_keys": initial_candidate_keys(
                initial_candidates=self.initial_candidates[: self.init_budget],
                param_space=self.env.param_space,
            ),
            "num_observations": len(self.campaign.observations.get_values(as_array=True)),
            "runtime_sec": round(runtime_sec, 3),
            "best_result": self._current_best(),
            "best_observation": self._best_observation(),
            "planner_surrogate_name": self.bo_tool.planner_diagnostics().get("surrogate_name"),
            "planner_acquisition_name": self.bo_tool.planner_diagnostics().get("acquisition_name"),
            "planner_encoding_name": self.bo_tool.planner_diagnostics().get("encoding_name"),
            "planner_fallback_reason": self.bo_tool.planner_diagnostics().get("fallback_reason"),
            "planner_runtime_env": self.bo_tool.planner_diagnostics().get("runtime_env"),
            "planner_runtime_python": self.bo_tool.planner_diagnostics().get("runtime_python"),
            "reviewed_knowledge_enabled": self.config.enable_reviewed_knowledge,
            "reviewed_knowledge_target_nodes": list(self.config.reviewed_knowledge_target_nodes),
            "reviewed_knowledge_total_retrieved": reviewed_total_retrieved,
            "reviewed_knowledge_hits_by_node": reviewed_hits_by_node,
            "reviewed_knowledge_hits_by_entry": reviewed_hits_by_entry,
            "reviewed_knowledge_loaded_titles": (
                self.reviewed_knowledge_store.titles()
                if self.reviewed_knowledge_store is not None
                else []
            ),
            "knowledge_hit_ledger_enabled": self.knowledge_hit_ledger_path is not None,
            "knowledge_hit_ledger_path": (
                str(self.knowledge_hit_ledger_path)
                if self.knowledge_hit_ledger_path is not None
                else None
            ),
            "reviewed_experience_loaded_titles": (
                self.reviewed_experience_store.titles()
                if self.reviewed_experience_store is not None
                else []
            ),
            "verification_pass_count": verification_count,
            "episodic_review_queue_enabled": self.config.enable_episodic_review_queue,
            "episodic_review_queue_summary": (
                self.episodic_review_queue.summary()
                if self.episodic_review_queue is not None
                else {"candidate_count": 0, "queue_path": None, "leakage_risk_counts": {}}
            ),
            **runtime_metadata(),
        }
        summary.update(self._protocol_metadata())
        self._emit_progress(
            "run_end",
            best_result=summary["best_result"],
            runtime_sec=summary["runtime_sec"],
            observations=summary["num_observations"],
            total_budget=self.total_budget,
        )
        return trace_df, summary

    def _best_observation(self) -> dict[str, Any] | None:
        params = self.campaign.observations.get_params(as_array=True)
        values = self.campaign.observations.get_values(as_array=True)
        if len(values) == 0:
            return None
        flat = values.reshape(-1)
        if self.env.goal == "minimize":
            best_idx = int(flat.argmin())
        else:
            best_idx = int(flat.argmax())
        names = [param.name for param in self.env.param_space]
        return {
            **{name: params[best_idx][i] for i, name in enumerate(names)},
            self.env.objective_name: float(flat[best_idx]),
        }
