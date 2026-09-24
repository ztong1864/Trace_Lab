"""Typed configuration schema for Agentic BO."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
    discrete_mainline = [
        "direct_bo_pick",
        "shape_only_bo_pick",
        "finite_pool_candidate_probe",
        "mask_low_repeat_resuggest",
        "mask_dominant_resuggest",
        "mask_scaffold_corridor_resuggest",
    ]
    rare_actions = [
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
            "allowed_mainline_actions": list(discrete_mainline),
            "allowed_rare_actions": list(rare_actions),
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
            "allowed_mainline_actions": list(discrete_mainline),
            "allowed_rare_actions": list(rare_actions),
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
            "allowed_mainline_actions": ["direct_bo_pick", "shape_only_bo_pick"],
            "allowed_rare_actions": list(rare_actions),
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
            "allowed_mainline_actions": list(discrete_mainline),
            "allowed_rare_actions": list(rare_actions),
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
            "allowed_mainline_actions": list(discrete_mainline),
            "allowed_rare_actions": list(rare_actions),
            "allowed_prompt_styles": ["default", "shape_only", "resuggest_bo_pick"],
            "allow_runtime_promotion": False,
            "allow_non_top_final_replacement": False,
            "non_top_replacement_requires_evidence_gate": True,
            "guardrail_mode": "veto_only",
            **thresholds,
        },
    }


@dataclass
class RuntimeConfig:
    env_profile: str = "atlas"
    model_name: str = "gpt-6-luna"
    api_base: str | None = None
    use_responses_api: bool = False
    reasoning_effort: str | None = None
    disable_response_storage: bool = False
    # None: don't send a temperature. Reasoning models such as gpt-6-luna reject it.
    temperature: float | None = None
    llm_timeout_sec: float = 45.0
    llm_request_max_retries: int = 2
    llm_structured_retry_attempts: int = 3
    llm_retry_backoff_sec: float = 1.5
    llm_retry_max_backoff_sec: float = 12.0
    llm_retry_jitter_sec: float = 0.5
    llm_fallback_model_name: str | None = None
    llm_fallback_attempts: int = 2
    llm_fail_on_nonretryable_error: bool = True
    llm_pricing_profile: str = "auto"
    llm_input_cost_per_1m: float | None = None
    llm_output_cost_per_1m: float | None = None
    llm_cached_input_cost_per_1m: float | None = None


@dataclass
class ExperimentConfig:
    dataset: str = "suzuki_hte_full"
    budget: int = 20
    total_budget: int | None = None
    num_init_design: int = 5
    init_budget: int | None = None
    seed: int = 7
    output_dir: str = "runs/agentic_bo"
    planner_name: str = "atlas"
    enable_workflow_report: bool = True
    workflow_report_format: str = "json"
    enable_knowledge: bool = False
    knowledge_local_dir: str = "knowledge/local"
    knowledge_rules_dir: str = "knowledge/curated_rules"
    knowledge_db_dir: str = "knowledge/lancedb"
    knowledge_table_name: str = "knowledge_units"
    knowledge_backend: str = "qdrant"
    enable_qdrant_documents: bool = False
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chem_documents"
    qdrant_dense_model: str = "BAAI/bge-small-en-v1.5"
    qdrant_sparse_model: str = "Qdrant/bm25"
    knowledge_bootstrap: bool = True
    enable_long_term_memory: bool = False
    long_term_memory_path: str = "knowledge/long_term_memory.json"
    enable_audit_artifacts: bool = True
    audit_artifact_dir: str | None = None
    init_strategy: str = "random"
    init_validator_mode: str = "off"
    init_validator_additive_patterns: tuple[str, ...] = ()
    init_validator_ligand_patterns: tuple[str, ...] = ()
    init_validator_base_patterns: tuple[str, ...] = ()
    init_validator_reactant_prefixes: tuple[str, ...] = ()
    protocol_mode: str = "benchmark_clean"
    enable_action_package_v2: bool = False
    enable_action_package_v06: bool = False
    enable_reviewed_knowledge: bool = False
    reviewed_knowledge_dir: str = "knowledge/reviewed"
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
    )
    enable_reviewed_experience: bool = False
    reviewed_experience_dir: str = "knowledge/reviewed_experience"
    reviewed_experience_top_k: int = 2
    enable_episodic_review_queue: bool = False
    episodic_review_queue_path: str | None = None
    enable_knowledge_hit_ledger: bool = True
    knowledge_hit_ledger_path: str | None = None
    terminal_verbosity: str = "progress"
    suppress_third_party_output: bool = True


@dataclass
class OrchestratorSettings:
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
    planner_action_policies: dict[str, dict[str, Any]] = field(
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
    enable_experience_promotion: bool = False
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
    init_strategy: str = "random"
    enable_llm_search_constraint: bool = False
    constraint_update_freq: int = 5
    min_constraint_pool_fraction: float = 0.05
    constraint_max_duration_rounds: int = 8
    terminal_verbosity: str = "progress"
    suppress_third_party_output: bool = True
    third_party_log_path: str | None = None
    progress_jsonl_path: str | None = None
    status_json_path: str | None = None


@dataclass
class PromptInitConfig:
    sample_pool_size: int = 12
    annotation_max_entries: int = 24


@dataclass
class PromptSearchConstraintConfig:
    annotation_max_entries: int = 80


@dataclass
class ReactionOverrideConfig:
    priority_columns: list[str] = field(default_factory=list)
    per_column_caps: dict[str, int] = field(default_factory=dict)


@dataclass
class PromptSkillsConfig:
    enabled: bool = True
    trace_loaded_skills: bool = True
    cards_dir: str | None = None


@dataclass
class PromptConfig:
    history_tail_window: int = 6
    history_reflection_max_chars: int = 160
    knowledge_max_items: int = 5
    knowledge_max_chars: int = 240
    decision_engine_knowledge_max_items: int = 5
    decision_engine_knowledge_max_chars: int = 400
    skills: PromptSkillsConfig = field(default_factory=PromptSkillsConfig)
    init: PromptInitConfig = field(default_factory=PromptInitConfig)
    search_constraint: PromptSearchConstraintConfig = field(default_factory=PromptSearchConstraintConfig)
    reaction_overrides: dict[str, ReactionOverrideConfig] = field(
        default_factory=lambda: {
            "buchwald": ReactionOverrideConfig(
                priority_columns=["Reactant2", "Ligand", "Base", "Additive"],
                per_column_caps={"Additive": 2},
            ),
            "arylation": ReactionOverrideConfig(
                priority_columns=[
                    "Aryl_halide_SMILES",
                    "Additive_SMILES",
                    "Ligand_SMILES",
                    "Base_SMILES",
                ],
                per_column_caps={
                    "Aryl_halide_SMILES": 4,
                    "Additive_SMILES": 4,
                    "Ligand_SMILES": 4,
                },
            ),
        }
    )


@dataclass
class AgenticBOConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    orchestrator: OrchestratorSettings = field(default_factory=OrchestratorSettings)
    prompt: PromptConfig = field(default_factory=PromptConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
