"""Prompt templates for Agentic Chemical BO decisions."""

from __future__ import annotations

import json
from typing import Any


def _to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _knowledge_context_block(
    knowledge_context: list[dict[str, Any]] | None,
    knowledge_meta: dict[str, Any] | None = None,
) -> str:
    snippets = knowledge_context or []
    meta = knowledge_meta or {}
    rule_query = str(meta.get("rule_query", "")).strip()
    document_query = str(meta.get("document_query", "")).strip()
    rule_items: list[dict[str, Any]] = []
    document_items: list[dict[str, Any]] = []
    memory_items: list[dict[str, Any]] = []
    for item in snippets:
        source_type = str(item.get("source_type", ""))
        if source_type in {"document_chunk", "document"}:
            document_items.append(item)
        elif source_type in {"long_term_memory", "reviewed_experience"}:
            memory_items.append(item)
        else:
            rule_items.append(item)
    return f"""
Knowledge context:

Rules:
{_to_json(rule_items)}

Document Evidence:
{_to_json(document_items)}

Memory:
{_to_json(memory_items)}

Rule retrieval query:
{rule_query or "N/A"}

Document retrieval query:
{document_query or "N/A"}
""".strip()


def _value_annotation_block(value_annotations: list[dict[str, Any]] | None) -> str:
    annotations = value_annotations or []
    if not annotations:
        return "Value translation / annotation context:\n[]"
    return f"""
Value translation / annotation context:
{_to_json(annotations)}

Use this as external context for interpreting cryptic strings (e.g. SMILES, abbreviations, opaque ligand names).
- Keep the original string unchanged in any executable candidate or constraint.
- Use the translated description and brief properties only as interpretation aids.
- If annotation confidence is low, treat it as soft evidence rather than a hard fact.
""".strip()


def _instruction_block(skill_block: str | None, fallback_block: str) -> str:
    text = str(skill_block or "").strip()
    if text:
        return text
    return fallback_block.strip()


SYSTEM_PROMPT = """
You are the decision policy of an Agentic Bayesian Optimization system for chemical reaction optimization.

Responsibilities:
1) generate high-level research hypotheses
2) diagnose stagnation and propose sparse interventions
3) choose when BO should run direct versus when shortlist reranking or focused filtering is justified
4) analyze coverage/information gain and summarize research state

Rules:
- Output must be strictly structured JSON matching the required schema.
- Prefer chemical plausibility and optimization progress.
- Avoid repeatedly proposing near-identical candidates unless strongly justified.
- For categorical/literal fields, only use allowed enum values.
- Keep BO as the proposal engine over the valid candidate set.
- You may control execution only within BO-proposed shortlist candidates or focused legal candidate pools.
""".strip()


def build_decision_action_prompt(
    decision_context: dict[str, Any],
    search_space: list[dict[str, Any]],
    search_space_meta: dict[str, Any] | None = None,
    value_annotations: list[dict[str, Any]] | None = None,
    skill_block: str | None = None,
) -> str:
    guidance_block = _instruction_block(
        skill_block,
        """
Decision guidance:
- Allowed fixed_variables_strategy: "llm_proposed", "best_observed", "fallback_default", "mixed".
- If best_improvement_last_3 is small and current_stage_streak is high, change either stage or active_variables pattern.
- Keep active_variables valid and concise; prefer 2-4 variables.
""",
    )
    return f"""
Task: produce the next DecisionAction for BO.

Decision context:
{_to_json(decision_context)}

Search space variables:
{_to_json(search_space)}

Search space metadata:
{_to_json(search_space_meta or {})}

{_value_annotation_block(value_annotations)}

{guidance_block}

Return JSON only for DecisionAction.
""".strip()


def build_completion_action_prompt(
    partial_candidate: dict[str, Any],
    decision_action: dict[str, Any],
    decision_context: dict[str, Any],
    search_space: list[dict[str, Any]],
    search_space_meta: dict[str, Any] | None = None,
    value_annotations: list[dict[str, Any]] | None = None,
) -> str:
    return f"""
Task: complete a BO partial candidate into a full executable candidate.
If search_space_meta.backend is "finite_pool", do not invent unseen combinations.
Only return values that are consistent with known options and context.

Partial BO candidate:
{_to_json(partial_candidate)}

DecisionAction:
{_to_json(decision_action)}

Decision context:
{_to_json(decision_context)}

Search space:
{_to_json(search_space)}

Search space metadata:
{_to_json(search_space_meta or {})}

{_value_annotation_block(value_annotations)}

Return JSON only for CompletionAction.
""".strip()


def build_hypothesis_action_prompt(
    decision_context: dict[str, Any],
    reaction_context: dict[str, Any],
    search_space: list[dict[str, Any]],
    search_space_meta: dict[str, Any] | None = None,
    knowledge_context: list[dict[str, Any]] | None = None,
    knowledge_meta: dict[str, Any] | None = None,
    value_annotations: list[dict[str, Any]] | None = None,
    skill_block: str | None = None,
) -> str:
    guidance_block = _instruction_block(
        skill_block,
        """
Guidance:
- Focus on scientific hypotheses and variable interaction insights.
- Do NOT output direct low-level candidate values.
""",
    )
    return f"""
Task: generate high-level research hypotheses for next optimization phase.

Decision context:
{_to_json(decision_context)}

Reaction context:
{_to_json(reaction_context)}

Search space:
{_to_json(search_space)}

Search space metadata:
{_to_json(search_space_meta or {})}

{_value_annotation_block(value_annotations)}

{_knowledge_context_block(knowledge_context, knowledge_meta)}

{guidance_block}

Return JSON only for HypothesisAction.
""".strip()


def build_stagnation_diagnosis_prompt(
    decision_context: dict[str, Any],
    history_tail: list[dict[str, Any]],
    knowledge_context: list[dict[str, Any]] | None = None,
    knowledge_meta: dict[str, Any] | None = None,
    value_annotations: list[dict[str, Any]] | None = None,
    skill_block: str | None = None,
) -> str:
    guidance_block = _instruction_block(
        skill_block,
        """
Guidance:
- Explain causes (e.g., local trapping, repeated patterns, low coverage).
- Recommend high-level intervention direction rather than direct candidate rewriting.
- In large finite-pool benchmarks, very low visited fraction by itself is not enough to rule out controller-side help.
- Distinguish search scope from execution mode: `keep_full_space` can still justify `bo_rerank_topk`.
- If there is sustained no-improvement plus one-axis local probing or repeated anchor reuse, treat that as actionable local lock even when the full pool is still mostly unvisited.
""",
    )
    return f"""
Task: diagnose whether optimization is stagnating and why.

Decision context:
{_to_json(decision_context)}

Recent history tail:
{_to_json(history_tail)}

{_value_annotation_block(value_annotations)}

{_knowledge_context_block(knowledge_context, knowledge_meta)}

{guidance_block}

Return JSON only for StagnationDiagnosis.
""".strip()


def build_coverage_insight_prompt(
    decision_context: dict[str, Any],
    search_space: list[dict[str, Any]],
    search_space_meta: dict[str, Any] | None = None,
    value_annotations: list[dict[str, Any]] | None = None,
) -> str:
    return f"""
Task: analyze exploration coverage and information gain status.

Decision context:
{_to_json(decision_context)}

Search space:
{_to_json(search_space)}

Search space metadata:
{_to_json(search_space_meta or {})}

{_value_annotation_block(value_annotations)}

Return JSON only for CoverageInsight.
""".strip()


def build_intervention_plan_prompt(
    decision_context: dict[str, Any],
    diagnosis: dict[str, Any],
    hypothesis_action: dict[str, Any],
    coverage_insight: dict[str, Any],
    search_space: list[dict[str, Any]],
    search_space_meta: dict[str, Any] | None = None,
    knowledge_context: list[dict[str, Any]] | None = None,
    knowledge_meta: dict[str, Any] | None = None,
    value_annotations: list[dict[str, Any]] | None = None,
) -> str:
    return f"""
Task: decide sparse high-level intervention plan.

Decision context:
{_to_json(decision_context)}

Stagnation diagnosis:
{_to_json(diagnosis)}

Hypothesis action:
{_to_json(hypothesis_action)}

Coverage insight:
{_to_json(coverage_insight)}

Search space:
{_to_json(search_space)}

Search space metadata:
{_to_json(search_space_meta or {})}

{_value_annotation_block(value_annotations)}

{_knowledge_context_block(knowledge_context, knowledge_meta)}

Guidance:
- BO remains the default proposal engine.
- Intervention should change execution behavior only when justified.
- Allow three modes:
  - bo_direct: execute BO top-1 directly
  - bo_rerank_topk: BO proposes shortlist, then rerank within shortlist
  - bo_focus_then_rerank: first narrow legal pool with focus variables, then rerank shortlist
- Use bo_focus_then_rerank only when diagnosis/coverage indicates local overfocus, low coverage, or repeated scaffold concentration.
- Keep intervention sparse and bounded in rounds.

Return JSON only for InterventionPlan.
""".strip()


def build_controller_plan_prompt(
    decision_context: dict[str, Any],
    diagnosis: dict[str, Any],
    hypothesis_action: dict[str, Any],
    coverage_insight: dict[str, Any],
    search_space: list[dict[str, Any]],
    search_space_meta: dict[str, Any] | None = None,
    controller_trigger_reasons: list[str] | None = None,
    value_annotations: list[dict[str, Any]] | None = None,
    skill_block: str | None = None,
    enable_action_package_v2: bool = False,
    enable_action_package_v06: bool = False,
    admissible_execution_actions: list[str] | None = None,
    preferred_execution_action: str | None = None,
) -> str:
    if enable_action_package_v06:
        policy_block = _instruction_block(
            skill_block,
            f"""
Execution action policy:
- You are deciding how the next unit of experimental budget should be spent.
- Choose exactly one `requested_execution_action` from this admissible set:
  {json.dumps(admissible_execution_actions or [])}
- Preferred default if evidence is mixed:
  {json.dumps(preferred_execution_action or "direct_bo_pick")}
- Execution actions:
  - direct_bo_pick: execute BO direct proposal with no shortlist intervention
  - shape_only_bo_pick: shape shortlist structure, then keep the BO-preferred surviving candidate
  - shape_then_probe_topk: shape shortlist structure, then run a tightly bounded shortlist comparison inside the BO near-top slice
  - shortlist_alt_pick: shape shortlist, then allow shortlist-internal alternative final selection
  - focused_shortlist_alt_pick: use one bounded focused shortlist pass, then allow shortlist-internal alternative final selection
  - finite_pool_candidate_probe: add a few legal finite-pool candidates suggested by observed high-value anchors, then keep final selection inside the normal BO/shortlist path
  - mask_scaffold_corridor_resuggest: before BO proposes the next point, exclude the currently dominant scaffold corridor/slice along the most concentrated scaffold dimension and ask BO to resuggest from a shifted legal pool
  - mask_dominant_resuggest: before BO proposes the next point, exclude the currently dominant repeated scaffold/motif region and ask BO to resuggest from the remaining legal pool
  - mask_low_repeat_resuggest: before BO proposes the next point, exclude high-repeat local regions and ask BO to resuggest toward lower-repeat scaffold coverage
- Fill these fields: requested_execution_action, intent, shortlist_policy, repeat_policy, verification_policy, focus_variables, window_rounds, reasoning.
- If and only if you choose `finite_pool_candidate_probe`, you may also fill `candidate_probe_include` as a short list of `column=value` direction anchors and `candidate_probe_reasoning`.
- intent:
  - exploit: prioritize near-term objective gain
  - probe: spend budget to test a hypothesis or gain information
  - balance: keep both gains and information in play
- shortlist_policy:
  - plain
  - diversity_shape
  - coverage_shape
  - contrast_shape
- repeat_policy:
  - allow
  - avoid_near_duplicate
  - avoid_anchor_repeat
- verification_policy:
  - normal
  - strict

Additional rules:
- Do not propose direct candidate values.
- `shape_only_bo_pick` means shortlist structure should change, but final pick should still stay BO-preferred among surviving candidates.
- `shape_then_probe_topk` is stronger than `shape_only_bo_pick` but still bounded. Use it when shortlist structure alone is not enough and the shortlist-internal final choice among BO near-top candidates is now the bottleneck.
- `finite_pool_candidate_probe` is a candidate-surfacing action, not a direct final pick. Use it when the likely bottleneck is that the finite-pool shortlist is missing plausible recombinations from observed high-value anchors.
- For `finite_pool_candidate_probe`, do not name a final candidate id. Use `candidate_probe_include` to state a small direction, for example fixing a ligand/base pair or reactant/ligand/base pattern that is supported by evidence. Prefer 1-3 anchors unless there is unusually strong evidence for a specific fourth dimension.
- `shortlist_alt_pick` is a rare escalation above shape-only. Use it only when the admissible set includes it and the shortlist-internal final choice itself is directly implicated.
- `focused_shortlist_alt_pick` is a shortlist escalation. Use it only when the admissible set includes it and there is sustained no-improvement plus both coverage pressure and local-lock / scaffold concentration evidence.
- `mask_scaffold_corridor_resuggest`, `mask_dominant_resuggest`, and `mask_low_repeat_resuggest` are pre-shortlist actions. Use them when repeated shape-only shortlist interventions have failed and the issue is the BO proposal distribution itself, not merely final shortlist ranking.
- `mask_scaffold_corridor_resuggest` is the strongest trajectory-shaping action in this set. Use it when the system appears stuck on one scaffold plane/corridor and simply removing an exact dominant scaffold pair is likely too weak.
- If the preferred default is a `mask_*_resuggest`, treat it as stronger than shortlist-level actions. Do not choose `shape_only_bo_pick`, `shortlist_alt_pick`, or `focused_shortlist_alt_pick` instead unless you have explicit contrary evidence that the shortlist-internal final choice, rather than the BO proposal distribution, is the actual bottleneck.
- If you choose `focused_shortlist_alt_pick`, provide 2-3 valid focus_variables and bounded window_rounds.
- If shortlist structure needs cleanup but the final shortlist-internal choice is not the core issue, prefer `shape_only_bo_pick`.
- If the shortlist already contains plausible near-top challengers and the main issue is that BO top-1 keeps winning by default inside the shaped shortlist, prefer `shape_then_probe_topk`.
- If there is strong verification caution, repeated anchor reuse, or local lock but the shortlist-internal final choice is not yet clearly the bottleneck, keep `shape_only_bo_pick`.
- Use `shortlist_alt_pick` only after shape-only style intervention is no longer enough and the shortlist-internal final choice clearly matters.
- If evidence is still healthy or mixed and no shortlist intervention is clearly necessary, prefer `direct_bo_pick`.
""",
        )
        task_line = "Task: choose the v0.6 execution action package for this BO iteration."
    elif enable_action_package_v2:
        policy_block = _instruction_block(
            skill_block,
            """
Action Package policy:
- You are not choosing a raw top-1 override policy. You are choosing how the next unit of experimental budget should be spent.
- Fill all action-package fields: intent, shortlist_policy, repeat_policy, selection_policy, verification_policy, focus_policy, focus_variables, window_rounds, reasoning.
- intent:
  - exploit: prioritize near-term objective gain
  - probe: spend budget to test a hypothesis or gain information
  - balance: keep both gains and information in play
- shortlist_policy:
  - plain: keep shortlist structure mostly unchanged
  - diversity_shape: reduce shortlist redundancy and preserve structural diversity
  - coverage_shape: preserve candidates that improve key-dimension coverage
  - contrast_shape: preserve candidates that test mechanistic or scaffold contrast
- repeat_policy:
  - allow: no extra suppression
  - avoid_near_duplicate: penalize repeated local variants
  - avoid_anchor_repeat: strongly avoid repeating the same local anchor under one-axis lock
- selection_policy:
  - bo_top1: execute the BO top-ranked candidate directly
  - bo_top1_from_shaped_shortlist: BO proposes a shortlist, controller shapes the shortlist structure, then BO keeps the best surviving candidate
  - select_from_shaped_shortlist: BO proposes a shortlist, controller shapes it, then final selection happens inside the shortlist
- verification_policy:
  - normal: standard advisory verification
  - strict: verification should be more cautious and more likely to surface extension risk
- focus_policy:
  - full_space: keep the legal pool broad
  - temporary_focus: for one bounded round, use focus_variables to narrow the legal pool before shortlist selection

Additional rules:
- Do not propose direct candidate values.
- Use temporary_focus only when there is sustained no-improvement plus both low coverage and scaffold concentration / local lock evidence.
- coverage_low alone usually supports coverage_shape or diversity_shape, not temporary_focus.
- In large finite-pool runs, low visited fraction alone is not enough to reject shortlist shaping.
- If the run just improved, local-lock evidence is weak, and the previous verification did not warn about extension risk or attribution uncertainty, prefer `intent=exploit` with `selection_policy=bo_top1`.
- If shortlist structure needs cleanup but BO ranking is still broadly trustworthy, prefer `bo_top1_from_shaped_shortlist` over a full shortlist override.
- If the run is locally locked or repetitive, prefer shortlist-based execution over `bo_top1`; use `select_from_shaped_shortlist` only when the final shortlist choice itself likely matters.
- Do not default to `select_from_shaped_shortlist`; use it when coverage, repetition, or verification evidence says shortlist structure needs intervention and shortlist-internal choice also needs intervention.
- If the last verification warned that the current motif is weak or attribution is uncertain, prefer probe/balance over pure exploit unless evidence clearly improved afterward.
- When remaining budget is very small and you are already penalizing near-duplicates or local lock, prefer `verification_policy=strict` so the final shortlist choice gets an explicit extension-risk check.
""",
        )
        task_line = "Task: choose the controller action package for this BO iteration."
    else:
        policy_block = _instruction_block(
            skill_block,
            """
Mode policy:
- bo_direct: default mode when no meaningful intervention is justified
- bo_rerank_topk: use when BO shortlist likely contains useful diversity and execution choice matters
- bo_focus_then_rerank: use only for a single iteration when there is both local overfocus and a convincing reason to do a short focused probe

Additional rules:
- If choosing bo_focus_then_rerank, provide 2-3 valid focus_variables and a bounded window_rounds value.
- coverage_low by itself should usually prefer bo_rerank_topk, not bo_focus_then_rerank.
- Choose bo_focus_then_rerank only when coverage_low and scaffold_concentration_high are both present and the recent run has shown several rounds without best-value improvement.
- If trigger reasons include sustained stagnation or repeated weak local probing, prefer bo_rerank_topk over bo_direct unless you have a strong concrete reason to keep BO fully direct.
- Treat search scope and execution mode separately: `keep_full_space` often pairs naturally with `bo_rerank_topk`, not `bo_direct`.
- In large finite-pool runs, low visited fraction alone is not a sufficient reason to avoid `bo_rerank_topk`.
- If decision context shows one-axis local sweep, repeated anchor reuse, or strong key-dimension concentration under sustained no-improvement, prefer `bo_rerank_topk` unless there is a concrete reason shortlist execution would be unhelpful.
- Use bo_direct only when you can explicitly justify that controller-side reranking would likely harm search quality more than help it.
- If no strong reason to intervene, choose bo_direct.
- Do not propose direct candidate values.
""",
        )
        task_line = "Task: choose the controller mode for this BO iteration."
    return f"""
{task_line}

Decision context:
{_to_json(decision_context)}

Stagnation diagnosis:
{_to_json(diagnosis)}

Hypothesis action:
{_to_json(hypothesis_action)}

Coverage insight:
{_to_json(coverage_insight)}

Controller trigger reasons:
{_to_json(controller_trigger_reasons or [])}

Search space:
{_to_json(search_space)}

Search space metadata:
{_to_json(search_space_meta or {})}

{_value_annotation_block(value_annotations)}

{policy_block}

Return JSON only for InterventionPlan.
""".strip()


def build_shortlist_rerank_prompt(
    decision_context: dict[str, Any],
    shortlist_candidates: list[dict[str, Any]],
    controller_plan: dict[str, Any],
    search_space_meta: dict[str, Any] | None = None,
    value_annotations: list[dict[str, Any]] | None = None,
    rerank_policy: dict[str, Any] | None = None,
    skill_block: str | None = None,
) -> str:
    policy = rerank_policy or {}
    prompt_style = str(policy.get("prompt_style", "default"))
    state_router_guidance = policy.get("state_router_guidance") or {}
    contrastive_evidence = policy.get("candidate_contrastive_evidence") or {}
    admissible_candidate_indices = list(state_router_guidance.get("admissible_candidate_indices") or [])
    preferred_candidate_indices = list(state_router_guidance.get("preferred_candidate_indices") or [])
    policy_lines = [
        "- Only choose from the shortlist candidate_index values provided.",
    ]
    if state_router_guidance:
        policy_lines.extend(
            [
                f"- Visible evidence state: {state_router_guidance.get('visible_evidence_state', 'unknown')}.",
                f"- Admissible candidate_index values for this turn: {admissible_candidate_indices}. Any other selected_index will be rejected downstream.",
                f"- Preferred candidate_index values when evidence is close: {preferred_candidate_indices}.",
                f"- Fallback candidate_index if your requested pick is invalid or inadmissible: {state_router_guidance.get('fallback_candidate_index', 0)}.",
            ]
        )
    if prompt_style == "challenger_with_incumbent":
        policy_lines.extend(
            [
                "- First compare BO top-1 against the strongest non-top challenger instead of mechanically following BO rank.",
                "- Use overall_score as a useful signal, but do not treat it as the only deciding rule when a challenger has materially stronger transfer or hypothesis evidence.",
                "- When incumbent state is already strong or the run is late-stage, keep BO top-1 unless a challenger has clearly stronger transfer plus hypothesis support.",
                "- If you override BO top-1, the alternative should test a concrete challenger hypothesis, not merely be different.",
            ]
        )
    elif prompt_style == "resuggest_probe_topk":
        policy_lines.extend(
            [
                "- The shortlist already comes from a post-resuggest BO call; treat this as a controlled probe inside the resuggested pool, not as a generic override step.",
                "- Keep the intervention bounded to the strongest near-top part of the resuggested shortlist; do not jump to a far-down candidate just because it is more novel.",
                "- Compare BO top-1 against the best near-top challenger candidates and only leave BO top-1 when the alternative has competitive plausibility plus better transfer or hypothesis value.",
                "- Novelty helps only after plausibility and transfer value are already competitive; do not trade away too much planner support for novelty alone.",
                "- Stay within the admissible shortlist and keep the intervention bounded; this is a probe_topk decision, not an unrestricted rerank.",
            ]
        )
    elif prompt_style == "shape_probe_topk":
        policy_lines.extend(
            [
                "- The shortlist has already been shaped; your job is to compare BO top-1 against the strongest bounded challengers inside the shaped shortlist.",
                "- Keep the comparison inside the admissible near-top slice. Do not jump to a far-down candidate unless the policy explicitly surfaced a deeper diversity challenger.",
                "- Prefer a challenger only when it has competitive plausibility and stronger transfer or hypothesis value than BO top-1.",
                "- If the best challenger is only marginally better, keep BO top-1. If the best challenger is clearly better on transfer and hypothesis value, it is acceptable to leave BO top-1.",
                "- For same-scaffold condition tweaks, use analogue and anchor support as primary evidence; for scaffold shifts, require stronger transfer evidence than for local condition tweaks.",
            ]
        )
    elif prompt_style == "candidate_direction_review":
        policy_lines.extend(
            [
                "- Treat this as a bounded review of BO candidate direction, not as a replacement for BO.",
                "- Keep BO top-1 unless another near-top candidate has clearer same-base, same-ligand, same-scaffold, or analogue support.",
                "- Planner support matters: prefer candidates near the BO top ranks when evidence is otherwise close.",
                "- Downside risk matters more than abstract novelty. Penalize high local_overfit_risk and weak local support.",
                "- Do not reward vague novelty by itself; a non-top pick must express a concrete local calibration or transfer reason.",
            ]
        )
    else:
        policy_lines.extend(
            [
                "- If the shortlist already looks well ranked by BO and there is no strong reason to override, keep the BO top-1 shortlist candidate.",
                "- Use overall_score as the final execution score.",
                "- Only override BO top-1 when the alternative tests a concrete hypothesis, not merely because it is different.",
            ]
        )
    policy_lines.append(
        _instruction_block(
            skill_block,
            """
- Consider plausibility, novelty, transfer value, hypothesis value, and local overfit risk.
- Treat local_overfit_risk as a real penalty on overall_score.
- The shortlist may mix a main_pool of BO top-ranked local probes with a diversity_pool of cross-scaffold comparators.
- If the current issue is overfocus, penalize candidates that are too similar to the recently overused scaffold.
- If BO top-1 belongs to a scaffold that has been heavily revisited recently, it is valid to prefer a cross-scaffold comparator with stronger information gain.
- Use recent_scaffold_hits, recent_primary_dim_hits, recent_secondary_dim_hits, pool_source, and shortlist_source when helpful: lower repeated local exposure is a real advantage when scores are otherwise close.
- Use candidate_contrastive_evidence to compare BO top-1 against the strongest challenger; this evidence is specifically constructed to expose coverage, repetition, and scaffold-contrast differences.
- When two candidates are close, prefer a diversity_pool candidate only if it has stronger transfer or contrastive hypothesis value.
- If all shortlist candidates are very local, keep BO top-1 unless a non-top candidate has a clear mechanistic_contrast or cross_scaffold_transfer hypothesis.
- For every candidate score, fill hypothesis_value_score, structural_shift_type, and hypothesis_summary.
- structural_shift_type must be one of: none, local_refinement, cross_scaffold_transfer, mechanistic_contrast.
- For finite_pool, do not invent or modify candidate values.
""",
        )
    )
    return f"""
Task: rerank a BO shortlist and choose the final executable candidate.

Decision context:
{_to_json(decision_context)}

Controller plan:
{_to_json(controller_plan)}

Shortlist candidates:
{_to_json(shortlist_candidates)}

Search space metadata:
{_to_json(search_space_meta or {})}

Rerank policy state:
{_to_json(policy)}

State router guidance:
{_to_json(state_router_guidance)}

Candidate contrastive evidence:
{_to_json(contrastive_evidence)}

{_value_annotation_block(value_annotations)}

Rerank policy:
{chr(10).join(policy_lines)}

Return JSON only for ShortlistRerankAction.
""".strip()


def build_lab_batch_composition_prompt(
    *,
    decision_context: dict[str, Any],
    candidate_pool: list[dict[str, Any]],
    controller_plan: dict[str, Any],
    diagnosis: dict[str, Any],
    hypothesis_action: dict[str, Any],
    coverage_insight: dict[str, Any],
    batch_size: int,
    search_space: list[dict[str, Any]],
    search_space_meta: dict[str, Any] | None = None,
    reaction_context: dict[str, Any] | None = None,
    value_annotations: list[dict[str, Any]] | None = None,
    knowledge_context: list[dict[str, Any]] | None = None,
    knowledge_meta: dict[str, Any] | None = None,
    skill_block: str | None = None,
) -> str:
    guidance_block = _instruction_block(
        skill_block,
        """
Lab batch composition guidance:
- Treat the batch as one experimental portfolio, not as a ranked top-k list.
- Choose exactly the requested number of candidate_index values when possible.
- Only select candidate_index values from the provided candidate pool; do not invent candidates or edit variable values.
- Keep at least one strong planner-supported anchor unless there is a concrete reason not to.
- Give each selected candidate a distinct role whenever possible, such as planner_anchor, local_refinement, additive_contrast, solvent_contrast, tempo_probe, coverage_probe, evidence_guided_probe, risk_check, or confirmation.
- If candidate_pool entries contain descriptor_profile or descriptor_contrast_to_anchor, use them to distinguish descriptor_contrast from simple categorical changes.
- If candidate_pool entries contain proposed_by, it records which acquisition function proposed the candidate (ei = expected improvement, ucb = upper confidence bound, ei+ucb = both).
- The role and rationale for each slot must explain why this candidate deserves one real experiment in this batch and how it complements the other selected candidates.
- Avoid duplicate candidates and avoid near-duplicate Additive-Solvent or Catalyst-Solvent pairs unless the slot is explicitly a local refinement or confirmation.
- Use scoped evidence only as advisory support. Do not infer hidden outcomes or benchmark oracle values.
""",
    )
    return f"""
Task: design one real-lab experimental batch as a bounded TRACE portfolio.

The planner has provided an admissible candidate pool. Your job is to design the whole batch at once: define the batch-level strategy, assign roles to the selected candidates, and explain why this set is a useful experimental portfolio under the current budget.

Requested batch size:
{int(batch_size)}

Decision context:
{_to_json(decision_context)}

Stagnation diagnosis:
{_to_json(diagnosis)}

Hypothesis action:
{_to_json(hypothesis_action)}

Coverage insight:
{_to_json(coverage_insight)}

Controller plan:
{_to_json(controller_plan)}

Candidate pool:
{_to_json(candidate_pool)}

Reaction context:
{_to_json(reaction_context or {})}

Search space:
{_to_json(search_space)}

Search space metadata:
{_to_json(search_space_meta or {})}

{_value_annotation_block(value_annotations)}

{_knowledge_context_block(knowledge_context, knowledge_meta)}

{guidance_block}

Return JSON only for LabBatchCompositionAction.
""".strip()


def build_feasibility_action_prompt(
    candidate_condition: dict[str, Any],
    reaction_context: dict[str, Any],
    decision_context: dict[str, Any],
    value_annotations: list[dict[str, Any]] | None = None,
) -> str:
    return f"""
Task: assess semantic feasibility and choose action.

Proposed experiment:
{_to_json(candidate_condition)}

Known reaction context:
{_to_json(reaction_context)}

Decision context:
{_to_json(decision_context)}

{_value_annotation_block(value_annotations)}

Action policy:
- Default to "accept" when candidate is executable and within known bounds.
- Use "revise" only for concrete issues (safety/range/inconsistency) and provide minimally changed revised_candidate.
- If no concrete issue is found, do not revise.

Return JSON only for FeasibilityAction.
""".strip()


def build_semantic_assessment_prompt(
    candidate_condition: dict[str, Any],
    reaction_context: dict[str, Any],
    decision_context: dict[str, Any],
    knowledge_context: list[dict[str, Any]] | None = None,
    knowledge_meta: dict[str, Any] | None = None,
    value_annotations: list[dict[str, Any]] | None = None,
    skill_block: str | None = None,
) -> str:
    guidance_block = _instruction_block(
        skill_block,
        """
Guidance:
- Default behavior is soft guidance, not hard rejection.
- Score plausibility and novelty, and provide concise risk explanation.
""",
    )
    return f"""
Task: provide semantic soft assessment for a BO candidate.

Proposed experiment:
{_to_json(candidate_condition)}

Reaction context:
{_to_json(reaction_context)}

Decision context:
{_to_json(decision_context)}

{_value_annotation_block(value_annotations)}

{_knowledge_context_block(knowledge_context, knowledge_meta)}

{guidance_block}

Return JSON only for SemanticAssessment.
""".strip()


def build_candidate_verification_prompt(
    candidate: dict[str, Any],
    reaction_context: dict[str, Any],
    decision_context: dict[str, Any],
    semantic_assessment: dict[str, Any],
    controller_plan: dict[str, Any],
    knowledge_context: list[dict[str, Any]] | None = None,
    knowledge_meta: dict[str, Any] | None = None,
    value_annotations: list[dict[str, Any]] | None = None,
) -> str:
    return f"""
Task: run a verification pass on the selected candidate and its semantic assessment.

Candidate:
{_to_json(candidate)}

Reaction context:
{_to_json(reaction_context)}

Decision context:
{_to_json(decision_context)}

Semantic assessment:
{_to_json(semantic_assessment)}

Controller plan:
{_to_json(controller_plan)}

{_value_annotation_block(value_annotations)}

{_knowledge_context_block(knowledge_context, knowledge_meta)}

Verification rules:
- Do NOT replace the candidate.
- Check for missing caveats, unsupported confidence, or internal inconsistency.
- Prefer "caution" over "fail_soft" unless there is a clear and concrete concern.
- This is a verifier, not a proposal engine.
- If the controller requests strict verification, be more willing to surface extension-risk or identity-attribution caveats.

Return JSON only for VerificationPass.
""".strip()


def build_reflection_action_prompt(
    decision_action: dict[str, Any],
    candidate: dict[str, Any],
    feasibility_action: dict[str, Any],
    result: float,
    decision_context: dict[str, Any],
    knowledge_context: list[dict[str, Any]] | None = None,
    knowledge_meta: dict[str, Any] | None = None,
    value_annotations: list[dict[str, Any]] | None = None,
) -> str:
    return f"""
Task: generate reflection memo for next step decision.

DecisionAction:
{_to_json(decision_action)}

FeasibilityAction:
{_to_json(feasibility_action)}

Executed candidate:
{_to_json(candidate)}

Actual result:
{result}

Decision context snapshot:
{_to_json(decision_context)}

{_value_annotation_block(value_annotations)}

{_knowledge_context_block(knowledge_context, knowledge_meta)}

Return JSON only for ReflectionAction.
""".strip()

REACTION_TYPE_CONSTRAINTS: dict[str, str] = {
    "suzuki_miyaura": """
- In each candidate, the electrophile (halide/triflate/leaving-group partner) and nucleophile (boron species: boronic acid, boronate ester, or trifluoroborate) must be DIFFERENT reaction partners.
- NEVER assign a boron-containing species (BPin, BF3K, Boronic Acid, Boronate) to the electrophile slot.
- NEVER assign a halide/triflate/leaving-group species to the nucleophile slot.
- Use the sample pool above to verify what valid (electrophile, nucleophile) pairings look like — only choose combinations that appear chemically consistent with those examples.
""".strip(),
    "buchwald_hartwig_amination": """
- In each candidate, Reactant2 is the aryl halide / electrophile partner and must stay in that role.
- Ligand must be treated as a catalyst-supporting ligand choice, not as a substrate.
- Base and Additive must be interpreted as condition variables; do not treat them as interchangeable with substrate roles.
- Use the sample pool above to verify what valid (Reactant2, Ligand, Base, Additive) combinations look like — only choose combinations that appear chemically consistent with those examples.
""".strip(),
    "c_h_arylation": """
- In each candidate, Aryl_halide_SMILES is the aryl halide electrophile and must stay in that role.
- Additive_SMILES is the heteroarene/additive partner used in this C-H arylation benchmark; do not treat it as a solvent or base.
- Base_SMILES is the base condition and Ligand_SMILES is the phosphine ligand condition.
- Keep all executable candidate values exactly as listed in the valid values. Translation annotations are only interpretation aids.
- Use the sample pool above to verify what valid (Aryl_halide_SMILES, Additive_SMILES, Base_SMILES, Ligand_SMILES) combinations look like.
""".strip(),
    "generic": "Use the sample pool above to verify what valid combinations look like and only choose combinations that appear chemically consistent with those examples.",
}


def _build_reaction_constraint_block(reaction_type: str) -> str:
    return REACTION_TYPE_CONSTRAINTS.get(reaction_type, REACTION_TYPE_CONSTRAINTS["generic"])


def _build_small_scaffold_guidance(
    scaffold_dims: list[str],
    valid_values: dict[str, list[str]],
) -> str:
    lines: list[str] = []
    for dim in scaffold_dims:
        values = [str(v) for v in valid_values.get(dim, [])]
        if 1 < len(values) <= 5:
            joined = ", ".join(values)
            lines.append(
                f"- For scaffold dimension `{dim}` with only {len(values)} values [{joined}], explicitly rank all values by your chemistry prior before designing candidates. Allocate more of the {len(values)} initial slots to the values you judge more promising, while still keeping coverage across the major scaffold values."
            )
    if not lines:
        return "- No special small-scaffold budget allocation rule is needed."
    return "\n".join(lines)


def _build_column_role_notes(
    feature_columns: list[str],
    valid_values: dict[str, list[str]],
    reaction_type: str = "generic",
) -> str:
    """Generate human-readable role annotations for each feature column to prevent LLM role confusion."""
    boron_keywords = {"bpin", "bf3k", "boronic", "boronate", "borate"}
    halide_keywords = {"cl", "br", "i-", "otf", "triflate", "iodide", "chloride", "bromide", "leaving"}
    reaction_type_lower = str(reaction_type or "").lower()
    lines = []
    for col in feature_columns:
        vals = valid_values.get(col, [])
        val_sample = ", ".join(str(v) for v in vals[:5])
        # Heuristically annotate known reaction roles
        col_lower = col.lower()
        if "aryl_halide" in col_lower:
            role = "ARYL HALIDE / ELECTROPHILE - the aryl halide coupling partner for oxidative addition."
        elif "electrophile" in col_lower:
            role = "ELECTROPHILE — the halide/triflate/leaving-group coupling partner (e.g., aryl chloride, aryl iodide, aryl triflate). Must NOT be a boron species."
        elif "nucleophile" in col_lower:
            role = "NUCLEOPHILE — the boron-containing transmetalation partner (e.g., boronic acid, pinacol boronate [BPin], trifluoroborate [BF3K]). Must NOT be a halide/triflate."
        elif "catalyst" in col_lower or "pd" in col_lower:
            role = "CATALYST — the palladium precatalyst."
        elif "ligand" in col_lower:
            role = "LIGAND — phosphine or other ancillary ligand controlling catalyst geometry and reactivity."
        elif "base" in col_lower:
            role = "BASE — stoichiometric base for transmetalation/deprotonation."
        elif "additive" in col_lower and reaction_type_lower == "c_h_arylation":
            role = "ADDITIVE / HETEROARENE PARTNER - benchmark additive/reactant partner for the C-H arylation space."
        elif "additive" in col_lower:
            role = "ADDITIVE - condition variable or additive; interpret using dataset-specific constraints."
        elif "solvent" in col_lower:
            role = "SOLVENT — reaction medium."
        else:
            role = f"VARIABLE — ({len(vals)} options)."
        lines.append(f"  {col}: {role}\n    First few valid values: [{val_sample}]")
    return "\n".join(lines)


def build_init_design_prompt(
    search_space_meta: dict[str, Any],
    dataset_meta: dict[str, Any],
    init_budget: int,
    sample_pool: list[dict[str, Any]],
    knowledge_context: list[dict[str, Any]] | None = None,
    knowledge_meta: dict[str, Any] | None = None,
    value_annotations: list[dict[str, Any]] | None = None,
    skill_block: str | None = None,
) -> str:
    feature_columns = search_space_meta.get("feature_columns", [])
    scaffold_dims = search_space_meta.get("scaffold_dims", [])
    valid_values = search_space_meta.get("valid_values_per_col", {})
    description = dataset_meta.get("description", "Chemical reaction optimization.")
    candidate_count = dataset_meta.get("candidate_count", "unknown")
    reaction_type = str(dataset_meta.get("reaction_type", "generic") or "generic")
    column_role_notes = _build_column_role_notes(feature_columns, valid_values, reaction_type)
    constraint_block = _build_reaction_constraint_block(reaction_type)
    small_scaffold_guidance = _build_small_scaffold_guidance(scaffold_dims, valid_values)
    requirement_block = _instruction_block(
        str(skill_block or "").format(init_budget=init_budget),
        f"""
Requirements:
1. Return exactly {init_budget} candidates.
2. Every candidate must use only values listed in "Valid values per column" above.
3. Cover ALL major values of scaffold_dims as much as the budget allows; if full one-shot coverage is impossible, prioritize broad scaffold coverage first.
4. For any scaffold dimension with very few values (<=5), explicitly rank the values by chemistry prior and allocate more slots to the values you judge more promising, while avoiding total collapse onto a single value unless the chemistry signal is overwhelming.
5. Vary non-scaffold condition variables to cover different mechanistic classes when such variables exist.
6. Prefer combinations you find chemically plausible, but do not invent values not in the valid list.
7. For each candidate provide a brief chemical rationale.
""",
    )
    return f"""
Task: design {init_budget} initial experiments for a Bayesian Optimization run.

Reaction description:
{description}

Reaction type:
{reaction_type}

Total candidate pool size: {candidate_count}

Feature columns and their chemical roles:
{column_role_notes}

Scaffold dimensions (most influential for diversity):
{_to_json(scaffold_dims)}

Valid values per column (ALL values are from the real candidate pool):
{_to_json(valid_values)}

{_value_annotation_block(value_annotations)}

Sample pool (representative valid combinations for reference):
{_to_json(sample_pool[:12])}

{_knowledge_context_block(knowledge_context, knowledge_meta)}

Reaction-specific chemical constraints:
{constraint_block}

Small-scaffold allocation guidance:
{small_scaffold_guidance}

{requirement_block}

IMPORTANT - output format for each element in "designed_experiments":
  - "assignments": a list of strings, one per feature column, formatted exactly as "column_name=value".
    Use the exact column names from "Feature columns" above. Use the exact values from "Valid values per column".
    Example for {feature_columns}:
      ["electrophile=1d, 6-I-Q", "nucleophile=2a, Boronic Acid", "catalyst=Pd(OAc)2", "ligand=P(tBu)3", "base=NaHCO3", "solvent=THF"]
  - "rationale": a string explaining the chemical reasoning.

Return JSON only for InitDesignAction.
""".strip()


def build_search_constraint_prompt(
    decision_context: dict[str, Any],
    search_space_meta: dict[str, Any],
    history_tail: list[dict[str, Any]],
    value_annotations: list[dict[str, Any]] | None = None,
    skill_block: str | None = None,
) -> str:
    feature_columns = search_space_meta.get("feature_columns", [])
    scaffold_dims = search_space_meta.get("scaffold_dims", [])
    valid_values = search_space_meta.get("valid_values_per_col", {})
    best_obs = decision_context.get("best_observation", {})
    best_val = decision_context.get("best_value", None)
    no_improve = decision_context.get("no_improvement_rounds", 0)
    scaffold_conc = decision_context.get("scaffold_concentration", {})
    instruction_block = _instruction_block(
        skill_block,
        """
Instructions:
- Generate constraints ONLY when you have clear chemical evidence supporting them.
- Each constraint targets ONE variable with either "include_values" (restrict to subset) or "exclude_values" (block subset).
- Values in constraints MUST appear in the "valid values per column" list above.
- Limit to at most 2-3 constrained variables per update.
- If no_improvement_rounds < 3 or there is no strong chemical evidence, return an empty constraints list.
- Include duration_rounds (how many BO rounds this constraint should stay active, 3-8 typical).
- If constraints would reduce the candidate pool below ~5%, set retain_full_space_fallback: true.
""",
    )
    return f"""
Task: decide whether to add include/exclude constraints on the BO candidate pool.

Current best result: {best_val}
Best observation: {_to_json(best_obs)}
Rounds without improvement: {no_improve}
Recent scaffold concentration: {_to_json(scaffold_conc)}

Recent experiment history (last rounds):
{_to_json(history_tail)}

Search space feature columns:
{_to_json(feature_columns)}

Scaffold dimensions (most influential for diversity):
{_to_json(scaffold_dims)}

Valid values per column (ONLY use these values in your constraints):
{_to_json(valid_values)}

{_value_annotation_block(value_annotations)}

Full decision context:
{_to_json(decision_context)}

{instruction_block}

Return JSON only for LLMSearchConstraintAction.
""".strip()
