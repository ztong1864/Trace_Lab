---
id: controller-plan-core
name: controller-plan-core
version: "2.1"
description: Sparse controller execution-action policy.
target_nodes:
  - controller_plan
---
Execution Action policy:
- Choose how the next unit of experimental budget should be spent, not a raw top-1 rewrite.
- First choose `requested_execution_action`.
- Then fill intent, shortlist_policy, repeat_policy, verification_policy, focus_variables, window_rounds, reasoning.
- Execution actions:
  - `direct_bo_pick`: BO proposes and BO top-1 executes directly
  - `shape_only_bo_pick`: controller shapes shortlist structure, but BO-preferred surviving candidate still executes
  - `shortlist_alt_pick`: controller shapes shortlist, then selects an alternative inside that shortlist
  - `focused_shortlist_alt_pick`: one bounded focused shortlist round, then select an alternative inside that shortlist
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
- verification_policy:
  - normal: standard advisory verification
  - strict: verification should surface extension-risk and attribution-risk caveats more aggressively

Additional rules:
- `focused_shortlist_alt_pick` is only appropriate when there is sustained no-improvement plus both low coverage and scaffold concentration / local lock evidence.
- coverage_low alone usually supports `shape_only_bo_pick` or `shortlist_alt_pick`, not focused execution.
- If there is sustained no-improvement plus one-axis local probing, repeated anchor reuse, or strong key-dimension concentration, prefer shortlist actions unless shortlist execution is concretely unlikely to help.
- Prefer `shape_only_bo_pick` when shortlist structure needs repair but evidence for replacing BO's surviving top candidate is still weak.
- If the last verification warned that the current motif is weak or attribution is uncertain, prefer probe/balance over pure exploit unless the evidence clearly improved afterward.
- When remaining budget is very small and you are already penalizing near-duplicates or local lock, prefer `verification_policy=strict` so the final shortlist choice gets an explicit extension-risk check.
- Do not propose direct candidate values.
