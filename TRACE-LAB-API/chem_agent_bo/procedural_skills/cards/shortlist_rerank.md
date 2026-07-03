---
id: shortlist-rerank-core
name: shortlist-rerank-core
version: "2.0"
description: Shared shortlist reranking policy.
target_nodes:
  - shortlist_rerank
---
- Consider plausibility, novelty, transfer value, hypothesis value, and local overfit risk.
- Treat local_overfit_risk as a real penalty on overall_score.
- Respect the controller action package. The shortlist may already be shaped for probe / exploit balance.
- The shortlist may mix a main_pool of BO top-ranked local probes with a diversity_pool of cross-scaffold comparators.
- If the current issue is overfocus, penalize candidates that are too similar to the recently overused scaffold.
- If BO top-1 belongs to a scaffold that has been heavily revisited recently, it is valid to prefer a cross-scaffold comparator with stronger information gain.
- Use recent_scaffold_hits, recent_primary_dim_hits, recent_secondary_dim_hits, pool_source, and shortlist_source when helpful: lower repeated local exposure is a real advantage when scores are otherwise close.
- When two candidates are close, prefer a diversity_pool candidate only if it has stronger transfer or contrastive hypothesis value.
- If controller intent is `probe`, favor candidates with better information gain or contrastive value when the overall score gap is small.
- If controller intent is `exploit`, keep BO-favored local candidates unless an alternative has clearly stronger transfer plus hypothesis value.
- If all shortlist candidates are very local, keep BO top-1 unless a non-top candidate has a clear mechanistic_contrast or cross_scaffold_transfer hypothesis.
- For every candidate score, fill hypothesis_value_score, structural_shift_type, and hypothesis_summary.
- structural_shift_type must be one of: none, local_refinement, cross_scaffold_transfer, mechanistic_contrast.
- For finite_pool, do not invent or modify candidate values.
