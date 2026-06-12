---
id: stagnation-diagnosis-core
name: stagnation-diagnosis-core
version: "1.0"
description: Stagnation diagnosis rubric.
target_nodes:
  - stagnation_diagnosis
---
Guidance:
- Explain causes (e.g., local trapping, repeated patterns, low coverage).
- Recommend high-level intervention direction rather than direct candidate rewriting.
- In large finite-pool benchmarks, low visited fraction alone does not mean controller intervention is unjustified.
- Distinguish search scope from execution mode: `keep_full_space` can still support shortlist reranking.
- Sustained no-improvement plus one-axis local probing or repeated anchor reuse should be treated as actionable local lock, even when the full pool is still mostly unexplored.
