---
id: init-design-core
name: init-design-core
version: "1.0"
description: Initial design procedure for finite-pool BO tasks.
target_nodes:
  - design_init_experiments
---
Requirements:
1. Return exactly {init_budget} candidates.
2. Every candidate must use only values listed in "Valid values per column" above.
3. Cover ALL major values of scaffold_dims as much as the budget allows; if full one-shot coverage is impossible, prioritize broad scaffold coverage first.
4. For any scaffold dimension with very few values (<=5), explicitly rank the values by chemistry prior and allocate more slots to the values you judge more promising, while avoiding total collapse onto a single value unless the chemistry signal is overwhelming.
5. Vary non-scaffold condition variables to cover different mechanistic classes when such variables exist.
6. Prefer combinations you find chemically plausible, but do not invent values not in the valid list.
7. For each candidate provide a brief chemical rationale.
8. Preserve some interpretability in the init slate: do not make every candidate orthogonal on every axis if that would leave later diagnosis with only bundled multi-axis changes.
9. Balance novelty with one or two partially anchored comparisons so early reflection can reason from more than isolated one-off points.
