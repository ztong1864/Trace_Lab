---
id: search-constraint-core
name: search-constraint-core
version: "1.0"
description: Rules for generating LLM search constraints.
target_nodes:
  - generate_search_constraints
---
Instructions:
- Generate constraints ONLY when you have clear chemical evidence supporting them.
- Each constraint targets ONE variable with either "include_values" (restrict to subset) or "exclude_values" (block subset).
- Values in constraints MUST appear in the "valid values per column" list above.
- Limit to at most 2-3 constrained variables per update.
- If no_improvement_rounds < 3 or there is no strong chemical evidence, return an empty constraints list.
- Include duration_rounds (how many BO rounds this constraint should stay active, 3-8 typical).
- If constraints would reduce the candidate pool below ~5%, set retain_full_space_fallback: true.
