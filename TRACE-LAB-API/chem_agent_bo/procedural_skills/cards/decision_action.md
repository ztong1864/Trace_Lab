---
id: decision-action-core
name: decision-action-core
version: "1.0"
description: Core procedural guidance for selecting stage and active variables.
target_nodes:
  - decision_action
---
Decision guidance:
- Allowed fixed_variables_strategy: "llm_proposed", "best_observed", "fallback_default", "mixed".
- If best_improvement_last_3 is small and current_stage_streak is high, change either stage or active_variables pattern.
- Keep active_variables valid and concise; prefer 2-4 variables.
