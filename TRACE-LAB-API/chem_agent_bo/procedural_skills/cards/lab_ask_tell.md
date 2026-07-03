---
id: lab_ask_tell
name: Real-lab ask/tell controller guardrails
description: Applies TRACE action reasoning to real-lab batch recommendations without oracle access.
version: "1.0"
target_nodes:
  - controller_plan
  - lab_batch_composition
  - shortlist_rerank
  - semantic_assessment
  - verification_pass
  - reflection_action
---

Use real-lab ask/tell constraints.

- Treat each recommendation as pending until a measured result is supplied by the experimenter.
- Do not assume candidate outcomes, oracle summaries, or hidden benchmark values.
- Prefer actions that improve batch usefulness under the same experimental budget: keep a strong planner anchor, add controlled coverage when the history is narrow, and avoid duplicate or near-duplicate pending conditions.
- Use scoped evidence as advisory support, not as a substitute for measured results.
- When reflecting after `tell`, distinguish measured success/failure from the earlier recommendation rationale.
