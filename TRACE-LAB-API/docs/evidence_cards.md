# Evidence Cards

Evidence cards are project-local scoped advisory inputs.  They are not hidden
outcomes, candidate lookup tables, or full-paper RAG chunks.

Example:

```json
{
  "card_id": "lit_001",
  "source": "paper or note",
  "summary": "Short scoped observation.",
  "reaction_scope": "oxidative esterification",
  "variable_scope": ["Additive"],
  "target_nodes": ["hypothesis_action", "lab_batch_composition"],
  "mapping_status": "same_reaction_family",
  "confidence": "medium",
  "allowed_use": "advisory",
  "source_type": "literature",
  "supporting_excerpt": "short source-grounded excerpt",
  "transferability_note": "why this source can and cannot transfer",
  "leakage_risk": "clean_literature_prior"
}
```

Supported `mapping_status` values:

- `direct`
- `same_start_end`
- `same_reaction_family`
- `variable_level`
- `background`
- `out_of_scope`

Cards marked `out_of_scope`, `blocked`, `do_not_use`, `oracle`, or with
candidate-lookup leakage risk are not exposed to the controller or UI review.

