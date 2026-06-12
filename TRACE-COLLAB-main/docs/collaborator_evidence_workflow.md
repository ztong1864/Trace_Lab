# Collaborator Evidence Workflow

TRACE Lab can run immediately with the included project-local
`evidence_cards.jsonl`.  The workflow below is for refreshing or extending the
knowledge/evidence cards when new literature arrives.

## Current Principle

Literature does not need to exactly match the target design-space fields.  Each
piece of evidence is treated as scoped advisory context and must declare how it
maps to the current reaction:

- `direct`: directly matches a variable/candidate in the current project.
- `same_start_end`: same start/end transformation but different condition space.
- `same_reaction_family`: same broad chemistry family.
- `variable_level`: useful for one variable class only.
- `background`: mechanism/trend context.
- `out_of_scope`: not exposed to the controller.

The controller must not see hidden outcomes, candidate-level lookup tables, or
unreviewed oracle-like claims.

## Lightweight Local PDF Pipeline

Install `pdftotext` and `pdfinfo` if they are not available.  On macOS:

```bash
brew install poppler
```

Build a manifest from the local PDFs:

```bash
python experiments/knowledge_curation/build_lab_pdf_manifest.py \
  --source-dir collaborator_materials/literature/Homogeneous\ metal-catalyzed \
  --output work/evidence/pdf_manifest.jsonl \
  --reaction-scope "oxidative esterification"
```

Generate project-specific extraction questions:

```bash
python experiments/knowledge_curation/generate_lab_evidence_questions.py \
  --project-dir runs/lab_projects/oxidative_esterification_demo \
  --output work/evidence/questions.jsonl
```

Extract candidate evidence snippets from PDF text:

```bash
python experiments/knowledge_curation/extract_lab_evidence_from_pdf_text.py \
  --questions work/evidence/questions.jsonl \
  --pdf-manifest work/evidence/pdf_manifest.jsonl \
  --output work/evidence/pilot_evidence_items.jsonl
```

Review `work/evidence/pilot_evidence_items.jsonl` manually.  Remove weak or
out-of-scope items, adjust mapping status if needed, and make sure no source is
being used as a hidden result table.

Publish reviewed cards into the lab project:

```bash
python experiments/knowledge_curation/publish_lab_evidence_cards.py \
  --questions work/evidence/questions.jsonl \
  --evidence work/evidence/pilot_evidence_items.jsonl \
  --output runs/lab_projects/oxidative_esterification_demo/evidence_cards.jsonl \
  --reaction-scope "oxidative esterification" \
  --default-mapping-status same_reaction_family
```

Restart or reload TRACE Lab.  The UI evidence panel will show which evidence
cards support the current batch and recommendation roles.

## Human Review Boundary

The scripts above are deliberately conservative.  They create review-needed
cards; they do not replace chemistry review.  Codex, Claude, or another
assistant can be used to speed up summarization and mapping review, but the
final `evidence_cards.jsonl` should be checked by a human before lab use.

