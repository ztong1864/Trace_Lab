# Evidence Cards From Papers

TRACE Lab can run with a project's own `evidence_cards.jsonl`. This page is for
building or extending those cards from papers you already have as PDFs.

There is no model training here. A card is one extracted, reviewed claim with the
quote it came from. The work is split so that the parts that can be wrong are
checked by code, and the judgement calls stay with the chemist:

| Step | Who | Command (from `TRACE-LAB-API/`) |
| --- | --- | --- |
| 1. Extract page text and the project's context | TRACE | `experiments/run_lab_bo.py evidence-prepare` |
| 2. Draft findings with verbatim quotes | an assistant (Claude Code, the tool-skills agent) | writes `evidence_work/drafts/*.jsonl` |
| 3. Check quotes, numbers and variable names against the paper | TRACE | `evidence-verify` |
| 4. Write a review sheet | TRACE | `evidence-sheet` |
| 5. Accept or reject each row, edit wording | the chemist | `evidence_work/review_sheet.csv` in Excel |
| 6. Import the accepted rows | TRACE | `evidence-accept` |
| 7. See what the controller will actually be shown | TRACE | `evidence-preview` |

The drafting rules for the assistant are in
[`trace-lab-skill/references/evidence-drafting.md`](../../trace-lab-skill/references/evidence-drafting.md).

## Example

```bash
cd TRACE-LAB-API
export PYTHONPATH=".;third_party/atlas/src;third_party/olympus/src"
P=runs/lab_projects/my_project

python experiments/run_lab_bo.py evidence-prepare --project-dir $P --pdf-dir path/to/papers
#   -> $P/evidence_work/packets/<paper>_pNN-MM.txt, context.json, sources.jsonl
#   The assistant reads packets/ and context.json and writes $P/evidence_work/drafts/*.jsonl
python experiments/run_lab_bo.py evidence-verify  --project-dir $P
python experiments/run_lab_bo.py evidence-sheet   --project-dir $P
#   Give $P/evidence_work/review_sheet.csv to the chemist: fill `decision` with accept / reject.
python experiments/run_lab_bo.py evidence-accept  --project-dir $P
python experiments/run_lab_bo.py evidence-preview --project-dir $P
```

`pdftotext` (poppler) must be installed, or `PDFTOTEXT` set to its path. It is
already on the PATH in Git Bash. Work files live in `<project>/evidence_work/`;
only `evidence-accept` changes the project's own evidence file (the old one is
backed up under `_backups/`).

## What the checks do, and what they don't

`evidence-verify` reads every draft and, for each one:

- finds the quote in the paper (ignoring spacing, dashes, sub/superscripts and
  line-break hyphenation). An approximate match is accepted with a warning only if
  every digit in the quote is present, so a changed yield is never treated as a typo;
  a wrong page number is corrected;
- requires every percentage in the summary to be in the quote;
- requires `variable_scope` to name real design variables;
- lets drafts propose only `same_reaction_family`, `variable_level`, `background` or
  `out_of_scope`. `direct` and `same_start_end` are the chemist's call and are
  downgraded on drafts;
- flags duplicates, existing cards from the same paper, and quotes that read like a
  per-candidate result table.

It **cannot** prove that a yield belongs to the condition the summary names, and
some PDF tables come out of the text layer with a column detached from its rows.
That is why the review sheet puts the quote next to the summary and why the chemist,
not the code, decides.

`evidence-accept` re-checks every accepted row (an edited quote must be in the paper word
for word, since approximate matching is only tolerated for a drafted quote whose warning was
on the sheet; an edited summary's percentages must still be in the quote) and refuses the
whole sheet if any row has a problem or a `decision` is mistyped.

## Will a new card be shown? Check with `evidence-preview`

Each `ask` retrieves evidence once, for all controller nodes together, and keeps the
top few cards (`knowledge_top_k`, default 5). Those same cards serve every node in
that ask.

**Ranking.** With the default setting `evidence_selection: score`, the score is roughly
`status score + 10 x confidence + 4 x variables named + 3 x nodes matched`. A precise
card that names one or two variables can therefore rank well below broad cards that
name six or seven, and may never be shown. Setting `evidence_selection: per_variable`
(`config --set evidence_selection=per_variable`, `PATCH /config`, or the web agent's
`update_project_config`) first gives each optimized variable its most specific eligible
card (mapping status first, then the card naming the fewest optimized variables, then
confidence), then fills the remaining places by score. Each variable prefers a card no other
variable has taken, so a single broad card cannot occupy every place; it is reused only for a
variable it is the sole candidate of. Because specificity outranks
confidence there, use it only once the chemist has confirmed the `confidence` and
`variable_scope` of the reviewed cards. It needs `knowledge_top_k` (orchestrator) and
`decision_engine_knowledge_max_items` (prompt) in the agent config to be at least the
number of optimized variables; the preview warns when they are not.

**Scope rule.** A card with `direct` / `same_start_end` status is hidden when its
`reaction_scope` is unrelated to the project's: neither one contains the other nor do
they share two key words (for example "oxidative" and "lactonization"). The rule reads
words, not chemistry, so a neighbouring reaction can pass; the card's status is still
the chemist's claim. The rank bonus for a matching scope stays strict.

**What the agent reads.** The decision engine keeps only the first
`decision_engine_knowledge_max_chars` (default 400) characters of each card, in the order
summary, mapping status line, transferability note, excerpt, and reads at most
`decision_engine_knowledge_max_items` (default 5) cards. The summary is the only part
that is always seen; the note is mostly cut and the excerpt rarely visible. Put the
finding first and keep the summary near 300 characters (`evidence-verify` warns with
`summary_long`).

`evidence-preview` shows all of this: the selection mode and limits, each retrieved
card with the variable it was picked for, whether the agent reads it, the exact text it
reads and how many characters are cut, the best card that missed the cut and its rank,
every card that is screened out with the reason, and warnings (limits that do not fit,
identical summaries taking several places). `project check` also warns about cards that
can never be retrieved and about cards with identical summaries.

## The older scripts

`experiments/knowledge_curation/` (`build_lab_pdf_manifest.py`,
`generate_lab_evidence_questions.py`, `extract_lab_evidence_from_pdf_text.py`,
`publish_lab_evidence_cards.py`) is a keyword-based pilot: it picks one paragraph per
fixed question and writes a templated claim, and `publish_lab_evidence_cards.py`
overwrites the evidence file without validation or backup. It is superseded by the
workflow above and kept for reference.
