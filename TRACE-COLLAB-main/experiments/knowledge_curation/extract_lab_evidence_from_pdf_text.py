#!/usr/bin/env python
"""Extract lightweight question-grounded lab evidence from local PDF text.

This deterministic extractor is intended for a small pilot loop. It produces
review-needed evidence items without calling an LLM; stronger extraction can be
run later with extract_evidence_from_pdfs.py --mode pdf_text_llm.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.knowledge_curation.source_acquisition.schema import read_questions


@dataclass
class LabEvidenceItem:
    evidence_id: str
    question_id: str
    source_id: str
    provider: str
    source_type: str
    title: str
    url: str
    doi: str
    claim: str
    supporting_excerpt: str
    bo_relevance: str
    confidence: float
    needs_review: bool
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--pdf-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--max-pdfs", type=int, default=0)
    parser.add_argument("--max-evidence-per-pdf", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = read_questions(_resolve(args.questions))
    manifest = _load_jsonl(_resolve(args.pdf_manifest))
    requested_sources = set(args.source_id or [])
    records: list[dict[str, Any]] = []
    pdf_count = 0
    for pdf_record in manifest:
        if pdf_record.get("status") != "downloaded":
            continue
        source_id = str(pdf_record.get("source_id") or pdf_record.get("pdf_id") or "")
        if requested_sources and source_id not in requested_sources:
            continue
        if args.max_pdfs and pdf_count >= int(args.max_pdfs):
            break
        pdf_count += 1
        text = _pdf_text(Path(str(pdf_record.get("local_path") or "")))
        if not text:
            continue
        for question in questions:
            excerpt = _select_excerpt(text, question)
            if not excerpt:
                continue
            item = _evidence_item(
                question=question,
                pdf_record=pdf_record,
                excerpt=excerpt,
            )
            records.append(asdict(item))
            if _items_for_source(records, source_id) >= int(args.max_evidence_per_pdf):
                break
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(records)} lab evidence items to {output}")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _pdf_text(path: Path) -> str:
    if not path.exists():
        return ""
    completed = subprocess.run(
        ["pdftotext", str(path), "-"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        return ""
    return re.sub(r"[ \t]+", " ", completed.stdout)


def _select_excerpt(text: str, question: dict[str, Any]) -> str:
    keywords = _keywords(question)
    paragraphs = [para.strip() for para in re.split(r"\n\s*\n", text) if para.strip()]
    scored = []
    for idx, paragraph in enumerate(paragraphs):
        if _reference_like(paragraph):
            continue
        lowered = paragraph.lower()
        score = sum(lowered.count(keyword) for keyword in keywords)
        if any(token in lowered for token in ("tempo", "nitroxyl", "fe(no3)", "solvent", "temperature")):
            score += 2
        for token in (
            "optimization",
            "optimal conditions",
            "screened",
            "solvents",
            "room temperature",
            "control experiments",
            "reaction was conducted",
            "reaction conditions",
            "table 1",
            "entry",
            "fe(no3)3",
            "fe(no3)3·9h2o",
            "4-oh-tempo",
        ):
            if token in lowered:
                score += 3
        if "fe(no3)" in lowered and "tempo" in lowered:
            score += 5
        if score > 0 and len(paragraph) > 80:
            scored.append((score, idx, paragraph))
    if not scored:
        return ""
    best = max(scored, key=lambda item: (item[0], -item[1]))[2]
    return re.sub(r"\s+", " ", best).strip()[:900]


def _keywords(question: dict[str, Any]) -> list[str]:
    text = " ".join(
        [
            str(question.get("question") or ""),
            " ".join(str(item) for item in question.get("variable_tags", [])),
            " ".join(str(item) for item in question.get("reaction_types", [])),
        ]
    ).lower()
    preferred = ["tempo", "nitroxyl", "fe(no3)", "nitrate", "solvent", "temperature", "oxidation"]
    tokens = [token for token in preferred if token in text or token in preferred]
    for token in re.findall(r"[a-z][a-z0-9()_-]{3,}", text):
        token = token.replace("_", "-")
        if token not in tokens:
            tokens.append(token)
    return tokens[:32]


def _reference_like(paragraph: str) -> bool:
    lowered = paragraph.strip().lower()
    if lowered.startswith("references") or lowered.startswith("reference "):
        return True
    citation_hits = sum(
        token in lowered
        for token in (
            " chem. rev.",
            " org. lett.",
            " j. am. chem.",
            " angew. chem.",
            " tetrahedron",
            " synthesis,",
            " chem. eur. j.",
        )
    )
    if citation_hits >= 2:
        return True
    return len(re.findall(r"\b\d{4},\s*\d+", paragraph)) >= 4


def _evidence_item(
    *,
    question: dict[str, Any],
    pdf_record: dict[str, Any],
    excerpt: str,
) -> LabEvidenceItem:
    qid = str(question.get("question_id") or "question")
    source_id = str(pdf_record.get("source_id") or pdf_record.get("pdf_id") or "source")
    title = str(pdf_record.get("title") or pdf_record.get("filename") or source_id)
    evidence_id = f"lab-evidence-{qid}-{_short_hash(source_id + excerpt)}"
    metadata = dict(pdf_record.get("metadata") or {})
    return LabEvidenceItem(
        evidence_id=evidence_id,
        question_id=qid,
        source_id=source_id,
        provider="local_pdf_text_heuristic",
        source_type="local_pdf_text",
        title=title,
        url=str(pdf_record.get("url") or ""),
        doi=str(pdf_record.get("doi") or ""),
        claim=_claim_for_question(question, title, metadata),
        supporting_excerpt=excerpt,
        bo_relevance=_bo_relevance(question),
        confidence=0.55,
        needs_review=True,
        metadata={
            **metadata,
            "local_path": pdf_record.get("local_path"),
            "filename": pdf_record.get("filename"),
            "extraction_method": "deterministic_pdf_text_excerpt",
        },
    )


def _claim_for_question(question: dict[str, Any], title: str, metadata: dict[str, Any]) -> str:
    nodes = set(str(item) for item in question.get("target_nodes") or [])
    tags = ", ".join(str(item) for item in metadata.get("filename_tags") or []) or title
    if "design_init_experiments" in nodes:
        return f"The source provides a literature anchor involving {tags}, useful only as scoped prior context for early lab initialization."
    if "stagnation_diagnosis" in nodes:
        return f"The source provides condition-regime context for {tags}, which can support diagnosis without implying hidden outcomes."
    if "verification_pass" in nodes or "semantic_assessment" in nodes:
        return f"The source can support semantic plausibility checks for {tags}, but transfer requires caution because the candidate space may differ."
    return f"The source provides scoped advisory evidence involving {tags} for lab batch composition."


def _bo_relevance(question: dict[str, Any]) -> str:
    nodes = set(str(item) for item in question.get("target_nodes") or [])
    if "design_init_experiments" in nodes:
        return "init"
    if "stagnation_diagnosis" in nodes:
        return "diagnosis"
    if "verification_pass" in nodes or "semantic_assessment" in nodes:
        return "verification"
    return "hypothesis_action"


def _items_for_source(records: list[dict[str, Any]], source_id: str) -> int:
    return sum(1 for item in records if item.get("source_id") == source_id)


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


if __name__ == "__main__":
    sys.exit(main())
