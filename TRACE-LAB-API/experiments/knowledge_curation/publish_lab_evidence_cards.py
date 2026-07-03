#!/usr/bin/env python
"""Publish reviewed or pilot evidence items as project-local lab evidence cards."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chem_agent_bo.lab.evidence import EvidenceCard, EvidenceStore
from experiments.knowledge_curation.source_acquisition.schema import read_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reaction-scope", default="oxidative esterification")
    parser.add_argument("--default-mapping-status", default="same_reaction_family")
    parser.add_argument("--allowed-use", default="advisory")
    parser.add_argument("--leakage-risk", default="clean_literature_prior")
    parser.add_argument("--max-cards", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = {
        str(item.get("question_id") or ""): item
        for item in read_questions(_resolve(args.questions))
    }
    evidence_items = _load_jsonl(_resolve(args.evidence))
    cards: list[EvidenceCard] = []
    for item in evidence_items:
        question = questions.get(str(item.get("question_id") or ""), {})
        cards.append(
            _card_from_evidence(
                item,
                question,
                reaction_scope=str(args.reaction_scope or ""),
                mapping_status=str(args.default_mapping_status or "same_reaction_family"),
                allowed_use=str(args.allowed_use or "advisory"),
                leakage_risk=str(args.leakage_risk or "clean_literature_prior"),
            )
        )
        if args.max_cards and len(cards) >= int(args.max_cards):
            break
    output = _resolve(args.output)
    EvidenceStore(cards).write_jsonl(output)
    print(f"wrote {len(cards)} lab evidence cards to {output}")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _card_from_evidence(
    item: dict[str, Any],
    question: dict[str, Any],
    *,
    reaction_scope: str,
    mapping_status: str,
    allowed_use: str,
    leakage_risk: str,
) -> EvidenceCard:
    metadata = dict(item.get("metadata") or {})
    evidence_id = str(item.get("evidence_id") or _short_hash(json.dumps(item, sort_keys=True)))
    title = str(item.get("title") or item.get("source_id") or "literature source")
    source_path = str(metadata.get("local_path") or item.get("source_path") or "")
    target_nodes = [str(value) for value in question.get("target_nodes") or []]
    variable_scope = _variable_scope(question, item)
    return EvidenceCard(
        card_id=f"lit_{_slug(evidence_id)}",
        source=title,
        summary=_clean(item.get("claim")),
        reaction_scope=reaction_scope,
        variable_scope=variable_scope,
        target_nodes=target_nodes,
        mapping_status=_mapping_status_for_item(item, question, fallback=mapping_status),
        confidence=str(item.get("confidence") or "medium"),
        allowed_use=allowed_use,
        source_type=str(item.get("source_type") or "literature"),
        source_path=source_path,
        doi=str(item.get("doi") or ""),
        supporting_excerpt=_clean(item.get("supporting_excerpt"), max_chars=700),
        transferability_note=_transferability_note(item, question),
        leakage_risk=leakage_risk,
        notes="Pilot lab evidence card generated from local literature; review before broad reuse.",
    )


def _variable_scope(question: dict[str, Any], item: dict[str, Any]) -> list[str]:
    variables = [str(value) for value in question.get("variable_tags") or []]
    text = " ".join(
        [
            str(item.get("claim") or ""),
            str(item.get("supporting_excerpt") or ""),
            " ".join(str(value) for value in (item.get("metadata") or {}).get("filename_tags", [])),
        ]
    ).lower()
    selected = [
        variable
        for variable in variables
        if _variable_token(variable) in text or variable.lower() in text
    ]
    return selected or variables[:4]


def _mapping_status_for_item(
    item: dict[str, Any],
    question: dict[str, Any],
    *,
    fallback: str,
) -> str:
    nodes = set(str(value) for value in question.get("target_nodes") or [])
    if "verification_pass" in nodes or "semantic_assessment" in nodes:
        return "background"
    if item.get("needs_review"):
        return fallback
    return fallback


def _transferability_note(item: dict[str, Any], question: dict[str, Any]) -> str:
    nodes = ", ".join(str(value) for value in question.get("target_nodes") or []) or "lab controller"
    return (
        f"Use as scoped advisory evidence for {nodes}; the literature source may "
        "not share the exact substrate, candidate pool, or objective with the current lab project."
    )


def _variable_token(variable: str) -> str:
    lowered = variable.lower()
    if "tempo" in lowered:
        return "tempo"
    if "solvent" in lowered:
        return "solvent"
    if "temperature" in lowered:
        return "temperature"
    if "additive" in lowered:
        return "additive"
    if "no3" in lowered or "metal" in lowered:
        return "fe(no3"
    return lowered


def _clean(value: object, *, max_chars: int = 420) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:max_chars].rstrip()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "evidence"


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


if __name__ == "__main__":
    sys.exit(main())
