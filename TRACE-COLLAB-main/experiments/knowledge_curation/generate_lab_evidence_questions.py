#!/usr/bin/env python
"""Generate lab-specific evidence extraction questions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chem_agent_bo.lab.project import LabProject


QUESTION_TEMPLATES = [
    {
        "suffix": "init-literature-anchors",
        "target_nodes": ["design_init_experiments"],
        "trigger_tags": ["early_init_design", "literature_anchor"],
        "question": (
            "Which source-grounded metal, nitroxyl, solvent, or temperature patterns "
            "can serve as conservative literature anchors for {reaction_family}?"
        ),
        "why": "Early lab batches need plausible anchors without turning literature conditions into oracle answers.",
    },
    {
        "suffix": "action-local-refinement",
        "target_nodes": ["hypothesis_action", "lab_batch_composition"],
        "trigger_tags": ["local_refinement", "batch_portfolio"],
        "question": (
            "After a promising {reaction_family} condition appears, which variables are "
            "reasonable to locally refine while keeping transfer from the literature interpretable?"
        ),
        "why": "This helps TRACE compose a real-lab batch as a portfolio instead of ranked top-k output.",
    },
    {
        "suffix": "diagnosis-regime-shift",
        "target_nodes": ["stagnation_diagnosis"],
        "trigger_tags": ["stagnation", "regime_shift"],
        "question": (
            "Which literature signals suggest that poor progress in {reaction_family} is "
            "more likely due to a regime mismatch than insufficient local calibration?"
        ),
        "why": "The diagnosis node needs scoped evidence to choose local refinement or broader exploration.",
    },
    {
        "suffix": "semantic-candidate-caution",
        "target_nodes": ["semantic_assessment", "verification_pass"],
        "trigger_tags": ["transferability", "evidence_sufficiency"],
        "question": (
            "Which claims about {reaction_family} candidate plausibility require caution "
            "because the literature source differs from the current candidate space?"
        ),
        "why": "Verification should downgrade confidence when a source is only variable-level or background evidence.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-dir", default="")
    parser.add_argument("--dataset", default="oxidative_esterification_demo")
    parser.add_argument("--reaction-family", default="oxidative esterification")
    parser.add_argument("--reaction-type", action="append", dest="reaction_types")
    parser.add_argument("--variable", action="append", dest="variables")
    parser.add_argument("--optimization-goal", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_variables: list[str] = []
    reaction_family = str(args.reaction_family or "oxidative esterification")
    dataset = str(args.dataset or "oxidative_esterification_demo")
    if args.project_dir:
        project = LabProject(_resolve(args.project_dir))
        config = project.load_config()
        design_space = project.load_design_space()
        project_variables = design_space.condition_names
        reaction_family = config.reaction_name or reaction_family
        dataset = config.project_id or dataset
    variables = _list_from_args(args.variables) or project_variables or [
        "Additive",
        "Solvent",
        "TEMPO derivative",
        "M(NO3)3",
        "Temperature",
    ]
    reaction_types = _list_from_args(args.reaction_types) or [reaction_family]
    goals = _list_from_args(args.optimization_goal) or ["real_lab_batch_ask_tell"]

    records = []
    for template in QUESTION_TEMPLATES:
        question_id = f"{_slug(dataset)}-{template['suffix']}"
        records.append(
            {
                "question_id": question_id,
                "dataset_scope": [dataset],
                "reaction_types": reaction_types,
                "reaction_family": reaction_family,
                "target_nodes": list(template["target_nodes"]),
                "variable_tags": variables,
                "trigger_tags": list(template["trigger_tags"]),
                "question": template["question"].format(reaction_family=reaction_family),
                "why_this_matters_for_bo": template["why"],
                "expected_card_type": "lab_evidence_card",
                "optimization_goals": goals,
            }
        )
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(records)} lab evidence questions to {output}")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _list_from_args(values: list[str] | None) -> list[str]:
    items: list[str] = []
    for value in values or []:
        items.extend(part.strip() for part in str(value).split(",") if part.strip())
    return list(dict.fromkeys(items))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "lab"


if __name__ == "__main__":
    sys.exit(main())
