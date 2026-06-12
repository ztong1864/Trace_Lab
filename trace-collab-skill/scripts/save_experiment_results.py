#!/usr/bin/env python
"""Create a TRACE Lab results CSV from user-provided experiment outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = ["recommendation_id", "status", "yield", "failure_reason", "notes"]
FINAL_STATUSES = {"completed", "failed", "skipped"}
ALLOWED_STATUSES = FINAL_STATUSES | {"pending"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill TRACE Lab result rows from recommendation CSV and outcome JSON."
    )
    parser.add_argument(
        "--recommendations-csv",
        required=True,
        help="CSV produced by save_recommendations.py for the previous round.",
    )
    parser.add_argument(
        "--outcomes-json",
        required=True,
        help="JSON list parsed from the user's reply. Each item needs rank or recommendation_id.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "output"),
        help="Directory for generated result CSV. Defaults to trace-collab-skill/output.",
    )
    parser.add_argument(
        "--output-name",
        help="Optional output file name. Defaults to <recommendations stem>_results.csv.",
    )
    args = parser.parse_args()

    recommendations = read_recommendations(Path(args.recommendations_csv))
    outcomes = read_outcomes(Path(args.outcomes_json))
    rows = build_result_rows(recommendations, outcomes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or default_output_name(Path(args.recommendations_csv))
    output_path = output_dir / output_name
    write_results(output_path, rows)

    print(f"Saved results CSV: {output_path}")
    print()
    print(format_markdown_table(rows))
    return 0


def read_recommendations(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"{path} has no recommendation rows.")
    return rows


def read_outcomes(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Outcome JSON must be a list.")
    return [dict(item) for item in data]


def build_result_rows(
    recommendations: list[dict[str, str]], outcomes: list[dict[str, Any]]
) -> list[dict[str, str]]:
    by_id = {row.get("recommendation_id", ""): row for row in recommendations}
    by_rank = {row.get("rank", ""): row for row in recommendations}
    result_rows = []
    seen = set()

    for outcome in outcomes:
        rec = match_recommendation(outcome, by_id, by_rank)
        recommendation_id = rec.get("recommendation_id", "")
        if recommendation_id in seen:
            raise ValueError(f"Duplicate outcome for {recommendation_id}.")
        seen.add(recommendation_id)

        status = normalize_status(outcome.get("status"))
        value = outcome.get("yield", "")
        failure_reason = as_text(outcome.get("failure_reason", ""))
        notes = as_text(outcome.get("notes", ""))
        if status == "completed" and as_text(value) == "":
            raise ValueError(f"{recommendation_id} is completed but has no yield value.")
        result_rows.append(
            {
                "recommendation_id": recommendation_id,
                "status": status,
                "yield": as_text(value),
                "failure_reason": failure_reason,
                "notes": notes,
            }
        )

    missing = [row.get("recommendation_id", "") for row in recommendations if row.get("recommendation_id", "") not in seen]
    if missing:
        shown = ", ".join(missing[:10])
        raise ValueError(f"Missing outcomes for recommendations: {shown}")
    return result_rows


def match_recommendation(
    outcome: dict[str, Any],
    by_id: dict[str, dict[str, str]],
    by_rank: dict[str, dict[str, str]],
) -> dict[str, str]:
    recommendation_id = as_text(outcome.get("recommendation_id", ""))
    if recommendation_id:
        if recommendation_id not in by_id:
            raise ValueError(f"Unknown recommendation_id: {recommendation_id}")
        return by_id[recommendation_id]
    rank = as_text(outcome.get("rank", ""))
    if rank:
        if rank not in by_rank:
            raise ValueError(f"Unknown rank: {rank}")
        return by_rank[rank]
    raise ValueError("Each outcome must include recommendation_id or rank.")


def normalize_status(value: object) -> str:
    status = as_text(value).lower()
    aliases = {
        "完成": "completed",
        "成功": "completed",
        "已完成": "completed",
        "失败": "failed",
        "跳过": "skipped",
        "未做": "skipped",
        "待做": "pending",
    }
    status = aliases.get(status, status)
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Unsupported status: {value}")
    return status


def write_results(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def default_output_name(recommendations_path: Path) -> str:
    stem = recommendations_path.stem
    if stem.endswith("_recommendations"):
        stem = stem[: -len("_recommendations")]
    return f"{stem}_results.csv"


def format_markdown_table(rows: list[dict[str, str]]) -> str:
    table = [
        "| recommendation_id | status | yield | failure_reason | notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        table.append(
            "| "
            + " | ".join(escape_cell(row.get(field, "")) for field in FIELDNAMES)
            + " |"
        )
    return "\n".join(table)


def as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def escape_cell(value: object) -> str:
    return " ".join(as_text(value).split()).replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
