#!/usr/bin/env python
"""Save TRACE Lab recommendations as CSV and print a Markdown table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert TRACE Lab recommendation JSON to CSV and Markdown."
    )
    parser.add_argument(
        "json_path",
        help="Path to an ask/next JSON response, recommendations response, or '-' for stdin.",
    )
    parser.add_argument("--project-id", help="Project id when it is absent from JSON.")
    parser.add_argument("--round-id", help="Round id when it is absent from JSON.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "output"),
        help="Directory for generated CSV. Defaults to trace-collab-skill/output.",
    )
    args = parser.parse_args()

    data = load_json(args.json_path)
    batch = select_batch(data)
    recommendations = list(batch.get("recommendations") or [])
    if not recommendations:
        raise SystemExit("No recommendations found in input JSON.")

    project_id = infer_project_id(data, args.project_id)
    round_id = str(args.round_id or batch.get("round_id") or data.get("round_id") or "round")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{safe_name(project_id)}_{safe_name(round_id)}_recommendations.csv"

    rows = [flatten_recommendation(item) for item in recommendations]
    write_csv(csv_path, rows)

    print(f"Saved CSV: {csv_path}")
    print()
    print(format_markdown_table(rows))
    return 0


def load_json(path: str) -> dict[str, Any]:
    if path == "-":
        text = sys.stdin.read()
    else:
        text = Path(path).read_text(encoding="utf-8-sig")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object.")
    return data


def select_batch(data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(data.get("recommendations"), list):
        return data
    batches = data.get("batches")
    if isinstance(batches, list) and batches:
        latest = batches[-1]
        if isinstance(latest, dict):
            return latest
    raise ValueError("Input JSON must contain recommendations or batches.")


def infer_project_id(data: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    project = data.get("project")
    if isinstance(project, dict) and project.get("project_id"):
        return str(project["project_id"])
    if data.get("project_id"):
        return str(data["project_id"])
    return "project"


def flatten_recommendation(item: dict[str, Any]) -> dict[str, str]:
    candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
    row: dict[str, str] = {
        "rank": as_text(item.get("rank")),
        "recommendation_id": as_text(item.get("recommendation_id")),
        "status": as_text(item.get("status") or "pending"),
        "candidate": "; ".join(f"{key}={value}" for key, value in candidate.items()),
        "batch_role": as_text(item.get("batch_role")),
        "rationale": single_line(item.get("rationale")),
    }
    for key, value in candidate.items():
        row[f"candidate__{key}"] = as_text(value)
    return row


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for preferred in ["rank", "recommendation_id", "status", "candidate", "batch_role", "rationale"]:
        if any(preferred in row for row in rows):
            fieldnames.append(preferred)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_markdown_table(rows: list[dict[str, str]]) -> str:
    candidate_headers = candidate_markdown_headers(rows)
    headers = ["rank", *candidate_headers, "rationale"]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            if header in candidate_headers:
                value = row.get(f"candidate__{header}", "")
            else:
                value = row.get(header, "")
            if header == "rationale" and len(value) > 160:
                value = value[:157].rstrip() + "..."
            values.append(escape_cell(value))
        table.append("| " + " | ".join(values) + " |")
    return "\n".join(table)


def candidate_markdown_headers(rows: list[dict[str, str]]) -> list[str]:
    headers: list[str] = []
    for row in rows:
        for key in row:
            if not key.startswith("candidate__"):
                continue
            header = key.removeprefix("candidate__")
            if header and header not in headers:
                headers.append(header)
    return headers or ["candidate"]


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return cleaned.strip("_") or "trace_lab"


def as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def single_line(value: object) -> str:
    return " ".join(as_text(value).split())


def escape_cell(value: object) -> str:
    return single_line(value).replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
