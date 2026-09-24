#!/usr/bin/env python
"""Small CLI wrapper for TRACE Lab FastAPI endpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8788"


def main() -> int:
    parser = argparse.ArgumentParser(description="Call TRACE Lab API endpoints.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    parser.add_argument("--format", choices=["json", "summary", "both"], default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    subparsers: list[argparse.ArgumentParser] = []

    subparsers.append(sub.add_parser("health"))
    subparsers.append(sub.add_parser("projects"))

    summary = sub.add_parser("summary")
    subparsers.append(summary)
    summary.add_argument("project_id")

    evidence = sub.add_parser("evidence")
    subparsers.append(evidence)
    evidence.add_argument("project_id")

    recommendations = sub.add_parser("recommendations")
    subparsers.append(recommendations)
    recommendations.add_argument("project_id")

    observations = sub.add_parser("observations")
    subparsers.append(observations)
    observations.add_argument("project_id")

    ask = sub.add_parser("ask")
    subparsers.append(ask)
    ask.add_argument("project_id")
    ask.add_argument("--batch-size", type=int)
    ask.add_argument("--planner-name")
    ask.add_argument("--controller-mode", choices=["agentic", "bo_only"], default="agentic")
    ask.add_argument("--agent-config-path")
    ask.add_argument("--planner-use-descriptors", choices=["true", "false"])
    ask.add_argument("--allow-pending", action="store_true")

    tell = sub.add_parser("tell")
    subparsers.append(tell)
    tell.add_argument("project_id")
    tell.add_argument("--results-json", help="JSON file with a list of result rows.")
    tell.add_argument("--results-csv", help="CSV file with result rows.")
    tell.add_argument("--defer-reflection", action="store_true")

    import_obs = sub.add_parser("import-observations")
    subparsers.append(import_obs)
    import_obs.add_argument("project_id")
    import_obs.add_argument("--rows-json")
    import_obs.add_argument("--rows-csv")
    import_obs.add_argument("--source", default="historical")
    import_obs.add_argument("--allow-duplicates", action="store_true")

    trace = sub.add_parser("trace")
    subparsers.append(trace)
    trace.add_argument("project_id")
    trace.add_argument("round_id")

    reset = sub.add_parser("reset")
    subparsers.append(reset)
    reset.add_argument("project_id")
    reset.add_argument("--no-backup", action="store_true")
    reset.add_argument("--keep-historical", action="store_true")

    next_batch = sub.add_parser("next")
    subparsers.append(next_batch)
    next_batch.add_argument("project_id")
    next_batch.add_argument("--batch-size", type=int)
    next_batch.add_argument("--planner-name")
    next_batch.add_argument("--controller-mode", choices=["agentic", "bo_only"], default="agentic")
    next_batch.add_argument("--planner-use-descriptors", choices=["true", "false"])

    create = sub.add_parser("create")
    subparsers.append(create)
    create.add_argument("project_id")
    create.add_argument("--design-csv", required=True)
    create.add_argument("--objective-name", default="yield")
    create.add_argument("--planner-name", default="atlas")
    create.add_argument("--controller-mode", choices=["agentic", "bo_only"], default="agentic")
    create.add_argument("--batch-size", type=int, default=6)
    create.add_argument("--overwrite", action="store_true")

    for subparser in subparsers:
        subparser.add_argument("--pretty", action="store_true", help=argparse.SUPPRESS)
        subparser.add_argument(
            "--format",
            choices=["json", "summary", "both"],
            default=None,
            help=argparse.SUPPRESS,
        )

    args = parser.parse_args()
    if args.format is None:
        args.format = "both" if args.command in {"ask", "next"} else "json"
    base_url = args.base_url.rstrip("/")

    try:
        result = dispatch(args, base_url)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print_output(
        result,
        pretty=args.pretty,
        output_format=args.format,
        command=args.command,
    )
    return 0


def dispatch(args: argparse.Namespace, base_url: str) -> Any:
    command = args.command
    if command == "health":
        return request_json(base_url, "GET", "/api/health")
    if command == "projects":
        return request_json(base_url, "GET", "/api/projects")
    if command == "summary":
        return request_json(base_url, "GET", f"/api/projects/{q(args.project_id)}")
    if command == "evidence":
        return request_json(base_url, "GET", f"/api/projects/{q(args.project_id)}/evidence")
    if command == "recommendations":
        return request_json(base_url, "GET", f"/api/projects/{q(args.project_id)}/recommendations")
    if command == "observations":
        return request_json(base_url, "GET", f"/api/projects/{q(args.project_id)}/observations")
    if command == "trace":
        return request_json(base_url, "GET", f"/api/projects/{q(args.project_id)}/trace/{q(args.round_id)}")
    if command == "ask":
        if not args.allow_pending:
            assert_no_pending(base_url, args.project_id)
        return ask(base_url, args)
    if command == "next":
        assert_no_pending(base_url, args.project_id)
        return ask(base_url, args)
    if command == "tell":
        rows = load_rows(args.results_json, args.results_csv, "results")
        return request_json(
            base_url,
            "POST",
            f"/api/projects/{q(args.project_id)}/tell",
            {"results": rows, "defer_reflection": bool(args.defer_reflection)},
        )
    if command == "import-observations":
        rows = load_rows(args.rows_json, args.rows_csv, "rows")
        return request_json(
            base_url,
            "POST",
            f"/api/projects/{q(args.project_id)}/observations/import",
            {
                "rows": rows,
                "source": args.source,
                "allow_duplicates": bool(args.allow_duplicates),
            },
        )
    if command == "reset":
        return request_json(
            base_url,
            "POST",
            f"/api/projects/{q(args.project_id)}/reset",
            {"backup": not args.no_backup, "keep_historical": bool(args.keep_historical)},
        )
    if command == "create":
        design_records = read_csv_rows(args.design_csv)
        return request_json(
            base_url,
            "POST",
            "/api/projects",
            {
                "project_id": args.project_id,
                "config": {
                    "project_id": args.project_id,
                    "objective_name": args.objective_name,
                    "planner_name": args.planner_name,
                    "controller_mode": args.controller_mode,
                    "batch_size": args.batch_size,
                },
                "design_records": design_records,
                "overwrite": bool(args.overwrite),
            },
        )
    raise ValueError(f"Unsupported command: {command}")


def ask(base_url: str, args: argparse.Namespace) -> Any:
    payload: dict[str, Any] = {}
    if getattr(args, "batch_size", None) is not None:
        payload["batch_size"] = args.batch_size
    if getattr(args, "planner_name", None):
        payload["planner_name"] = args.planner_name
    payload["controller_mode"] = getattr(args, "controller_mode", None) or "agentic"
    if getattr(args, "agent_config_path", None):
        payload["agent_config_path"] = args.agent_config_path
    if getattr(args, "planner_use_descriptors", None) is not None:
        payload["planner_use_descriptors"] = args.planner_use_descriptors == "true"
    return request_json(base_url, "POST", f"/api/projects/{q(args.project_id)}/ask", payload)


def assert_no_pending(base_url: str, project_id: str) -> None:
    data = request_json(base_url, "GET", f"/api/projects/{q(project_id)}/recommendations")
    pending = []
    for batch in data.get("batches", []):
        for rec in batch.get("recommendations", []):
            if str(rec.get("status") or "pending").lower() == "pending":
                pending.append(rec.get("recommendation_id"))
    if pending:
        shown = ", ".join(str(item) for item in pending[:10])
        more = "" if len(pending) <= 10 else f", ... ({len(pending)} total)"
        raise RuntimeError(
            "Pending recommendations exist. Submit/fail/skip them before asking again, "
            f"or pass --allow-pending. Pending: {shown}{more}"
        )


def load_rows(json_path: str | None, csv_path: str | None, label: str) -> list[dict[str, Any]]:
    if bool(json_path) == bool(csv_path):
        raise ValueError(f"Provide exactly one of --{label}-json or --{label}-csv.")
    if json_path:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{json_path} must contain a JSON list.")
        return [dict(row) for row in data]
    assert csv_path is not None
    return read_csv_rows(csv_path)


def read_csv_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def request_json(base_url: str, method: str, path: str, payload: Any | None = None) -> Any:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(base_url + path, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=300) as response:
            text = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {method} {path}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach TRACE Lab API at {base_url}: {exc.reason}") from exc
    return json.loads(text) if text.strip() else {}


def q(value: str) -> str:
    return quote(str(value), safe="")


def print_output(
    data: Any,
    *,
    pretty: bool,
    output_format: str,
    command: str,
) -> None:
    if output_format in {"summary", "both"}:
        summary = format_summary(data, command=command)
        if summary:
            print(summary)
    if output_format in {"json", "both"}:
        if output_format == "both":
            print("\nJSON:")
        print_json(data, pretty=pretty)


def print_json(data: Any, *, pretty: bool) -> None:
    if pretty:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(data, ensure_ascii=False, separators=(",", ":")))


def format_summary(data: Any, *, command: str) -> str:
    if command in {"ask", "next"}:
        return format_recommendations(data)
    if command == "tell":
        return format_tell_result(data)
    if command == "recommendations":
        batches = data.get("batches", []) if isinstance(data, dict) else []
        if not batches:
            return "No recommendation batches."
        return format_recommendations(batches[-1])
    return ""


def format_recommendations(data: dict[str, Any]) -> str:
    recommendations = data.get("recommendations") or []
    if not recommendations:
        return "No recommendations."
    project = data.get("project") or {}
    lines = [
        f"Round: {data.get('round_id', '')}",
        f"Controller: {data.get('controller_mode', project.get('controller_mode', ''))}",
        "",
    ]
    planner_error = next(
        (item.get("planner_error") for item in recommendations if item.get("planner_error")),
        "",
    )
    if planner_error:
        lines[-1:-1] = [
            f"WARNING: planner fallback -- the optimizer failed ({single_line(planner_error)}), "
            "so these are random candidates, not Bayesian-optimization recommendations.",
        ]
    planner_notes = recommendations[0].get("planner_warnings") or []
    lines[-1:-1] = [f"NOTE: {single_line(note)}" for note in planner_notes]
    rows = []
    for item in recommendations:
        candidate = item.get("candidate") or {}
        candidate_text = "; ".join(f"{key}={value}" for key, value in candidate.items())
        rationale = single_line(item.get("rationale", ""))
        if len(rationale) > 160:
            rationale = rationale[:157].rstrip() + "..."
        rows.append(
            [
                str(item.get("rank", "")),
                str(item.get("recommendation_id", "")),
                candidate_text,
                str(item.get("batch_role", "")),
                str(item.get("proposed_by", "")),
                rationale,
            ]
        )
    lines.extend(
        markdown_table(["rank", "recommendation_id", "candidate", "role", "proposed_by", "rationale"], rows)
    )
    return "\n".join(lines)


def format_tell_result(data: dict[str, Any]) -> str:
    appended = data.get("appended") or []
    counts: dict[str, int] = {}
    for row in appended:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    status_text = ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"
    return "\n".join(
        [
            f"Project: {data.get('project_id', '')}",
            f"Appended: {len(appended)} ({status_text})",
            f"Observation count: {data.get('observation_count', '')}",
            f"Completed count: {data.get('completed_observation_count', '')}",
            f"Best so far: {data.get('best_so_far', '')}",
            f"Reflection: {data.get('reflection_status', '')}",
        ]
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(escape_cell(value) for value in row) + " |")
    return table


def escape_cell(value: object) -> str:
    return single_line(value).replace("|", "\\|")


def single_line(value: object) -> str:
    return " ".join(str(value or "").split())


if __name__ == "__main__":
    raise SystemExit(main())
