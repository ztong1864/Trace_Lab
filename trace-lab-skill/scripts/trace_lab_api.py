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
    ask.add_argument("--controller-mode", choices=["agentic", "bo_only"], help="Default: the project's setting.")
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
    next_batch.add_argument("--controller-mode", choices=["agentic", "bo_only"], help="Default: the project's setting.")
    next_batch.add_argument("--planner-use-descriptors", choices=["true", "false"])

    create = sub.add_parser("create")
    subparsers.append(create)
    create.add_argument("project_id")
    create.add_argument("--design-csv", required=True)
    create.add_argument("--objective-name", default="yield")
    create.add_argument("--planner-name", default="chunked_gp")
    create.add_argument("--acquisition-function", choices=["ei", "ucb", "ei_ucb"])
    create.add_argument("--controller-mode", choices=["agentic", "bo_only"], default="agentic")
    create.add_argument("--batch-size", type=int, default=6)
    create.add_argument("--planner-use-descriptors", action="store_true")
    create.add_argument(
        "--descriptor",
        nargs=3,
        action="append",
        default=[],
        metavar=("VARIABLE", "VALUE_COLUMN", "CSV"),
        help="Descriptor table for one variable (one row per option); repeatable.",
    )
    create.add_argument(
        "--alias",
        nargs=3,
        action="append",
        default=[],
        metavar=("VARIABLE", "OPTION", "TABLE_VALUE"),
        help="Match a design option to a differently spelled descriptor-table row; repeatable.",
    )
    create.add_argument("--observations-csv", help="Historical results to import with the project.")
    create.add_argument("--evidence", help="Evidence cards (.jsonl or .csv) to import with the project.")
    create.add_argument("--overwrite", action="store_true")

    register = sub.add_parser("register", help="Use an existing project folder on the server's disk, in place.")
    subparsers.append(register)
    register.add_argument("path", help="Absolute folder path on the server.")
    register.add_argument("--handle", help="Handle to use (default: the folder name).")
    register.add_argument(
        "--drop-config-key",
        action="append",
        default=[],
        help="Unknown project.yaml setting to remove (backed up first); repeatable.",
    )

    unregister = sub.add_parser("unregister", help="Forget a registered project (its files are untouched).")
    subparsers.append(unregister)
    unregister.add_argument("project_id")

    check = sub.add_parser("check", help="Check a project, or a folder on the server with --path.")
    subparsers.append(check)
    check.add_argument("project_id", nargs="?")
    check.add_argument("--path", help="Absolute folder path on the server (instead of a handle).")

    descriptors = sub.add_parser("descriptors", help="Attach a descriptor table to one design variable.")
    subparsers.append(descriptors)
    descriptors.add_argument("project_id")
    descriptors.add_argument("--variable", required=True)
    descriptors.add_argument("--value-column", required=True)
    descriptors.add_argument("--csv", required=True)
    descriptors.add_argument("--alias", action="append", default=[], metavar="OPTION=TABLE_VALUE")
    descriptors.add_argument("--replace", action="store_true")

    import_evidence = sub.add_parser("import-evidence", help="Add or replace evidence cards (.jsonl or .csv).")
    subparsers.append(import_evidence)
    import_evidence.add_argument("project_id")
    import_evidence.add_argument("--file", required=True)
    import_evidence.add_argument("--mode", choices=["append", "replace"], default="append")

    config = sub.add_parser("config", help="Show or change a project's settings (project.yaml).")
    subparsers.append(config)
    config.add_argument("project_id")
    config.add_argument(
        "--set",
        dest="settings",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Setting to change; repeatable. Dotted keys set planner options, e.g. "
            "planner_options.chunked_gp.min_changed_variables=3. Values are read as JSON "
            "when possible (3, true, null removes an option), otherwise as text."
        ),
    )
    config.add_argument("--json", dest="settings_json", help="JSON object of settings to change.")

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
        return request_json(base_url, "POST", "/api/projects", create_payload(args))
    if command == "register":
        return request_json(
            base_url,
            "POST",
            "/api/projects/register",
            {"path": args.path, "handle": args.handle, "drop_config_keys": args.drop_config_key},
        )
    if command == "unregister":
        return request_json(base_url, "DELETE", f"/api/projects/{q(args.project_id)}/registration")
    if command == "check":
        if bool(args.project_id) == bool(args.path):
            raise ValueError("Give a project handle or --path, not both.")
        if args.path:
            return request_json(base_url, "POST", "/api/projects/check", {"path": args.path})
        return request_json(base_url, "GET", f"/api/projects/{q(args.project_id)}/check")
    if command == "descriptors":
        return request_json(
            base_url,
            "POST",
            f"/api/projects/{q(args.project_id)}/descriptors",
            {
                "variable": args.variable,
                "value_column": args.value_column,
                "csv": read_text(args.csv),
                "aliases": parse_aliases(args.alias),
                "replace": bool(args.replace),
            },
        )
    if command == "import-evidence":
        key = "csv" if Path(args.file).suffix.lower() == ".csv" else "jsonl"
        return request_json(
            base_url,
            "POST",
            f"/api/projects/{q(args.project_id)}/evidence/import",
            {key: read_text(args.file), "mode": args.mode},
        )
    if command == "config":
        updates = parse_config_settings(args.settings, args.settings_json)
        if not updates:
            return request_json(base_url, "GET", f"/api/projects/{q(args.project_id)}").get("project", {})
        return request_json(base_url, "PATCH", f"/api/projects/{q(args.project_id)}/config", updates)
    raise ValueError(f"Unsupported command: {command}")


def create_payload(args: argparse.Namespace) -> dict[str, Any]:
    """POST /api/projects body: design CSV plus optional descriptor tables, results and evidence."""
    config: dict[str, Any] = {
        "project_id": args.project_id,
        "objective_name": args.objective_name,
        "planner_name": args.planner_name,
        "controller_mode": args.controller_mode,
        "batch_size": args.batch_size,
        "planner_use_descriptors": bool(args.planner_use_descriptors),
    }
    if args.acquisition_function:
        config["acquisition_function"] = args.acquisition_function
    aliases: dict[str, dict[str, str]] = {}
    for variable, option, table_value in args.alias:
        aliases.setdefault(variable, {})[option] = table_value
    unknown = sorted(set(aliases) - {variable for variable, _, _ in args.descriptor})
    if unknown:
        raise ValueError(f"--alias given for variables without a --descriptor table: {unknown}.")
    payload: dict[str, Any] = {
        "project_id": args.project_id,
        "config": config,
        "design_csv": read_text(args.design_csv),
        "descriptor_tables": [
            {
                "variable": variable,
                "value_column": value_column,
                "csv": read_text(path),
                "aliases": aliases.get(variable, {}),
            }
            for variable, value_column, path in args.descriptor
        ],
        "overwrite": bool(args.overwrite),
    }
    if args.observations_csv:
        payload["observations_csv"] = read_text(args.observations_csv)
    if args.evidence:
        key = "evidence_csv" if Path(args.evidence).suffix.lower() == ".csv" else "evidence_jsonl"
        payload[key] = read_text(args.evidence)
    return payload


def parse_aliases(items: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for item in items:
        option, sep, table_value = item.partition("=")
        if not sep or not option.strip():
            raise ValueError(f"--alias expects OPTION=TABLE_VALUE, got `{item}`.")
        aliases[option.strip()] = table_value.strip()
    return aliases


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8-sig")


def parse_config_settings(settings: list[str], settings_json: str | None) -> dict[str, Any]:
    """Build a settings change from --json and repeatable --set KEY=VALUE (dotted keys nest)."""
    updates: dict[str, Any] = json.loads(settings_json) if settings_json else {}
    if not isinstance(updates, dict):
        raise ValueError("--json must be a JSON object.")
    for item in settings:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects KEY=VALUE, got `{item}`.")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        parts = [part.strip() for part in key.split(".")]
        target = updates
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"--set `{key}` conflicts with another setting.")
        target[parts[-1]] = value
    return updates


def ask(base_url: str, args: argparse.Namespace) -> Any:
    payload: dict[str, Any] = {}
    if getattr(args, "batch_size", None) is not None:
        payload["batch_size"] = args.batch_size
    if getattr(args, "planner_name", None):
        payload["planner_name"] = args.planner_name
    if getattr(args, "controller_mode", None):
        payload["controller_mode"] = args.controller_mode
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
    if command == "check":
        return format_check(data)
    if command == "register":
        header = f"Registered `{data.get('handle', '')}` -> {data.get('project_dir', '')}"
        return header + "\n" + format_check(data.get("check") or {})
    if command == "projects":
        return format_projects(data)
    return ""


def format_check(report: dict[str, Any]) -> str:
    errors = report.get("errors") or []
    warnings = report.get("warnings") or []
    status = "OK" if report.get("ok") else "FAILED"
    lines = [f"Check: {status} ({len(errors)} error(s), {len(warnings)} warning(s))"]
    for kind, items in (("ERROR", errors), ("WARNING", warnings)):
        for item in items:
            where = item.get("file", "") + (f" line {item['line']}" if item.get("line") else "")
            lines.append(f"{kind}: {where}: {single_line(item.get('message', ''))}")
    facts = report.get("facts") or {}
    shown = (
        "combination_count",
        "observation_count",
        "completed_observation_count",
        "recommendation_rounds",
        "pending_recommendation_count",
        "evidence_card_count",
    )
    if facts:
        lines.append("Facts: " + ", ".join(f"{key}={facts[key]}" for key in shown if key in facts))
    return "\n".join(lines)


def format_projects(data: dict[str, Any]) -> str:
    rows = []
    for item in data.get("projects") or []:
        project = item.get("project") or {}
        rows.append(
            [
                str(item.get("handle", "")),
                "yes" if item.get("registered") else "no",
                str(project.get("project_id", "")),
                str(item.get("observation_count", "")),
                str(item.get("pending_recommendation_count", "")),
                single_line(item.get("error", "")),
            ]
        )
    if not rows:
        return "No projects."
    headers = ["handle", "registered", "project_id", "observations", "pending", "problem"]
    return "\n".join(markdown_table(headers, rows))


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
    lines = [
        f"Project: {data.get('project_id', '')}",
        f"Appended: {len(appended)} ({status_text})",
        f"Observation count: {data.get('observation_count', '')}",
        f"Completed count: {data.get('completed_observation_count', '')}",
        f"Best so far: {data.get('best_so_far', '')}",
        f"Reflection: {data.get('reflection_status', '')}",
    ]
    for item in data.get("reflection_errors") or []:
        lines.append(
            f"WARNING: reflection failed for {item.get('recommendation_id', '')} "
            f"({single_line(item.get('error', ''))}); the result itself was saved."
        )
    return "\n".join(lines)


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
