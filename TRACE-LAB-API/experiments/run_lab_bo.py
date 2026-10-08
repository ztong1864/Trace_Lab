"""CLI for TRACE real-lab ask/tell projects."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VENDOR_DIR = PROJECT_ROOT / ".vendor_py"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    # Keep environment packages first to avoid version shadowing.
    sys.path.append(str(VENDOR_DIR))

from chem_agent_bo.lab.design_space import DescriptorTableError, DesignSpace
from chem_agent_bo.lab.evidence import EvidenceImportError
from chem_agent_bo.lab.project import LabProject, ProjectConfig
from chem_agent_bo.lab.registry import default_handle
from chem_agent_bo.lab.service import LabBOService, ProjectCheckError, read_results_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage TRACE real-lab ask/tell projects.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a lab project from normalized design_space.csv.")
    _add_project_args(init_parser)
    init_parser.add_argument("--design-space-csv", required=True)
    init_parser.add_argument("--overwrite", action="store_true")

    collab_parser = subparsers.add_parser(
        "init-collaborator",
        help="Create a lab project from collaborator additive/solvent/TEMPO descriptor CSVs.",
    )
    _add_project_args(collab_parser)
    collab_parser.add_argument("--additive-csv", required=True)
    collab_parser.add_argument("--solvent-csv", required=True)
    collab_parser.add_argument("--tempo-csv", required=True)
    collab_parser.add_argument("--overwrite", action="store_true")

    ask_parser = subparsers.add_parser("ask", help="Generate the next lab recommendation batch.")
    ask_parser.add_argument("--project-dir", required=True)
    ask_parser.add_argument("--batch-size", type=int, default=None)
    ask_parser.add_argument("--planner-name", choices=("atlas", "chunked_gp", "random"), default=None)
    ask_parser.add_argument("--controller-mode", choices=("agentic", "bo_only"), default=None)
    ask_parser.add_argument("--agent-config", default=None)
    ask_parser.add_argument(
        "--planner-use-descriptors",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the project descriptor-planner setting for this ask call.",
    )

    tell_parser = subparsers.add_parser("tell", help="Import measured results for pending recommendations.")
    tell_parser.add_argument("--project-dir", required=True)
    tell_parser.add_argument("--results-csv", required=True)

    import_parser = subparsers.add_parser(
        "import-observations",
        help="Import historical or manual observations into a lab project.",
    )
    import_parser.add_argument("--project-dir", required=True)
    import_parser.add_argument("--observations-csv", required=True)
    import_parser.add_argument("--source", default="historical")
    import_parser.add_argument("--allow-duplicates", action="store_true")

    summary_parser = subparsers.add_parser("summary", help="Print lab project summary.")
    summary_parser.add_argument("--project-dir", required=True)

    register_parser = subparsers.add_parser(
        "register",
        help="Make existing project folders available to the web API by handle, used in place.",
    )
    _add_projects_root_arg(register_parser)
    target = register_parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--path", help="One project folder (containing project.yaml).")
    target.add_argument("--scan", help="Register every project folder found under this folder.")
    register_parser.add_argument("--handle", help="Handle for --path (default: the folder name).")
    register_parser.add_argument(
        "--drop-config-key",
        action="append",
        default=[],
        help="Unknown project.yaml setting to remove (backed up first); repeatable. --path only.",
    )
    register_parser.add_argument(
        "--set-config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Editable setting to change before the check, e.g. agent_config_path=configs/agent_bo.yaml "
        "(VALUE is read as JSON when it parses, so batch_size=4 is a number); repeatable. --path only.",
    )
    register_parser.add_argument("--dry-run", action="store_true", help="With --scan: check only, register nothing.")

    unregister_parser = subparsers.add_parser("unregister", help="Forget a registered project (files untouched).")
    _add_projects_root_arg(unregister_parser)
    unregister_parser.add_argument("--handle", required=True)

    check_parser = subparsers.add_parser("check", help="Check a project folder for problems.")
    check_parser.add_argument("--project-dir", required=True)

    descriptor_parser = subparsers.add_parser(
        "descriptors",
        help="Attach a per-option descriptor CSV to one design variable (design_space.csv is backed up).",
    )
    descriptor_parser.add_argument("--project-dir", required=True)
    descriptor_parser.add_argument("--variable", required=True)
    descriptor_parser.add_argument("--csv", required=True, help="Descriptor table, one row per option.")
    descriptor_parser.add_argument("--value-column", required=True, help="Column holding the option values.")
    descriptor_parser.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="OPTION=TABLE_VALUE",
        help="Match a design option to a differently spelled table row; repeatable.",
    )
    descriptor_parser.add_argument("--replace", action="store_true", help="Drop the variable's old descriptors first.")

    evidence_parser = subparsers.add_parser(
        "import-evidence",
        help="Add or replace evidence cards from a .jsonl or .csv file (validated strictly, old file backed up).",
    )
    evidence_parser.add_argument("--project-dir", required=True)
    evidence_parser.add_argument("--file", required=True, help="Evidence cards as .jsonl or .csv.")
    evidence_parser.add_argument("--mode", choices=("append", "replace"), default="append")

    preview_parser = subparsers.add_parser(
        "evidence-preview",
        help="Show which evidence cards the controller would be shown, and why the others never are.",
    )
    preview_parser.add_argument("--project-dir", required=True)
    preview_parser.add_argument("--json", action="store_true", help="Print the full result as JSON.")

    prepare_parser = subparsers.add_parser(
        "evidence-prepare",
        help="Extract page text from papers and write the drafting packets and project context.",
    )
    prepare_parser.add_argument("--project-dir", required=True)
    prepare_parser.add_argument("--pdf-dir", action="append", default=[], help="Folder with PDFs (repeatable).")
    prepare_parser.add_argument("--pdf", action="append", default=[], help="One PDF file (repeatable).")

    verify_parser = subparsers.add_parser(
        "evidence-verify",
        help="Check drafted findings in evidence_work/drafts/ against the paper text.",
    )
    verify_parser.add_argument("--project-dir", required=True)
    verify_parser.add_argument("--json", action="store_true", help="Print every checked record as JSON.")

    sheet_parser = subparsers.add_parser(
        "evidence-sheet",
        help="Write review_sheet.csv (one row per verified draft) for the chemist to accept or reject.",
    )
    sheet_parser.add_argument("--project-dir", required=True)
    sheet_parser.add_argument("--force", action="store_true", help="Replace a sheet that already has decisions.")

    accept_parser = subparsers.add_parser(
        "evidence-accept",
        help="Import the rows marked `accept` in the review sheet (re-checked; old evidence file backed up).",
    )
    accept_parser.add_argument("--project-dir", required=True)
    accept_parser.add_argument("--sheet", default="", help="Sheet to read (default: evidence_work/review_sheet.csv).")
    return parser.parse_args()


def _add_projects_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--projects-root",
        default=os.environ.get("TRACE_LAB_PROJECTS_ROOT", "runs/lab_projects"),
        help="The web API's projects root, where registry.yaml lives.",
    )


def _add_project_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--reaction-name", default="oxidative_esterification")
    parser.add_argument("--objective-name", default="yield")
    parser.add_argument("--goal", choices=("maximize", "minimize"), default="maximize")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--planner-name", choices=("atlas", "chunked_gp", "random"), default="atlas")
    parser.add_argument("--controller-mode", choices=("agentic", "bo_only"), default="agentic")
    parser.add_argument("--agent-config", default="configs/agent_bo.yaml")
    parser.add_argument("--planner-use-descriptors", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reaction-scope", default="")


def main() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        # Piped I/O on a Windows code page (gbk, cp1252) cannot carry every character the API returns.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    service = LabBOService()
    if args.command == "init":
        design = DesignSpace.read_csv(args.design_space_csv)
        project = service.create_project(
            args.project_dir,
            config=_config_from_args(args),
            design_space=design,
            overwrite=bool(args.overwrite),
        )
        _print_json(project.summary())
        return
    if args.command == "init-collaborator":
        design = DesignSpace.from_collaborator_csvs(
            additive_csv=args.additive_csv,
            solvent_csv=args.solvent_csv,
            tempo_csv=args.tempo_csv,
        )
        project = service.create_project(
            args.project_dir,
            config=_config_from_args(args),
            design_space=design,
            overwrite=bool(args.overwrite),
        )
        _print_json(project.summary())
        return
    if args.command == "ask":
        _print_json(
            service.ask(
                args.project_dir,
                batch_size=args.batch_size,
                planner_name=args.planner_name,
                controller_mode=args.controller_mode,
                agent_config_path=args.agent_config,
                planner_use_descriptors=args.planner_use_descriptors,
            )
        )
        return
    if args.command == "tell":
        _print_json(
            service.tell(
                args.project_dir,
                results=read_results_csv(args.results_csv),
            )
        )
        return
    if args.command == "import-observations":
        _print_json(
            service.import_observations(
                args.project_dir,
                rows=read_results_csv(args.observations_csv),
                source=args.source,
                allow_duplicates=bool(args.allow_duplicates),
            )
        )
        return
    if args.command == "summary":
        _print_json(LabProject(Path(args.project_dir)).summary())
        return
    if args.command == "check":
        from chem_agent_bo.lab.project_check import check_project

        _print_json(check_project(Path(args.project_dir)))
        return
    if args.command == "register":
        rooted = LabBOService(projects_root=args.projects_root)
        if args.path:
            try:
                _print_json(
                    rooted.register_project(
                        Path(args.path).resolve(),
                        handle=args.handle,
                        drop_config_keys=args.drop_config_key,
                        set_config=_key_values(args.set_config) or None,
                    )
                )
            except ProjectCheckError as exc:
                _print_json({"registered": False, "error": str(exc), "check": exc.report})
                raise SystemExit(1) from exc
            return
        _print_json(_register_scan(rooted, Path(args.scan).resolve(), dry_run=bool(args.dry_run)))
        return
    if args.command == "descriptors":
        aliases = dict(item.split("=", 1) for item in args.alias)
        try:
            _print_json(
                service.add_descriptor_table(
                    args.project_dir,
                    variable=args.variable,
                    value_column=args.value_column,
                    csv_text=Path(args.csv).read_text(encoding="utf-8-sig"),
                    aliases=aliases,
                    replace=bool(args.replace),
                )
            )
        except DescriptorTableError as exc:
            _print_json({"changed": False, "error": str(exc), "report": exc.report})
            raise SystemExit(1) from exc
        return
    if args.command == "import-evidence":
        text = Path(args.file).read_text(encoding="utf-8-sig")
        is_csv = Path(args.file).suffix.lower() == ".csv"
        try:
            _print_json(
                service.import_evidence(
                    args.project_dir,
                    jsonl=None if is_csv else text,
                    csv_text=text if is_csv else None,
                    mode=args.mode,
                )
            )
        except EvidenceImportError as exc:
            _print_json({"imported": False, "error": str(exc), "problems": exc.problems})
            raise SystemExit(1) from exc
        return
    if args.command == "evidence-preview":
        preview = service.evidence_preview(args.project_dir)
        if args.json:
            _print_json(preview)
        else:
            _print_evidence_preview(preview)
        return
    if args.command == "evidence-prepare":
        from chem_agent_bo.lab.literature import LiteratureError, prepare_sources

        pdfs = [Path(item) for item in args.pdf]
        for folder in args.pdf_dir:
            pdfs.extend(sorted(Path(folder).glob("*.pdf")))
        if not pdfs:
            raise SystemExit("Give at least one --pdf or a --pdf-dir that contains PDFs.")
        try:
            result = prepare_sources(service.project_path(args.project_dir), pdfs)
        except LiteratureError as exc:
            raise SystemExit(str(exc)) from exc
        _print_prepare_result(result)
        return
    if args.command == "evidence-verify":
        from chem_agent_bo.lab.literature import LiteratureError, verify_drafts

        try:
            report = verify_drafts(service.project_path(args.project_dir))
        except LiteratureError as exc:
            raise SystemExit(str(exc)) from exc
        if args.json:
            _print_json(report)
        else:
            _print_verify_report(report)
        if report["rejected"] or report["unreadable_lines"]:
            raise SystemExit(1)
        return
    if args.command == "evidence-sheet":
        from chem_agent_bo.lab.literature import LiteratureError, build_sheet

        try:
            result = build_sheet(service.project_path(args.project_dir), force=args.force)
        except LiteratureError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"Wrote {result['row_count']} row(s) to {result['sheet_path']} ({result['with_checks']} with checks to read).")
        if result["skipped"]:
            print(f"{result['skipped']} draft(s) left out because they were rejected or duplicates; see evidence-verify.")
        print("Give it to the chemist: set `decision` to accept or reject on each row (edit the wording if needed).")
        return
    if args.command == "evidence-accept":
        from chem_agent_bo.lab.literature import LiteratureError

        try:
            result = service.accept_evidence_sheet(args.project_dir, sheet_path=args.sheet or None)
        except EvidenceImportError as exc:
            _print_json({"imported": False, "error": str(exc), "problems": exc.problems})
            raise SystemExit(1) from exc
        except LiteratureError as exc:
            _print_json({"imported": False, "error": str(exc), "problems": getattr(exc, "problems", [])})
            raise SystemExit(1) from exc
        print(
            f"Accepted {result['accepted']}, rejected {result['rejected']}, undecided {result['undecided']}. "
            f"Imported {result['imported_count']} card(s)."
        )
        for card_id in result["card_ids"]:
            print(f"  + {card_id}")
        if result["imported_count"]:
            print(f"Project now has {result['card_count']} card(s). Previous file backed up in {result['backup_dir']}")
            print("Run evidence-preview to see which cards the controller will now be shown.")
        return
    if args.command == "unregister":
        _print_json(LabBOService(projects_root=args.projects_root).unregister_project(args.handle))
        return
    raise ValueError(f"Unknown command: {args.command}")


def _print_verify_report(report: dict[str, Any]) -> None:
    print(
        f"Checked {report['checked']} draft(s): {report['ok']} ok ({report['with_warnings']} with warnings), "
        f"{report['rejected']} rejected, {report['duplicate']} duplicate.\n"
    )
    for item in report["unreadable_lines"]:
        print(f"  UNREADABLE {item['file']} line {item['line']}: {item['message']}")
    for record in report["records"]:
        if record["status"] == "ok" and not record["warnings"]:
            continue
        print(f"{record['status'].upper():<9} {record['card_id'] or '(no id)'}")
        for item in record["errors"]:
            print(f"    error   [{item['code']}] {item['message']}")
        for item in record["warnings"]:
            print(f"    warning [{item['code']}] {item['message']}")
    if report["by_code"]:
        print("\nBy code: " + ", ".join(f"{code} x{count}" for code, count in report["by_code"].items()))
    print(f"\nFull results: {report['verified_path']}")


def _print_prepare_result(result: dict[str, Any]) -> None:
    print(f"Work folder: {result['work_dir']}")
    print(f"Project context: {result['context']}\n")
    for item in result["sources"]:
        if item["status"] == "ready":
            pages = f"{item['page_count']} page(s)"
            refs = f", references from page {item['references_from_page']}" if item["references_from_page"] else ""
            print(f"  ready      {item['source_id']}  {pages}{refs}, {len(item['packets'])} packet(s), doi {item['doi'] or '-'}")
        else:
            print(f"  {item['status']:<10} {item['file']}: {item.get('note') or item.get('error', '')}")
    print(f"\n{result['ready_count']} of {len(result['sources'])} paper(s) ready to draft from (packets/ folder).")


def _print_evidence_preview(preview: dict[str, Any]) -> None:
    print(f"Project scope: {preview['reaction_scope']}")
    print(
        f"{preview['card_count']} card(s); selection `{preview['selection']}`; each ask retrieves the top "
        f"{preview['top_k']} ({preview['top_k_source']}), the agent reads {preview['prompt_max_items']} of them, "
        f"{preview['prompt_card_chars']} characters each.\n"
    )
    print("Shown to the controller (one fixed set per ask, used by every node):")
    for item in preview["retrieved"]:
        slots = f" [for: {', '.join(item['slots'])}]" if item.get("slots") else ""
        unread = "  (NOT read: beyond the agent's limit)" if not item.get("read_by_agent", True) else ""
        print(f"  {item['rank']:>3}  {item['score']:>6}  {item['mapping_status']:<20} {item['card_id']}{slots}{unread}")
        text = str(item.get("prompt_text", "")).replace("\n", " ")
        if text:
            cut = f"  [+{item['hidden_chars']} characters not seen]" if item.get("hidden_chars") else ""
            print(f"        the agent reads: {text}{cut}")
    if preview["not_shown"]:
        nearest = preview["not_shown"][0]
        print(
            f"\nEligible but not shown: {len(preview['not_shown'])} card(s); "
            f"the best of them ranks {nearest['rank']} with score {nearest['score']}."
        )
    if preview["never_retrieved"]:
        print("\nNever shown:")
        for item in preview["never_retrieved"]:
            print(f"  {item['card_id']} ({item['mapping_status']}): {item['reason_text']}")
    for line in preview.get("warnings", []):
        print(f"\nWARNING: {line}")


def _register_scan(service: LabBOService, folder: Path, *, dry_run: bool) -> dict[str, Any]:
    """Check (and unless dry_run, register) every project folder under `folder`."""
    from chem_agent_bo.lab.project_check import check_project

    found = sorted(
        path.parent
        for path in folder.rglob("project.yaml")
        if not any(part == "_backups" or part.startswith(".") for part in path.relative_to(folder).parts)
    )
    handles = [default_handle(path) for path in found]
    results = []
    for path, handle in zip(found, handles):
        item: dict[str, Any] = {"project_dir": str(path), "handle": handle}
        existing = service.registry.handle_for(path)
        if existing:
            item.update(status="already_registered", handle=existing)
        elif handles.count(handle) > 1:
            item.update(status="handle_conflict", message="Several folders share this name; register them one by one with --path and --handle.")
        else:
            report = check_project(path)
            item.update(errors=report["errors"], warning_count=len(report["warnings"]))
            if not report["ok"]:
                item["status"] = "check_failed"
            elif dry_run:
                item["status"] = "would_register"
            else:
                try:
                    service.register_project(path, handle=handle)
                    item["status"] = "registered"
                except Exception as exc:  # noqa: BLE001 -- keep going with the other folders
                    item.update(status="failed", message=f"{type(exc).__name__}: {exc}")
        results.append(item)
    counts: dict[str, int] = {}
    for item in results:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {"scanned": str(folder), "dry_run": dry_run, "counts": counts, "projects": results}


def _config_from_args(args: argparse.Namespace) -> ProjectConfig:
    return ProjectConfig(
        project_id=args.project_id,
        reaction_name=args.reaction_name,
        objective_name=args.objective_name,
        goal=args.goal,
        batch_size=args.batch_size,
        planner_name=args.planner_name,
        controller_mode=args.controller_mode,
        agent_config_path=args.agent_config,
        planner_use_descriptors=bool(args.planner_use_descriptors),
        seed=args.seed,
        reaction_scope=args.reaction_scope,
    )


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _key_values(items: list[str]) -> dict[str, Any]:
    """KEY=VALUE pairs; VALUE is JSON when it parses (4, true, {...}), otherwise the plain string."""
    values: dict[str, Any] = {}
    for item in items:
        key, sep, raw = str(item).partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--set-config needs KEY=VALUE, got `{item}`.")
        try:
            values[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            values[key.strip()] = raw
    return values


if __name__ == "__main__":
    main()
