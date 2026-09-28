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

from chem_agent_bo.lab.design_space import DesignSpace
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
    register_parser.add_argument("--dry-run", action="store_true", help="With --scan: check only, register nothing.")

    unregister_parser = subparsers.add_parser("unregister", help="Forget a registered project (files untouched).")
    _add_projects_root_arg(unregister_parser)
    unregister_parser.add_argument("--handle", required=True)

    check_parser = subparsers.add_parser("check", help="Check a project folder for problems.")
    check_parser.add_argument("--project-dir", required=True)
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
                    )
                )
            except ProjectCheckError as exc:
                _print_json({"registered": False, "error": str(exc), "check": exc.report})
                raise SystemExit(1) from exc
            return
        _print_json(_register_scan(rooted, Path(args.scan).resolve(), dry_run=bool(args.dry_run)))
        return
    if args.command == "unregister":
        _print_json(LabBOService(projects_root=args.projects_root).unregister_project(args.handle))
        return
    raise ValueError(f"Unknown command: {args.command}")


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


if __name__ == "__main__":
    main()
