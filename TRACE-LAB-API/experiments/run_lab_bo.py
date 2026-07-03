"""CLI for TRACE real-lab ask/tell projects."""

from __future__ import annotations

import argparse
import json
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
from chem_agent_bo.lab.service import LabBOService, read_results_csv


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
    ask_parser.add_argument("--planner-name", choices=("atlas", "random"), default=None)
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
    return parser.parse_args()


def _add_project_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--reaction-name", default="oxidative_esterification")
    parser.add_argument("--objective-name", default="yield")
    parser.add_argument("--goal", choices=("maximize", "minimize"), default="maximize")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--planner-name", choices=("atlas", "random"), default="atlas")
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
    raise ValueError(f"Unknown command: {args.command}")


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
