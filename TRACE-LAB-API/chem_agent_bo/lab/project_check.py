"""Consistency check for a lab project folder, run before it is registered or used.

`check_project` never raises for problems in the project itself: everything it
finds goes into `errors` (the project can't be used safely as it is) or
`warnings` (usable, but worth a look), each naming the file and, where there is
one, the CSV line. `facts` holds counts a person can compare with a README.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from chem_agent_bo.lab.design_space import DesignSpace
from chem_agent_bo.lab.evidence import (
    ALLOWED_MAPPING_STATUSES,
    SCREEN_REASONS,
    EvidenceStore,
    _card_from_dict,
    retrieval_preview,
)
from chem_agent_bo.lab.project import ObservationTable, ProjectConfig
from chem_agent_bo.lab.service import (
    LAB_DESCRIPTOR_PLANNERS,
    _canonical_design_value,
    _descriptor_ineligibility_error,
    _estimated_design_space_size,
    _validate_project_config,
)

# Above this many combinations Atlas's optimizer takes many minutes or never finishes.
ATLAS_LARGE_SPACE = 100_000
OBSERVATION_STATUSES = {"completed", "failed", "skipped", "pending"}
_ROUND_FILE = re.compile(r"^recommendations_round_(\d+)\.json$")
_HISTORICAL_ROW = re.compile(r"^Historical row \d+ ")


class _Report:
    def __init__(self) -> None:
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.facts: dict[str, Any] = {}

    def error(self, file: str, message: str, line: int | None = None) -> None:
        self.errors.append(_item(file, message, line))

    def warn(self, file: str, message: str, line: int | None = None) -> None:
        self.warnings.append(_item(file, message, line))


def _item(file: str, message: str, line: int | None) -> dict[str, Any]:
    item: dict[str, Any] = {"file": file, "message": message}
    if line is not None:
        item["line"] = line
    return item


def check_project(project_dir: str | Path) -> dict[str, Any]:
    folder = Path(project_dir)
    report = _Report()
    if not folder.is_dir():
        report.error(str(folder), "Project folder does not exist.")
        return _result(folder, report)
    config = _check_config(folder, report)
    design = _check_design_space(folder, config, report)
    if config is not None and design is not None:
        _check_observations(folder, config, design, report)
        _check_batches(folder, design, report)
    _check_evidence(folder, config, design, report)
    return _result(folder, report)


def _result(folder: Path, report: _Report) -> dict[str, Any]:
    return {
        "project_dir": str(folder),
        "ok": not report.errors,
        "errors": report.errors,
        "warnings": report.warnings,
        "facts": report.facts,
    }


def _check_config(folder: Path, report: _Report) -> ProjectConfig | None:
    path = folder / "project.yaml"
    if not path.exists():
        report.error("project.yaml", "Missing project.yaml.")
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        report.error("project.yaml", f"Not valid YAML: {exc}")
        return None
    if not isinstance(payload, dict):
        report.error("project.yaml", "Expected a mapping of settings.")
        return None
    known = {field.name for field in fields(ProjectConfig)}
    unknown = sorted(set(payload) - known)
    if unknown:
        report.error(
            "project.yaml",
            f"Unknown settings {', '.join(f'`{key}`' for key in unknown)} (not used by this TRACE "
            "version, e.g. from another TRACE fork). Remove them, or register with "
            f"drop_config_keys={unknown}.",
        )
    try:
        config = ProjectConfig(**{key: value for key, value in payload.items() if key in known}).normalized()
    except Exception as exc:  # noqa: BLE001 -- report any malformed value
        report.error("project.yaml", f"Invalid settings: {type(exc).__name__}: {exc}")
        return None
    try:
        _validate_project_config(config, project_dir=folder)
    except ValueError as exc:
        report.error("project.yaml", str(exc))
    report.facts.update(
        {
            "project_id": config.project_id,
            "objective_name": config.objective_name,
            "planner_name": config.planner_name,
            "acquisition_function": config.acquisition_function,
            "controller_mode": config.controller_mode,
            "planner_use_descriptors": config.planner_use_descriptors,
        }
    )
    return config


def _check_design_space(folder: Path, config: ProjectConfig | None, report: _Report) -> DesignSpace | None:
    path = folder / "design_space.csv"
    if not path.exists():
        report.error("design_space.csv", "Missing design_space.csv.")
        return None
    try:
        design = DesignSpace.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        report.error("design_space.csv", f"Can't read the design space: {type(exc).__name__}: {exc}")
        return None
    combinations = _estimated_design_space_size(design)
    report.facts["variables"] = {
        variable.name: (len(variable.options) if variable.kind != "continuous" else "continuous")
        for variable in design.variables
        if variable.optimization_active
    }
    report.facts["combination_count"] = combinations
    if config is None:
        return design
    if config.planner_use_descriptors and config.planner_name in LAB_DESCRIPTOR_PLANNERS:
        message = _descriptor_ineligibility_error(
            design.planner_descriptor_eligibility(variable_names=design.variable_names)
        )
        if message:
            report.error("design_space.csv", message)
    if config.planner_name == "atlas" and combinations is not None and combinations > ATLAS_LARGE_SPACE:
        report.warn(
            "project.yaml",
            f"Atlas on {combinations:,} combinations can take many minutes or not finish; "
            "planner_name: chunked_gp scores every combination in seconds.",
        )
    return design


def _check_observations(folder: Path, config: ProjectConfig, design: DesignSpace, report: _Report) -> None:
    path = folder / "observations.csv"
    if not path.exists():
        report.warn("observations.csv", "No observations.csv; the project has no results yet.")
        report.facts.update({"observation_count": 0, "completed_observation_count": 0})
        return
    try:
        table = ObservationTable.load(path)
    except Exception as exc:  # noqa: BLE001
        report.error("observations.csv", f"Can't read observations: {type(exc).__name__}: {exc}")
        return
    rows = table.rows
    statuses: Counter[str] = Counter()
    completed = 0
    if rows and config.objective_name not in rows[0]:
        report.error("observations.csv", f"No `{config.objective_name}` column (the project's objective).")
    for index, row in enumerate(rows):
        line = index + 2  # header is line 1
        status = str(row.get("status") or "").strip().lower() or "completed"
        statuses[status] += 1
        if status not in OBSERVATION_STATUSES:
            report.error("observations.csv", f"Unknown status `{status}`.", line)
            continue
        if status == "completed":
            value = str(row.get(config.objective_name) or "").strip()
            try:
                float(value)
                completed += 1
            except ValueError:
                report.error(
                    "observations.csv",
                    f"Completed row needs a numeric `{config.objective_name}`, got `{value}`.",
                    line,
                )
        if status in {"failed", "skipped"}:
            continue
        for name in design.variable_names:
            try:
                _canonical_design_value(
                    design, name, str(row.get(name) or "").strip(), strict=True, row_index=line
                )
            except ValueError as exc:
                report.error("observations.csv", _value_problem(exc), line)
    report.facts.update(
        {
            "observation_count": len(rows),
            "completed_observation_count": completed,
            "observation_statuses": dict(statuses),
        }
    )


def _check_batches(folder: Path, design: DesignSpace, report: _Report) -> None:
    numbers: list[int] = []
    pending = 0
    for path in sorted(folder.glob("recommendations_round_*.json")):
        match = _ROUND_FILE.match(path.name)
        if not match:
            continue
        numbers.append(int(match.group(1)))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.error(path.name, f"Can't read the batch: {exc}")
            continue
        round_id = f"round_{int(match.group(1)):03d}"
        if str(payload.get("round_id") or "") != round_id:
            report.error(path.name, f"round_id is `{payload.get('round_id')}`, expected `{round_id}`.")
        if not (folder / f"trace_{round_id}.jsonl").exists():
            report.warn(f"trace_{round_id}.jsonl", "Missing trace file for this round.")
        for recommendation in payload.get("recommendations") or []:
            rec_id = str(recommendation.get("recommendation_id") or "?")
            status = str(recommendation.get("status") or "pending").strip().lower()
            if status == "pending":
                pending += 1
            if status not in OBSERVATION_STATUSES:
                report.error(path.name, f"{rec_id}: unknown status `{status}`.")
            candidate = recommendation.get("candidate") or {}
            for name in design.variable_names:
                try:
                    _canonical_design_value(
                        design, name, str(candidate.get(name) or "").strip(), strict=True, row_index=0
                    )
                except ValueError as exc:
                    message = f"{rec_id}: {_value_problem(exc)}"
                    # A pending recommendation outside the design space can't be told back.
                    (report.error if status == "pending" else report.warn)(path.name, message)
    expected = list(range(1, len(numbers) + 1))
    if sorted(numbers) != expected:
        report.error(
            "recommendations_round_*.json",
            f"Round numbers {sorted(numbers)} are not 1..{len(numbers)} without gaps; "
            "the next ask would reuse an existing round id and overwrite that batch.",
        )
    report.facts.update({"recommendation_rounds": len(numbers), "pending_recommendation_count": pending})


def _check_evidence(
    folder: Path, config: ProjectConfig | None, design: DesignSpace | None, report: _Report
) -> None:
    filename = config.evidence_file if config is not None else "evidence_cards.jsonl"
    path = folder / filename
    if not path.exists():
        report.facts["evidence_card_count"] = 0
        return
    try:
        cards = _read_raw_cards(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.error(filename, f"Can't read evidence cards: {exc}")
        return
    known_variables = {variable.name for variable in design.variables} if design is not None else set()
    ids: Counter[str] = Counter()
    # Summarized per issue rather than per card: projects commonly have 100+ cards.
    bad_statuses: dict[str, list[str]] = {}
    unknown_scope: dict[str, list[str]] = {}
    for line, card in cards:
        card_id = str(card.get("card_id") or card.get("id") or "").strip()
        if not card_id:
            report.error(filename, "Card without card_id.", line)
            continue
        ids[card_id] += 1
        status = str(card.get("mapping_status") or "background").strip().lower()
        if status not in ALLOWED_MAPPING_STATUSES:
            bad_statuses.setdefault(status, []).append(card_id)
        scope = card.get("variable_scope") or []
        names = [item.strip() for item in scope.replace(";", ",").split(",")] if isinstance(scope, str) else scope
        for name in names:
            name = str(name).strip()
            if name and known_variables and name not in known_variables:
                unknown_scope.setdefault(name, []).append(card_id)
    for status, card_ids in sorted(bad_statuses.items()):
        report.warn(
            filename,
            f"{len(card_ids)} card(s) have mapping_status `{status}`, which is not one of "
            f"{sorted(ALLOWED_MAPPING_STATUSES)} and is treated as `background` "
            f"(e.g. {', '.join(card_ids[:3])}).",
        )
    for name, card_ids in sorted(unknown_scope.items()):
        report.warn(
            filename,
            f"{len(card_ids)} card(s) list `{name}` in variable_scope, which is not a design variable "
            f"(e.g. {', '.join(card_ids[:3])}).",
        )
    for card_id, count in sorted(ids.items()):
        if count > 1:
            report.error(filename, f"card_id `{card_id}` appears {count} times.")
    report.facts["evidence_card_count"] = len(cards)
    if design is not None and config is not None:
        _warn_unretrievable_cards(filename, cards, design, config, report)


def _warn_unretrievable_cards(
    filename: str, cards: list[tuple[int, dict[str, Any]]], design: DesignSpace, config: ProjectConfig, report: _Report
) -> None:
    """Warn about cards the controller can never see by accident (not the deliberately blocked ones)."""
    store = EvidenceStore([_card_from_dict(card) for _line, card in cards])
    preview = retrieval_preview(
        store, variables=design.variable_names, reaction_scope=config.reaction_scope, top_k=len(cards) or 1
    )
    by_reason: dict[str, list[str]] = {}
    for item in preview["never_retrieved"]:
        if item["reason"] in {"no_variable_overlap", "target_node_mismatch", "reaction_scope_mismatch"}:
            by_reason.setdefault(item["reason"], []).append(item["card_id"])
    for reason, card_ids in sorted(by_reason.items()):
        report.warn(
            filename,
            f"{len(card_ids)} card(s) can never be shown to the controller: {SCREEN_REASONS[reason]} "
            f"(e.g. {', '.join(card_ids[:3])}). See the evidence preview.",
        )


def _value_problem(exc: Exception) -> str:
    """Reword `_canonical_design_value` errors, which are phrased for imported rows."""
    message = _HISTORICAL_ROW.sub("", str(exc))
    for verb in ("is ", "has "):
        if message.startswith(verb):
            return message[len(verb):]
    return message


def _read_raw_cards(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        cards = []
        for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if text.strip():
                cards.append((number, json.loads(text)))
        return cards
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [(index + 2, dict(row)) for index, row in enumerate(csv.DictReader(handle))]
    raise ValueError(f"Unsupported evidence card file type: {path.name}")

