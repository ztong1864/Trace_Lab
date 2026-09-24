"""File-backed real-lab TRACE project state."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from chem_agent_bo.lab.design_space import DesignSpace, _parse_bool


@dataclass
class ProjectConfig:
    project_id: str
    reaction_name: str = "untitled_reaction"
    objective_name: str = "yield"
    goal: str = "maximize"
    batch_size: int = 6
    planner_name: str = "atlas"
    # Honored by the "atlas" and "chunked_gp" planners, which each validate it
    # against their own supported set when a batch is requested.
    acquisition_function: str = "ei"
    # When the planner fails, either fall back to random candidates (flagged with
    # a warning) or, if False, fail the ask without writing a batch.
    allow_random_fallback: bool = True
    seed: int = 7
    reaction_scope: str = ""
    evidence_file: str = "evidence_cards.jsonl"
    controller_mode: str = "agentic"
    agent_config_path: str = "configs/agent_bo.yaml"
    planner_use_descriptors: bool = False
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "ProjectConfig":
        now = _now_iso()
        created = self.created_at or now
        return ProjectConfig(
            project_id=_safe_project_id(self.project_id),
            reaction_name=str(self.reaction_name or "untitled_reaction"),
            objective_name=str(self.objective_name or "yield"),
            goal="minimize" if str(self.goal).lower().startswith("min") else "maximize",
            batch_size=max(1, int(self.batch_size or 1)),
            planner_name=str(self.planner_name or "atlas").strip().lower(),
            acquisition_function=str(self.acquisition_function or "ei").strip().lower(),
            allow_random_fallback=_parse_bool(self.allow_random_fallback, default=True),
            seed=int(self.seed or 7),
            reaction_scope=str(self.reaction_scope or ""),
            evidence_file=str(self.evidence_file or "evidence_cards.jsonl"),
            controller_mode=(
                "bo_only"
                if str(self.controller_mode or "agentic").strip().lower() == "bo_only"
                else "agentic"
            ),
            agent_config_path=str(self.agent_config_path or "configs/agent_bo.yaml"),
            planner_use_descriptors=_parse_bool(self.planner_use_descriptors, default=False),
            created_at=created,
            updated_at=now,
            metadata=dict(self.metadata or {}),
        )


@dataclass
class RecommendationBatch:
    round_id: str
    recommendations: list[dict[str, Any]]
    trace_records: list[dict[str, Any]]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "created_at": self.created_at,
            "recommendations": self.recommendations,
            "trace_records": self.trace_records,
        }


class ObservationTable:
    """CSV-backed observation table for completed, failed, and skipped runs."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = list(rows or [])

    @classmethod
    def load(cls, path: str | Path) -> "ObservationTable":
        path_obj = Path(path)
        if not path_obj.exists():
            return cls([])
        with path_obj.open("r", encoding="utf-8-sig", newline="") as handle:
            return cls(list(csv.DictReader(handle)))

    def write(self, path: str | Path, *, variable_names: list[str], objective_name: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        standard_fields = {
            "observation_id",
            "round_id",
            "recommendation_id",
            "source",
            "stage",
            "status",
            objective_name,
            "failure_reason",
            "notes",
            "observed_at",
            "chemist_override",
            *variable_names,
        }
        extra_fields = sorted(
            {
                key
                for row in self.rows
                for key in row
                if key and key not in standard_fields
            }
        )
        fieldnames = [
            "observation_id",
            "round_id",
            "recommendation_id",
            "source",
            "stage",
            *variable_names,
            objective_name,
            "status",
            "failure_reason",
            "notes",
            "observed_at",
            "chemist_override",
            *extra_fields,
        ]
        with path_obj.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def completed_rows(self, objective_name: str) -> list[dict[str, Any]]:
        completed = []
        for row in self.rows:
            if str(row.get("status", "completed")).strip().lower() != "completed":
                continue
            if _clean(row.get(objective_name)) == "":
                continue
            completed.append(row)
        return completed

    def candidate_keys(self, variable_names: list[str]) -> set[tuple[str, ...]]:
        keys = set()
        for row in self.rows:
            if str(row.get("status", "")).strip().lower() in {"failed", "skipped"}:
                continue
            keys.add(tuple(str(row.get(name, "")) for name in variable_names))
        return keys

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            source = str(row.get("source") or "unspecified").strip() or "unspecified"
            counts[source] = counts.get(source, 0) + 1
        return counts

    def historical_rows(self) -> list[dict[str, Any]]:
        return [
            row
            for row in self.rows
            if str(row.get("source") or "").strip().lower() == "historical"
        ]

    def best_so_far(self, *, objective_name: str, goal: str) -> float | None:
        values: list[float] = []
        for row in self.completed_rows(objective_name):
            try:
                values.append(float(row[objective_name]))
            except (TypeError, ValueError):
                continue
        if not values:
            return None
        return min(values) if goal == "minimize" else max(values)


class LabProject:
    """A real-lab TRACE project stored as a directory."""

    def __init__(self, project_dir: str | Path) -> None:
        self.project_dir = Path(project_dir)

    @classmethod
    def create(
        cls,
        project_dir: str | Path,
        *,
        config: ProjectConfig,
        design_space: DesignSpace,
        overwrite: bool = False,
    ) -> "LabProject":
        project = cls(project_dir)
        if project.project_dir.exists() and not overwrite and project.config_path.exists():
            raise FileExistsError(f"Lab project already exists: {project.project_dir}")
        project.project_dir.mkdir(parents=True, exist_ok=True)
        normalized = config.normalized()
        project.write_config(normalized)
        design_space.write_csv(project.design_space_path)
        ObservationTable([]).write(
            project.observations_path,
            variable_names=design_space.condition_names,
            objective_name=normalized.objective_name,
        )
        if not project.evidence_path.exists():
            project.evidence_path.write_text("", encoding="utf-8")
        return project

    @property
    def config_path(self) -> Path:
        return self.project_dir / "project.yaml"

    @property
    def design_space_path(self) -> Path:
        return self.project_dir / "design_space.csv"

    @property
    def observations_path(self) -> Path:
        return self.project_dir / "observations.csv"

    @property
    def evidence_path(self) -> Path:
        config = self.load_config(required=False)
        filename = config.evidence_file if config is not None else "evidence_cards.jsonl"
        return self.project_dir / filename

    def load_config(self, *, required: bool = True) -> ProjectConfig | None:
        if not self.config_path.exists():
            if required:
                raise FileNotFoundError(f"Missing lab project config: {self.config_path}")
            return None
        with self.config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return ProjectConfig(**payload).normalized()

    def write_config(self, config: ProjectConfig) -> None:
        normalized = config.normalized()
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(asdict(normalized), handle, sort_keys=False, allow_unicode=True)

    def load_design_space(self) -> DesignSpace:
        return DesignSpace.read_csv(self.design_space_path)

    def load_observations(self) -> ObservationTable:
        return ObservationTable.load(self.observations_path)

    def write_observations(self, observations: ObservationTable) -> None:
        config = self.load_config()
        design = self.load_design_space()
        observations.write(
            self.observations_path,
            variable_names=design.condition_names,
            objective_name=config.objective_name,
        )
        config.updated_at = _now_iso()
        self.write_config(config)

    def next_round_id(self) -> str:
        existing = sorted(self.project_dir.glob("recommendations_round_*.json"))
        return f"round_{len(existing) + 1:03d}"

    def recommendation_json_path(self, round_id: str) -> Path:
        return self.project_dir / f"recommendations_{round_id}.json"

    def recommendation_csv_path(self, round_id: str) -> Path:
        return self.project_dir / f"recommendations_{round_id}.csv"

    def trace_path(self, round_id: str) -> Path:
        return self.project_dir / f"trace_{round_id}.jsonl"

    def write_batch(self, batch: RecommendationBatch, *, variable_names: list[str]) -> None:
        json_path = self.recommendation_json_path(batch.round_id)
        csv_path = self.recommendation_csv_path(batch.round_id)
        trace_path = self.trace_path(batch.round_id)
        json_path.write_text(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        _write_recommendation_csv(csv_path, batch.recommendations, variable_names=variable_names)
        with trace_path.open("w", encoding="utf-8") as handle:
            for record in batch.trace_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_batches(self) -> list[RecommendationBatch]:
        batches: list[RecommendationBatch] = []
        for path in sorted(self.project_dir.glob("recommendations_round_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            recommendations = list(payload.get("recommendations") or [])
            trace_records = list(payload.get("trace_records") or [])
            _backfill_planner_errors(recommendations, trace_records)
            batches.append(
                RecommendationBatch(
                    round_id=str(payload["round_id"]),
                    created_at=str(payload.get("created_at") or ""),
                    recommendations=recommendations,
                    trace_records=trace_records,
                )
            )
        return batches

    def find_recommendation(self, recommendation_id: str) -> tuple[RecommendationBatch, dict[str, Any]] | None:
        for batch in self.load_batches():
            for recommendation in batch.recommendations:
                if str(recommendation.get("recommendation_id")) == str(recommendation_id):
                    return batch, recommendation
        return None

    def summary(self) -> dict[str, Any]:
        config = self.load_config()
        design = self.load_design_space()
        observations = self.load_observations()
        batches = self.load_batches()
        return {
            "project": asdict(config),
            "design_space": design.describe(),
            "observation_count": len(observations.rows),
            "completed_observation_count": len(observations.completed_rows(config.objective_name)),
            "historical_observation_count": len(observations.historical_rows()),
            "observation_source_counts": observations.source_counts(),
            "best_so_far": observations.best_so_far(
                objective_name=config.objective_name,
                goal=config.goal,
            ),
            "recommendation_rounds": [batch.round_id for batch in batches],
        }


def _write_recommendation_csv(
    path: str | Path,
    recommendations: list[dict[str, Any]],
    *,
    variable_names: list[str],
) -> None:
    base_fields = [
        "round_id",
        "recommendation_id",
        "rank",
        "status",
        *variable_names,
        "rationale",
        "planner_name",
        "controller_action",
        "batch_role",
        "created_at",
        "result",
        "failure_reason",
        "notes",
    ]
    path_obj = Path(path)
    with path_obj.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields)
        writer.writeheader()
        for item in recommendations:
            candidate = item.get("candidate") or {}
            row = {field: item.get(field, "") for field in base_fields}
            for name in variable_names:
                row[name] = candidate.get(name, row.get(name, ""))
            writer.writerow(row)


def _backfill_planner_errors(
    recommendations: list[dict[str, Any]],
    trace_records: list[dict[str, Any]],
) -> None:
    """Copy `planner_error` from trace records onto recommendations that predate the field.

    Batches written before recommendations carried `planner_error` only recorded it in
    their traces, so a random-fallback batch would otherwise look like a normal one.
    """
    errors = {
        str(record.get("recommendation_id")): str(record.get("planner_error") or "")
        for record in trace_records
        if record.get("recommendation_id")
    }
    for recommendation in recommendations:
        if "planner_error" not in recommendation:
            recommendation["planner_error"] = errors.get(str(recommendation.get("recommendation_id")), "")


def _safe_project_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    if not cleaned:
        raise ValueError("project_id cannot be empty.")
    return cleaned


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
