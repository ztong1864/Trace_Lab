"""Ask/tell service layer for real-lab TRACE projects."""

from __future__ import annotations

import csv
import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterContinuous, ParameterVector

from chem_agent_bo.bo.base import ExclusionConstraint
from chem_agent_bo.bo.registry import build_planner
from chem_agent_bo.config import load_agentic_bo_config
from chem_agent_bo.config.schema import AgenticBOConfig
from chem_agent_bo.lab.design_space import DesignSpace
from chem_agent_bo.lab.evidence import EvidenceStore
from chem_agent_bo.lab.project import (
    LabProject,
    ObservationTable,
    ProjectConfig,
    RecommendationBatch,
)
from chem_agent_bo.runtime import ActionCapabilityPolicy, ControllerRuntime, ControllerRuntimeConfig
from chem_agent_bo.runtime.lab_evidence import LabEvidenceProvider
from chem_agent_bo.utils.run_io import capture_third_party_output


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAB_SUPPORTED_PLANNERS = {"atlas", "chunked_gp", "random"}
# Planners that can use per-option numeric descriptors as GP inputs.
LAB_DESCRIPTOR_PLANNERS = {"atlas", "chunked_gp"}
LAB_EVIDENCE_TARGET_NODES = (
    "design_init_experiments",
    "stagnation_diagnosis",
    "hypothesis_action",
    "semantic_assessment",
    "verification_pass",
    "reflection_action",
    "lab_batch_composition",
)


class LabBOService:
    """Batch ask/tell service for wet-lab optimization."""

    def __init__(self, projects_root: str | Path | None = None) -> None:
        self.projects_root = Path(projects_root) if projects_root is not None else None

    def project_path(self, project_id_or_dir: str | Path) -> Path:
        raw = Path(project_id_or_dir)
        if raw.exists() or raw.is_absolute() or self.projects_root is None:
            return raw
        return self.projects_root / str(project_id_or_dir)

    def create_project(
        self,
        project_id_or_dir: str | Path,
        *,
        config: ProjectConfig,
        design_space: DesignSpace,
        overwrite: bool = False,
    ) -> LabProject:
        return LabProject.create(
            self.project_path(project_id_or_dir),
            config=config,
            design_space=design_space,
            overwrite=overwrite,
        )

    def reset_project(
        self,
        project_id_or_dir: str | Path,
        *,
        backup: bool = True,
        keep_historical: bool = False,
    ) -> dict[str, Any]:
        """Reset run state for one lab project while preserving its setup files.

        This is intended for local debugging. It keeps `project.yaml`,
        `design_space.csv`, and evidence cards, clears recommendations,
        traces, and local third-party logs, and can optionally preserve
        rows with `source=historical`.
        """

        project = LabProject(self.project_path(project_id_or_dir))
        config = project.load_config()
        _ = project.load_design_space()
        backup_dir: Path | None = None
        if backup:
            backup_root = project.project_dir.parent / "_backups"
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_dir = _unique_backup_dir(
                backup_root / f"{config.project_id}_reset_{_timestamp_for_path()}"
            )
            shutil.copytree(project.project_dir, backup_dir)

        removed_files: list[str] = []
        for pattern in (
            "recommendations_round_*.csv",
            "recommendations_round_*.json",
            "trace_round_*.jsonl",
        ):
            for path in project.project_dir.glob(pattern):
                if path.is_file():
                    removed_files.append(str(path))
                    path.unlink()

        observations = project.load_observations()
        kept_rows = observations.historical_rows() if keep_historical else []
        project.write_observations(ObservationTable(kept_rows))
        log_path = project.project_dir / "logs" / "third_party.log"
        if log_path.exists():
            log_path.write_text("", encoding="utf-8")
        return {
            "project_id": config.project_id,
            "backup_dir": "" if backup_dir is None else str(backup_dir),
            "removed_files": removed_files,
            "cleared_observations": len(observations.rows) - len(kept_rows) > 0,
            "keep_historical": bool(keep_historical),
            "preserved_historical_observation_count": len(kept_rows),
            "removed_observation_count": len(observations.rows) - len(kept_rows),
            "summary": LabProject(project.project_dir).summary(),
            "preserved_files": [
                str(project.config_path),
                str(project.design_space_path),
                str(project.evidence_path),
            ],
        }

    def ask(
        self,
        project_id_or_dir: str | Path,
        *,
        batch_size: int | None = None,
        planner_name: str | None = None,
        controller_mode: str | None = None,
        agent_config_path: str | None = None,
        planner_use_descriptors: bool | None = None,
    ) -> dict[str, Any]:
        project = LabProject(self.project_path(project_id_or_dir))
        config = project.load_config()
        design_space = project.load_design_space()
        observations = project.load_observations()
        active_variables = design_space.variable_names
        condition_variables = design_space.condition_names
        effective_batch_size = max(1, int(batch_size or config.batch_size))
        effective_planner_name = str(planner_name or config.planner_name).strip().lower()
        effective_controller_mode = str(
            controller_mode or config.controller_mode or "agentic"
        ).strip().lower()
        if effective_controller_mode not in {"agentic", "bo_only"}:
            raise ValueError(
                "Lab controller_mode must be `agentic` or `bo_only`; "
                f"got `{effective_controller_mode}`."
            )
        if effective_planner_name not in LAB_SUPPORTED_PLANNERS:
            raise ValueError(
                f"Lab mode supports {sorted(LAB_SUPPORTED_PLANNERS)} in the first release; "
                f"got `{effective_planner_name}`."
            )
        requested_planner_descriptors = (
            bool(config.planner_use_descriptors)
            if planner_use_descriptors is None
            else bool(planner_use_descriptors)
        )
        descriptor_eligibility = design_space.planner_descriptor_eligibility(
            variable_names=active_variables,
        )
        effective_planner_descriptors = (
            requested_planner_descriptors
            and effective_planner_name in LAB_DESCRIPTOR_PLANNERS
            and bool(descriptor_eligibility.get("eligible"))
        )

        param_space = design_space.param_space(
            include_descriptors=effective_planner_descriptors,
        )
        value_space = _value_space(config.objective_name)
        campaign = _campaign_from_observations(
            observations=observations,
            design_space=design_space,
            param_space=param_space,
            value_space=value_space,
            variable_names=active_variables,
            objective_name=config.objective_name,
        )
        banned_keys = _observation_candidate_keys(observations, design_space, active_variables)
        banned_keys.update(_pending_recommendation_keys(project, design_space, active_variables))
        constraints = (
            [_exclude_candidate_keys(active_variables, banned_keys, design_space=design_space)]
            if banned_keys
            else []
        )
        constraint_signature = f"lab_exclude:{len(banned_keys)}"
        round_id = project.next_round_id()
        candidate_pool_size = effective_batch_size
        agent_config: AgenticBOConfig | None = None
        if effective_controller_mode == "agentic":
            agent_config = _load_agentic_config(
                agent_config_path or config.agent_config_path,
                project_dir=project.project_dir,
            )
            candidate_pool_size = max(
                effective_batch_size,
                int(agent_config.orchestrator.shortlist_candidate_pool_size or 12),
                12,
            )
        finite_space_size = _estimated_design_space_size(design_space)
        if finite_space_size is not None:
            remaining = int(finite_space_size) - len(banned_keys)
            if remaining <= 0:
                raise RuntimeError(
                    "No feasible lab candidates remain after excluding completed "
                    "and pending recommendations."
                )
            candidate_pool_size = min(candidate_pool_size, remaining)
            effective_batch_size = min(effective_batch_size, remaining)

        planner_error = ""
        requested_planner_name = effective_planner_name
        third_party_log_path = project.project_dir / "logs" / "third_party.log"
        with capture_third_party_output(
            enabled=True,
            log_path=third_party_log_path,
            label=f"lab_planner_init:{round_id}:{effective_planner_name}",
        ):
            planner = build_planner(
                effective_planner_name,
                env=_LabPlannerEnv(goal=config.goal),
                init_budget=0,
                seed=config.seed,
                known_constraints=constraints,
                use_descriptors=effective_planner_descriptors,
                acquisition_type=config.acquisition_function,
            )
        try:
            with capture_third_party_output(
                enabled=True,
                log_path=third_party_log_path,
                label=f"lab_ask:{round_id}:{effective_planner_name}",
            ):
                raw_candidates = planner.suggest_shortlist(
                    observations=campaign.observations,
                    subspace=param_space,
                    shortlist_size=candidate_pool_size,
                    known_constraints=constraints,
                    known_constraints_signature=constraint_signature,
                )
        except Exception as exc:  # noqa: BLE001
            if effective_planner_name == "random":
                raise
            planner_error = f"{type(exc).__name__}: {exc}"
            if not config.allow_random_fallback:
                raise RuntimeError(
                    f"The {effective_planner_name} planner failed ({planner_error}); "
                    "no recommendations were written. Set `allow_random_fallback: true` "
                    "in project.yaml to accept random candidates instead."
                ) from exc
            with capture_third_party_output(
                enabled=True,
                log_path=third_party_log_path,
                label=f"lab_planner_init:{round_id}:random_fallback",
            ):
                fallback = build_planner(
                    "random",
                    env=_LabPlannerEnv(goal=config.goal),
                    init_budget=0,
                    seed=config.seed,
                    known_constraints=constraints,
                    use_descriptors=False,
                )
            with capture_third_party_output(
                enabled=True,
                log_path=third_party_log_path,
                label="lab_ask:fallback:random",
            ):
                fallback_param_space = design_space.param_space(include_descriptors=False)
                raw_candidates = fallback.suggest_shortlist(
                    observations=campaign.observations,
                    subspace=fallback_param_space,
                    shortlist_size=candidate_pool_size,
                    known_constraints=constraints,
                    known_constraints_signature=constraint_signature,
                )
            effective_planner_name = "random"
            effective_planner_descriptors = False
            param_space = fallback_param_space

        evidence_store = EvidenceStore.load(project.evidence_path)
        evidence_cards = evidence_store.applicable(
            variables=active_variables,
            reaction_scope=config.reaction_scope,
            target_nodes=list(LAB_EVIDENCE_TARGET_NODES),
            max_items=5,
        )
        created_at = _now_iso()
        candidate_pool = _candidate_pool_items(
            raw_candidates=raw_candidates,
            param_space=param_space,
            design_space=design_space,
            variable_names=active_variables,
            static_conditions=design_space.static_conditions(),
            max_items=candidate_pool_size,
            proposed_by=(
                None if planner_error else planner.planner_diagnostics().get("pick_acquisitions")
            ),
        )
        candidate_pool = _attach_descriptor_context(
            candidate_pool=candidate_pool,
            design_space=design_space,
            variable_names=active_variables,
        )
        if effective_controller_mode == "agentic":
            assert agent_config is not None
            decision_engine = _build_decision_engine(agent_config)
            runtime = _build_lab_controller_runtime(
                agent_config=agent_config,
                decision_engine=decision_engine,
                planner_name=effective_planner_name,
            )
            evidence_provider = LabEvidenceProvider(evidence_store)
            knowledge_units, knowledge_meta = evidence_provider.retrieve(
                variables=active_variables,
                reaction_scope=config.reaction_scope,
                target_nodes=list(LAB_EVIDENCE_TARGET_NODES),
                max_items=int(agent_config.orchestrator.knowledge_top_k or 5),
            )
            evidence_trace = {
                **knowledge_meta,
                "retrieved_units": knowledge_units,
            }
            decision = runtime.plan_batch(
                history=_history_from_observations(
                    observations=observations,
                    variable_names=active_variables,
                    objective_name=config.objective_name,
                    goal=config.goal,
                ),
                candidate_pool=candidate_pool,
                batch_size=effective_batch_size,
                search_space=param_space,
                search_space_meta=_lab_search_space_meta(
                    design_space,
                    planner_use_descriptors=effective_planner_descriptors,
                    planner_descriptor_eligibility=descriptor_eligibility,
                ),
                reaction_context=_lab_reaction_context(config),
                goal=config.goal,
                objective_name=config.objective_name,
                iteration=len(project.load_batches()) + 1,
                observations=len(observations.completed_rows(config.objective_name)),
                total_budget=None,
                knowledge_units=knowledge_units,
                knowledge_meta=knowledge_meta,
                evidence_trace=evidence_trace,
            )
            recommendations, trace_records = _recommendations_from_decision(
                round_id=round_id,
                decision=decision,
                planner_name=effective_planner_name,
                planner_diagnostics=planner.planner_diagnostics(),
                planner_error=planner_error,
                created_at=created_at,
            )
        else:
            recommendations, trace_records = _bo_only_recommendations(
                round_id=round_id,
                candidate_pool=candidate_pool,
                batch_size=effective_batch_size,
                planner_name=effective_planner_name,
                planner_diagnostics=planner.planner_diagnostics(),
                planner_error=planner_error,
                evidence_cards=evidence_cards,
                created_at=created_at,
            )

        if not recommendations:
            raise RuntimeError("No feasible lab recommendations could be generated.")

        batch = RecommendationBatch(
            round_id=round_id,
            recommendations=recommendations,
            trace_records=trace_records,
            created_at=created_at,
        )
        project.write_batch(batch, variable_names=condition_variables)
        return {
            "project": asdict(config),
            "round_id": round_id,
            "recommendations": recommendations,
            "trace_path": str(project.trace_path(round_id)),
            "recommendations_json_path": str(project.recommendation_json_path(round_id)),
            "recommendations_csv_path": str(project.recommendation_csv_path(round_id)),
            "third_party_log_path": str(third_party_log_path),
            "oracle_evaluation_disabled": True,
            "controller_mode": effective_controller_mode,
            "planner_use_descriptors": effective_planner_descriptors,
            "planner_descriptor_eligibility": descriptor_eligibility,
            "planner_fallback": _planner_fallback_summary(
                requested_planner_name=requested_planner_name,
                planner_error=planner_error,
            ),
            "planner_warnings": _planner_warnings(planner.planner_diagnostics()),
        }

    def import_observations(
        self,
        project_id_or_dir: str | Path,
        *,
        rows: list[dict[str, Any]],
        source: str = "historical",
        allow_duplicates: bool = False,
    ) -> dict[str, Any]:
        """Import measured lab observations that did not originate from TRACE recommendations."""

        if not rows:
            raise ValueError("import_observations requires at least one row.")
        project = LabProject(self.project_path(project_id_or_dir))
        config = project.load_config()
        design_space = project.load_design_space()
        observations = project.load_observations()
        variable_names = design_space.variable_names
        condition_variables = design_space.condition_names
        imported: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        seen_keys = _observation_candidate_keys(observations, design_space, variable_names)
        existing_count = len(observations.rows)
        for raw_index, raw in enumerate(rows, start=1):
            normalized = _normalize_imported_observation(
                raw,
                row_index=raw_index,
                config=config,
                design_space=design_space,
                default_source=source,
                next_observation_index=len(observations.rows) + 1,
            )
            candidate_key = _optimization_key(design_space, normalized, variable_names)
            if candidate_key in seen_keys and not allow_duplicates:
                skipped.append(
                    {
                        "row_index": raw_index,
                        "reason": "duplicate_candidate",
                        "candidate_key": list(candidate_key),
                    }
                )
                continue
            observations.rows.append(normalized)
            imported.append(normalized)
            if str(normalized.get("status") or "").strip().lower() not in {"failed", "skipped"}:
                seen_keys.add(candidate_key)
        if imported:
            project.write_observations(observations)
        return {
            "project_id": config.project_id,
            "imported": imported,
            "skipped": skipped,
            "imported_count": len(imported),
            "skipped_count": len(skipped),
            "source": source,
            "observation_count": len(observations.rows),
            "previous_observation_count": existing_count,
            "completed_observation_count": len(observations.completed_rows(config.objective_name)),
            "best_so_far": observations.best_so_far(
                objective_name=config.objective_name,
                goal=config.goal,
            ),
            "summary": LabProject(project.project_dir).summary(),
        }

    def tell(
        self,
        project_id_or_dir: str | Path,
        *,
        results: list[dict[str, Any]],
        reflect_results: bool = True,
    ) -> dict[str, Any]:
        if not results:
            raise ValueError("tell requires at least one result row.")
        project = LabProject(self.project_path(project_id_or_dir))
        config = project.load_config()
        design_space = project.load_design_space()
        observations = project.load_observations()
        variable_names = design_space.variable_names
        condition_variables = design_space.condition_names
        batches = project.load_batches()
        batches_by_round = {batch.round_id: batch for batch in batches}
        recommendation_index: dict[str, tuple[RecommendationBatch, dict[str, Any]]] = {}
        for batch in batches:
            for recommendation in batch.recommendations:
                recommendation_index[str(recommendation.get("recommendation_id"))] = (
                    batch,
                    recommendation,
                )
        terminal_observations = {
            str(row.get("recommendation_id") or "").strip(): row
            for row in observations.rows
            if str(row.get("recommendation_id") or "").strip()
            and str(row.get("status") or "").strip().lower() in {"completed", "failed", "skipped"}
        }
        runtime: ControllerRuntime | None = None
        knowledge_units: list[dict[str, Any]] = []
        knowledge_meta: dict[str, Any] = {}
        param_space = design_space.param_space()
        agentic_mode = str(config.controller_mode or "agentic").strip().lower() == "agentic"
        if reflect_results and agentic_mode:
            agent_config = _load_agentic_config(
                config.agent_config_path,
                project_dir=project.project_dir,
            )
            decision_engine = _build_decision_engine(agent_config)
            runtime = _build_lab_controller_runtime(
                agent_config=agent_config,
                decision_engine=decision_engine,
                planner_name=config.planner_name,
            )
            evidence_provider = LabEvidenceProvider(EvidenceStore.load(project.evidence_path))
            knowledge_units, knowledge_meta = evidence_provider.retrieve(
                variables=variable_names,
                reaction_scope=config.reaction_scope,
                target_nodes=list(LAB_EVIDENCE_TARGET_NODES),
                max_items=int(agent_config.orchestrator.knowledge_top_k or 5),
            )

        appended: list[dict[str, Any]] = []
        reflections: list[dict[str, Any]] = []
        pending_reflection_ids: list[str] = []
        submitted_ids: set[str] = set()
        for raw in results:
            recommendation_id = str(raw.get("recommendation_id") or "").strip()
            if not recommendation_id:
                raise ValueError("Each lab result must include recommendation_id.")
            if recommendation_id in submitted_ids:
                raise ValueError(f"Duplicate result row for `{recommendation_id}` in one tell request.")
            submitted_ids.add(recommendation_id)
            found = recommendation_index.get(recommendation_id)
            if found is None:
                raise ValueError(f"Unknown recommendation_id `{recommendation_id}`.")
            batch, recommendation = found
            candidate = dict(recommendation.get("candidate") or {})
            status = str(raw.get("status") or "completed").strip().lower()
            if status not in {"completed", "failed", "skipped", "pending"}:
                raise ValueError(f"Unsupported lab result status `{status}`.")
            existing = terminal_observations.get(recommendation_id)
            if existing is not None and status in {"completed", "failed", "skipped"}:
                raise ValueError(
                    f"Recommendation `{recommendation_id}` already has a "
                    f"{existing.get('status', 'recorded')} observation."
                )
            recommendation_status = str(recommendation.get("status") or "pending").strip().lower()
            if recommendation_status in {"completed", "failed", "skipped"} and status in {
                "completed",
                "failed",
                "skipped",
            }:
                raise ValueError(
                    f"Recommendation `{recommendation_id}` is already marked "
                    f"{recommendation_status}."
                )
            # Chain with `or` on the *cleaned strings*, not the raw values -- a genuine
            # 0/0.0 result is falsy in Python and would otherwise be treated as "missing",
            # incorrectly falling through to the next key (or to `float()` on an empty string).
            objective_value = _clean(raw.get(config.objective_name)) or _clean(raw.get("result"))
            if status == "completed":
                try:
                    objective_value = str(float(objective_value))
                except ValueError as exc:
                    raise ValueError(
                        f"Completed result for `{recommendation_id}` requires numeric "
                        f"`{config.objective_name}`."
                    ) from exc
            row = {
                "observation_id": f"obs_{len(observations.rows) + 1:05d}",
                "round_id": batch.round_id,
                "recommendation_id": recommendation_id,
                "source": "recommendation",
                "stage": str(raw.get("stage") or "lab_recommendation"),
                **{name: candidate.get(name, "") for name in condition_variables},
                config.objective_name: objective_value if status == "completed" else "",
                "status": status,
                "failure_reason": str(raw.get("failure_reason") or ""),
                "notes": str(raw.get("notes") or ""),
                "observed_at": str(raw.get("observed_at") or _now_iso()),
                "chemist_override": str(raw.get("chemist_override") or ""),
            }
            observations.rows.append(row)
            appended.append(row)
            recommendation["status"] = status
            recommendation["result"] = objective_value if status == "completed" else ""
            recommendation["failure_reason"] = row["failure_reason"]
            recommendation["notes"] = row["notes"]
            recommendation["observed_at"] = row["observed_at"]
            if status == "completed" and runtime is not None:
                reflection_record = self._reflect_completed_recommendation(
                    runtime=runtime,
                    batch=batch,
                    recommendation=recommendation,
                    recommendation_id=recommendation_id,
                    candidate=candidate,
                    result=float(objective_value),
                    observations=observations,
                    variable_names=variable_names,
                    objective_name=config.objective_name,
                    goal=config.goal,
                    design_space=design_space,
                    param_space=param_space,
                    knowledge_units=knowledge_units,
                    knowledge_meta=knowledge_meta,
                )
                reflections.append(reflection_record)
            elif status == "completed" and agentic_mode:
                pending_reflection_ids.append(recommendation_id)

        project.write_observations(observations)
        for batch in batches_by_round.values():
            project.write_batch(batch, variable_names=condition_variables)
        reflection_status = "none"
        if reflections:
            reflection_status = "completed"
        elif pending_reflection_ids:
            reflection_status = "deferred"
        return {
            "project_id": config.project_id,
            "appended": appended,
            "observation_count": len(observations.rows),
            "completed_observation_count": len(observations.completed_rows(config.objective_name)),
            "best_so_far": observations.best_so_far(
                objective_name=config.objective_name,
                goal=config.goal,
            ),
            "reflections": reflections,
            "reflection_status": reflection_status,
            "reflection_recommendation_ids": pending_reflection_ids,
        }

    def reflect_completed_recommendations(
        self,
        project_id_or_dir: str | Path,
        *,
        recommendation_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Append delayed reflection records for completed lab recommendations."""

        project = LabProject(self.project_path(project_id_or_dir))
        config = project.load_config()
        if str(config.controller_mode or "agentic").strip().lower() != "agentic":
            return {"project_id": config.project_id, "reflections": [], "skipped": "bo_only"}

        design_space = project.load_design_space()
        observations = project.load_observations()
        variable_names = design_space.variable_names
        condition_variables = design_space.condition_names
        completed_by_rec = {
            str(row.get("recommendation_id") or ""): row
            for row in observations.completed_rows(config.objective_name)
            if str(row.get("recommendation_id") or "")
        }
        target_ids = list(recommendation_ids or completed_by_rec.keys())
        if not target_ids:
            return {"project_id": config.project_id, "reflections": []}

        batches = project.load_batches()
        recommendation_index: dict[str, tuple[RecommendationBatch, dict[str, Any]]] = {}
        for batch in batches:
            for recommendation in batch.recommendations:
                recommendation_index[str(recommendation.get("recommendation_id"))] = (
                    batch,
                    recommendation,
                )

        agent_config = _load_agentic_config(
            config.agent_config_path,
            project_dir=project.project_dir,
        )
        decision_engine = _build_decision_engine(agent_config)
        runtime = _build_lab_controller_runtime(
            agent_config=agent_config,
            decision_engine=decision_engine,
            planner_name=config.planner_name,
        )
        evidence_provider = LabEvidenceProvider(EvidenceStore.load(project.evidence_path))
        knowledge_units, knowledge_meta = evidence_provider.retrieve(
            variables=variable_names,
            reaction_scope=config.reaction_scope,
            target_nodes=list(LAB_EVIDENCE_TARGET_NODES),
            max_items=int(agent_config.orchestrator.knowledge_top_k or 5),
        )
        param_space = design_space.param_space()
        reflections: list[dict[str, Any]] = []
        for recommendation_id in target_ids:
            found = recommendation_index.get(str(recommendation_id))
            observed = completed_by_rec.get(str(recommendation_id))
            if found is None or observed is None:
                continue
            batch, recommendation = found
            if recommendation.get("reflection"):
                continue
            try:
                result = float(observed[config.objective_name])
            except (TypeError, ValueError):
                continue
            record = self._reflect_completed_recommendation(
                runtime=runtime,
                batch=batch,
                recommendation=recommendation,
                recommendation_id=str(recommendation_id),
                candidate=dict(recommendation.get("candidate") or {}),
                result=result,
                observations=observations,
                variable_names=variable_names,
                objective_name=config.objective_name,
                goal=config.goal,
                design_space=design_space,
                param_space=param_space,
                knowledge_units=knowledge_units,
                knowledge_meta=knowledge_meta,
            )
            reflections.append(record)

        if reflections:
            for batch in batches:
                project.write_batch(batch, variable_names=condition_variables)
        return {
            "project_id": config.project_id,
            "reflections": reflections,
            "reflection_count": len(reflections),
        }

    def _reflect_completed_recommendation(
        self,
        *,
        runtime: ControllerRuntime,
        batch: RecommendationBatch,
        recommendation: dict[str, Any],
        recommendation_id: str,
        candidate: dict[str, Any],
        result: float,
        observations: ObservationTable,
        variable_names: list[str],
        objective_name: str,
        goal: str,
        design_space: DesignSpace,
        param_space: ParameterSpace,
        knowledge_units: list[dict[str, Any]],
        knowledge_meta: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            reflection = runtime.reflect_after_result(
                history=_history_from_observations(
                    observations=observations,
                    variable_names=variable_names,
                    objective_name=objective_name,
                    goal=goal,
                ),
                candidate=candidate,
                result=result,
                action_package=dict(recommendation.get("action_package") or {}),
                semantic_assessment=dict(recommendation.get("semantic_assessment") or {}),
                search_space=param_space,
                search_space_meta=_lab_search_space_meta(design_space),
                goal=goal,
                objective_name=objective_name,
                iteration=len(observations.rows),
                observations=len(observations.completed_rows(objective_name)),
                total_budget=None,
                knowledge_units=knowledge_units,
                knowledge_meta=knowledge_meta,
            )
            recommendation["reflection"] = reflection
            reflection_record = {
                "run_mode": "lab_tell",
                "event": "tell_reflection",
                "round_id": batch.round_id,
                "recommendation_id": recommendation_id,
                "candidate": candidate,
                "result": result,
                "reflection": reflection,
                "oracle_evaluation_disabled": True,
                "created_at": _now_iso(),
            }
        except Exception as exc:  # noqa: BLE001
            reflection_record = {
                "run_mode": "lab_tell",
                "event": "tell_reflection_error",
                "round_id": batch.round_id,
                "recommendation_id": recommendation_id,
                "candidate": candidate,
                "result": result,
                "error": f"{type(exc).__name__}: {exc}",
                "oracle_evaluation_disabled": True,
                "created_at": _now_iso(),
            }
        batch.trace_records.append(reflection_record)
        return reflection_record


def read_results_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_imported_observation(
    raw: dict[str, Any],
    *,
    row_index: int,
    config: ProjectConfig,
    design_space: DesignSpace,
    default_source: str,
    next_observation_index: int,
) -> dict[str, Any]:
    # Clean each candidate to a string before `or`-chaining -- a genuine 0/0.0 result is
    # falsy in Python, so chaining on the raw values would treat it as "missing" and fall
    # through to the next key (or ultimately fail the float() conversion below).
    objective_value = (
        _clean(_row_value(raw, config.objective_name))
        or _clean(_row_value(raw, "result"))
        or _clean(_row_value(raw, "objective"))
    )
    status = _clean(_row_value(raw, "status")).lower()
    if not status:
        status = "completed" if objective_value else "failed"
    if status not in {"completed", "failed", "skipped", "pending"}:
        raise ValueError(f"Historical row {row_index} has unsupported status `{status}`.")
    if status == "completed":
        try:
            objective_value = str(float(objective_value))
        except ValueError as exc:
            raise ValueError(
                f"Historical row {row_index} requires numeric `{config.objective_name}` "
                "for completed status."
            ) from exc
    else:
        objective_value = ""

    condition_values: dict[str, str] = {}
    static_conditions = design_space.static_conditions()
    active_names = set(design_space.variable_names)
    for name in design_space.condition_names:
        value = _clean(_row_value(raw, name))
        if not value and name in static_conditions:
            value = static_conditions[name]
        if name in active_names:
            condition_values[name] = _canonical_design_value(
                design_space,
                name,
                value,
                strict=True,
                row_index=row_index,
            )
        elif value:
            condition_values[name] = _canonical_design_value(
                design_space,
                name,
                value,
                strict=False,
                row_index=row_index,
            )
        else:
            condition_values[name] = ""

    observation = {
        "observation_id": _clean(_row_value(raw, "observation_id"))
        or f"obs_{next_observation_index:05d}",
        "round_id": _clean(_row_value(raw, "round_id")) or "historical",
        "recommendation_id": _clean(_row_value(raw, "recommendation_id")),
        "source": _clean(_row_value(raw, "source")) or _clean(default_source) or "historical",
        "stage": _clean(_row_value(raw, "stage")) or "historical",
        **condition_values,
        config.objective_name: objective_value,
        "status": status,
        "failure_reason": _clean(_row_value(raw, "failure_reason")),
        "notes": _clean(_row_value(raw, "notes")),
        "observed_at": _clean(_row_value(raw, "observed_at")) or _now_iso(),
        "chemist_override": _clean(_row_value(raw, "chemist_override")),
    }
    consumed = {
        "observation_id",
        "round_id",
        "recommendation_id",
        "source",
        "stage",
        "status",
        "result",
        "objective",
        config.objective_name,
        "failure_reason",
        "notes",
        "observed_at",
        "chemist_override",
        *design_space.condition_names,
    }
    consumed_keys = {_normalize_key(key) for key in consumed}
    for key, value in raw.items():
        if _normalize_key(key) in consumed_keys:
            continue
        if _clean(value):
            observation[str(key)] = value
    return observation


def _row_value(row: dict[str, Any], key: str) -> Any:
    if key in row:
        return row.get(key)
    wanted = _normalize_key(key)
    for candidate_key, value in row.items():
        if _normalize_key(candidate_key) == wanted:
            return value
    return ""


def _canonical_design_value(
    design_space: DesignSpace,
    variable_name: str,
    value: str,
    *,
    strict: bool,
    row_index: int,
) -> str:
    variable = _find_design_variable(design_space, variable_name)
    if variable is None:
        if strict:
            raise ValueError(f"Historical row {row_index} references unknown variable `{variable_name}`.")
        return value
    if variable.kind == "continuous":
        if not value:
            raise ValueError(f"Historical row {row_index} is missing `{variable_name}`.")
        numeric = _parse_numeric_value(value)
        if numeric is None:
            raise ValueError(
                f"Historical row {row_index} has non-numeric value `{value}` "
                f"for continuous variable `{variable_name}`."
            )
        numeric = _round_numeric_for_variable(variable, numeric)
        if variable.low is not None and numeric < float(variable.low):
            raise ValueError(f"Historical row {row_index} has `{variable_name}` below its range.")
        if variable.high is not None and numeric > float(variable.high):
            raise ValueError(f"Historical row {row_index} has `{variable_name}` above its range.")
        return _format_numeric_with_unit(numeric, variable.unit)

    if not value:
        if strict:
            raise ValueError(f"Historical row {row_index} is missing `{variable_name}`.")
        return value
    matched = _match_design_option(variable, value)
    if matched is not None:
        if variable.kind == "discrete_numeric":
            return _display_design_option(variable, matched)
        if variable.optimization_active:
            return str(matched.value)
        return _display_design_option(variable, matched)
    if strict:
        raise ValueError(
            f"Historical row {row_index} value `{value}` is not in design-space "
            f"options for `{variable_name}`."
        )
    return value


def _find_design_variable(design_space: DesignSpace, variable_name: str):
    for variable in design_space.variables:
        if variable.name == variable_name:
            return variable
    return None


def _match_design_option(variable: Any, value: str):
    normalized = _normalize_condition_value(value)
    for option in variable.options:
        candidates = {
            str(option.value),
            _display_design_option(variable, option),
        }
        for candidate in candidates:
            if _normalize_condition_value(candidate) == normalized:
                return option
    return None


def _display_design_option(variable: Any, option: Any) -> str:
    unit = _clean(getattr(option, "unit", "") or getattr(variable, "unit", ""))
    raw = str(getattr(option, "value", ""))
    if unit and raw.strip().lower().endswith(unit.lower()):
        return raw
    return f"{raw} {unit}".strip() if unit else raw


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _normalize_condition_value(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _format_float(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value)


def _parse_numeric_value(value: Any) -> float | None:
    text = _clean(value)
    if not text:
        return None
    token = text.split()[0]
    try:
        return float(token)
    except ValueError:
        return None


def _round_numeric_for_variable(variable: Any, value: float) -> float:
    numeric = float(value)
    low = getattr(variable, "low", None)
    high = getattr(variable, "high", None)
    step = getattr(variable, "step", None)
    if step:
        base = float(low) if low is not None else 0.0
        numeric = base + round((numeric - base) / float(step)) * float(step)
    if low is not None:
        numeric = max(float(low), numeric)
    if high is not None:
        numeric = min(float(high), numeric)
    return numeric


def _format_numeric_with_unit(value: float | None, unit: str = "") -> str:
    if value is None:
        return ""
    raw = _format_float(float(value))
    unit_text = _clean(unit)
    return f"{raw} {unit_text}" if unit_text else raw


def _load_agentic_config(path: str | Path, *, project_dir: Path) -> AgenticBOConfig:
    return load_agentic_bo_config(_resolve_agent_config_path(path, project_dir=project_dir))


def _resolve_agent_config_path(path: str | Path, *, project_dir: Path) -> Path:
    """Resolve a relative agent config path against the project folder, then the repo.

    Project-relative paths (e.g. `agent_bo.yaml`) keep a project working after it is
    moved out of `runs/lab_projects`; repo-relative ones (e.g. `configs/agent_bo.yaml`)
    keep working as before.
    """
    raw = Path(path)
    if raw.is_absolute():
        return raw
    candidates = [Path(project_dir) / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Agent config `{path}` not found; tried: {tried}.")


def _resolve_api_base(config: AgenticBOConfig) -> str | None:
    return (
        config.runtime.api_base
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )


def _build_decision_engine(config: AgenticBOConfig):
    try:
        from chem_agent_bo.agent.decision_engine import DecisionEngine
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Agentic lab mode requires DecisionEngine dependencies such as "
            "langchain and langchain-openai. Use controller_mode=bo_only or "
            "install the agent dependencies."
        ) from exc

    runtime_cfg = config.runtime
    return DecisionEngine(
        model_name=runtime_cfg.model_name,
        temperature=runtime_cfg.temperature,
        api_base=_resolve_api_base(config),
        timeout_sec=runtime_cfg.llm_timeout_sec,
        request_max_retries=runtime_cfg.llm_request_max_retries,
        structured_retry_attempts=runtime_cfg.llm_structured_retry_attempts,
        retry_backoff_sec=runtime_cfg.llm_retry_backoff_sec,
        retry_max_backoff_sec=runtime_cfg.llm_retry_max_backoff_sec,
        retry_jitter_sec=runtime_cfg.llm_retry_jitter_sec,
        fallback_model_name=runtime_cfg.llm_fallback_model_name,
        fallback_attempts=runtime_cfg.llm_fallback_attempts,
        fail_on_nonretryable_error=runtime_cfg.llm_fail_on_nonretryable_error,
        pricing_profile=runtime_cfg.llm_pricing_profile,
        input_cost_per_1m=runtime_cfg.llm_input_cost_per_1m,
        output_cost_per_1m=runtime_cfg.llm_output_cost_per_1m,
        cached_input_cost_per_1m=runtime_cfg.llm_cached_input_cost_per_1m,
        prompt_config=config.prompt,
        use_responses_api=runtime_cfg.use_responses_api,
        reasoning_effort=runtime_cfg.reasoning_effort,
        disable_response_storage=runtime_cfg.disable_response_storage,
    )


def _build_lab_controller_runtime(
    *,
    agent_config: AgenticBOConfig,
    decision_engine: DecisionEngine,
    planner_name: str,
) -> ControllerRuntime:
    orchestrator_cfg = agent_config.orchestrator
    runtime_config = ControllerRuntimeConfig(
        planner_name=planner_name,
        planner_action_policies=dict(orchestrator_cfg.planner_action_policies or {}),
        enable_action_package_v2=bool(orchestrator_cfg.enable_action_package_v2),
        enable_action_package_v06=bool(orchestrator_cfg.enable_action_package_v06),
        verification_mode=str(orchestrator_cfg.verification_mode or "advisory"),
        controller_reflection_input_mode=str(
            orchestrator_cfg.controller_reflection_input_mode or "full"
        ),
        lab_mode=True,
    )
    return ControllerRuntime(
        decision_engine=decision_engine,
        config=runtime_config,
        action_policy=ActionCapabilityPolicy.for_lab(),
    )


def _lab_reaction_context(config: ProjectConfig) -> dict[str, Any]:
    return {
        "dataset": config.project_id,
        "reaction_type": config.reaction_name,
        "objective": config.objective_name,
        "goal": config.goal,
        "backend": "real_lab_ask_tell",
    }


def _lab_search_space_meta(
    design_space: DesignSpace,
    *,
    planner_use_descriptors: bool = False,
    planner_descriptor_eligibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    valid_values = {
        variable.name: variable.option_values()
        for variable in design_space.variables
        if variable.optimization_active and variable.kind != "continuous"
    }
    continuous_ranges = {
        variable.name: {
            "low": variable.low,
            "high": variable.high,
            "step": variable.step,
            "unit": variable.unit,
        }
        for variable in design_space.variables
        if variable.optimization_active and variable.kind == "continuous"
    }
    key_dimensions = list(design_space.variable_names[:3])
    return {
        "dataset_name": "real_lab_project",
        "backend": "real_lab_ask_tell",
        "feature_columns": list(design_space.variable_names),
        "condition_columns": list(design_space.condition_names),
        "valid_values_per_col": valid_values,
        "continuous_ranges": continuous_ranges,
        "fixed_conditions": design_space.fixed_conditions(),
        "controlled_conditions": design_space.controlled_conditions(),
        "static_conditions": design_space.static_conditions(),
        "descriptor_status": {
            "controller_metadata_enabled": True,
            "candidate_descriptor_profiles_enabled": True,
            "planner_descriptor_mode": "enabled" if planner_use_descriptors else "disabled",
            "planner_descriptor_reason": (
                "Atlas descriptor optimization is enabled for this ask."
                if planner_use_descriptors
                else "Descriptors are exposed to TRACE controller and trace; "
                "Atlas descriptor optimization remains disabled unless explicitly enabled and eligible."
            ),
            "planner_descriptor_eligibility": planner_descriptor_eligibility
            or design_space.planner_descriptor_eligibility(
                variable_names=design_space.variable_names,
            ),
        },
        "descriptor_overview": design_space.descriptor_overview(
            variable_names=design_space.variable_names,
        ),
        "stages": design_space.describe().get("stages", []),
        "key_dimensions": key_dimensions,
        "scaffold_dims": key_dimensions,
        "candidate_count": None,
        "oracle_evaluation_disabled": True,
    }


def _planner_fallback_summary(*, requested_planner_name: str, planner_error: str) -> dict[str, Any]:
    if not planner_error:
        return {"used": False}
    return {
        "used": True,
        "requested_planner": requested_planner_name,
        "error": planner_error,
        "warning": (
            f"The {requested_planner_name} planner failed, so this batch contains random "
            "candidates, not Bayesian-optimization recommendations."
        ),
    }


def _planner_warnings(planner_diagnostics: dict[str, Any] | None) -> list[str]:
    return [str(item) for item in (planner_diagnostics or {}).get("warnings") or []]


def _estimated_design_space_size(design_space: DesignSpace) -> int | None:
    size = 1
    for variable in design_space.variables:
        if not variable.optimization_active:
            continue
        if variable.kind == "continuous":
            return None
        size *= max(1, len(variable.options))
    return size


def _history_from_observations(
    *,
    observations: ObservationTable,
    variable_names: list[str],
    objective_name: str,
    goal: str,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    best: float | None = None
    for idx, row in enumerate(observations.completed_rows(objective_name), start=1):
        result = float(row[objective_name])
        improved = (
            best is None
            or (result < best if goal == "minimize" else result > best)
        )
        best = result if best is None else (min(best, result) if goal == "minimize" else max(best, result))
        candidate = {name: row.get(name, "") for name in variable_names}
        history.append(
            {
                "iteration": idx,
                "stage": "lab_observation",
                "trigger_reasons": ["lab_measured_result"],
                "controller_mode": "lab_tell",
                "intervention_type": "lab_tell",
                "subspace_active": False,
                "active_variables": list(variable_names),
                "candidate": candidate,
                "result": result,
                "improved_best": improved,
                "best_result": best,
                "feasibility_action": "accept",
            }
        )
    return history


def _candidate_pool_items(
    *,
    raw_candidates: list[Any],
    param_space: ParameterSpace,
    design_space: DesignSpace,
    variable_names: list[str],
    static_conditions: dict[str, str] | None = None,
    max_items: int,
    proposed_by: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build the planner shortlist; `proposed_by[i]` tags raw candidate i with the
    acquisition(s) that proposed it (chunked_gp), and is shown to the LLM."""
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    static = dict(static_conditions or {})
    tags = list(proposed_by or [])
    for raw_index, raw_candidate in enumerate(raw_candidates):
        optimizer_candidate = _candidate_to_dict(raw_candidate, param_space)
        candidate = _lab_display_candidate(
            design_space,
            {**optimizer_candidate, **static},
            condition_names=design_space.condition_names,
        )
        key = _optimization_key(design_space, candidate, variable_names)
        if key in seen:
            continue
        seen.add(key)
        index = len(items)
        items.append(
            {
                "candidate_index": index,
                "bo_rank": index + 1,
                "main_pool_rank": index + 1,
                "is_main_bo_top1": index == 0,
                "shortlist_source": "planner_shortlist",
                "pool_source": "main_pool",
                "candidate": candidate,
            }
        )
        if raw_index < len(tags) and tags[raw_index]:
            items[-1]["proposed_by"] = str(tags[raw_index])
        if len(items) >= max(1, int(max_items)):
            break
    return items


def _attach_descriptor_context(
    *,
    candidate_pool: list[dict[str, Any]],
    design_space: DesignSpace,
    variable_names: list[str],
) -> list[dict[str, Any]]:
    if not candidate_pool:
        return candidate_pool
    anchor = dict(candidate_pool[0].get("candidate") or {})
    enriched: list[dict[str, Any]] = []
    for item in candidate_pool:
        candidate = dict(item.get("candidate") or {})
        descriptor_profile = design_space.candidate_descriptor_profile(
            candidate,
            variable_names=variable_names,
        )
        descriptor_contrast = design_space.candidate_descriptor_contrast(
            candidate,
            anchor,
            variable_names=variable_names,
        )
        enriched.append(
            {
                **item,
                "descriptor_profile": descriptor_profile,
                "descriptor_contrast_to_anchor": descriptor_contrast,
                "descriptor_signal": _descriptor_signal_summary(
                    descriptor_profile,
                    descriptor_contrast,
                ),
            }
        )
    return enriched


def _recommendations_from_decision(
    *,
    round_id: str,
    decision,
    planner_name: str,
    planner_diagnostics: dict[str, Any],
    planner_error: str,
    created_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_refs = list(
        (decision.evidence_trace or {}).get("retrieved_card_ids")
        or [
            item.get("id")
            for item in (decision.evidence_trace or {}).get("retrieved_units", [])
            if item.get("id")
        ]
    )
    recommendations: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    for rank, candidate in enumerate(decision.selected_candidates, start=1):
        recommendation_id = f"{round_id}_rec_{rank:03d}"
        trace_record = (
            decision.trace_records[rank - 1]
            if rank - 1 < len(decision.trace_records)
            else {}
        )
        semantic = (
            decision.semantic_assessments[rank - 1]
            if rank - 1 < len(decision.semantic_assessments)
            else {}
        )
        verification = (
            decision.verification_passes[rank - 1]
            if rank - 1 < len(decision.verification_passes)
            else None
        )
        rationale = _build_agentic_rationale(
            controller_plan=decision.controller_plan,
            action_package=decision.action_package,
            semantic_assessment=semantic,
            evidence_refs=evidence_refs,
            planner_error=planner_error,
            batch_role=str(trace_record.get("batch_role") or ""),
            batch_role_reason=str(trace_record.get("batch_role_reason") or ""),
            batch_contract=dict(decision.batch_contract or {}),
            batch_slot=dict(trace_record.get("batch_slot") or {}),
            descriptor_contrast=dict(trace_record.get("descriptor_contrast_to_anchor") or {}),
            descriptor_signal=dict(trace_record.get("descriptor_signal") or {}),
        )
        recommendation = {
            "round_id": round_id,
            "recommendation_id": recommendation_id,
            "rank": rank,
            "status": "pending",
            "candidate": candidate,
            "rationale": rationale,
            "planner_name": planner_name,
            "planner_error": planner_error,
            "planner_warnings": _planner_warnings(planner_diagnostics),
            "proposed_by": str(trace_record.get("proposed_by") or ""),
            "batch_role": trace_record.get("batch_role", ""),
            "batch_role_reason": trace_record.get("batch_role_reason", ""),
            "batch_slot": trace_record.get("batch_slot", {}),
            "batch_contract": decision.batch_contract or {},
            "batch_validation_report": decision.batch_validation_report or {},
            "selection_trace": trace_record.get("selection_trace", {}),
            "controller_action": decision.action_package.get(
                "executed_execution_action",
                decision.controller_plan.get("executed_execution_action", "direct_bo_pick"),
            ),
            "requested_controller_action": decision.action_package.get(
                "requested_execution_action",
                decision.controller_plan.get("requested_execution_action"),
            ),
            "action_package": decision.action_package,
            "controller_plan": decision.controller_plan,
            "semantic_assessment": semantic,
            "verification_pass": verification,
            "descriptor_profile": trace_record.get("descriptor_profile", {}),
            "descriptor_contrast_to_anchor": trace_record.get(
                "descriptor_contrast_to_anchor",
                {},
            ),
            "descriptor_signal": trace_record.get("descriptor_signal", {}),
            "evidence_refs": evidence_refs,
            "created_at": created_at,
            "result": "",
            "failure_reason": "",
            "notes": "",
        }
        trace = {
            **trace_record,
            "run_mode": "lab_ask",
            "round_id": round_id,
            "recommendation_id": recommendation_id,
            "planner_name": planner_name,
            "planner_diagnostics": planner_diagnostics,
            "planner_error": planner_error,
            "oracle_evaluation_disabled": True,
            "evidence_refs": evidence_refs,
            "skill_trace": decision.skill_trace,
            "diagnosis": decision.diagnosis,
            "hypothesis_action": decision.hypothesis_action,
            "coverage_insight": decision.coverage_insight,
            "action_capability": decision.action_capability,
            "rerank_action": decision.rerank_action,
            "descriptor_profile": trace_record.get("descriptor_profile", {}),
            "descriptor_contrast_to_anchor": trace_record.get(
                "descriptor_contrast_to_anchor",
                {},
            ),
            "descriptor_signal": trace_record.get("descriptor_signal", {}),
            "batch_contract": decision.batch_contract or {},
            "batch_validation_report": decision.batch_validation_report or {},
            "created_at": created_at,
        }
        recommendations.append(recommendation)
        trace_records.append(trace)
    return recommendations, trace_records


def _bo_only_recommendations(
    *,
    round_id: str,
    candidate_pool: list[dict[str, Any]],
    batch_size: int,
    planner_name: str,
    planner_diagnostics: dict[str, Any],
    planner_error: str,
    evidence_cards: list[Any],
    created_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recommendations: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    for item in candidate_pool[: max(1, int(batch_size))]:
        rank = len(recommendations) + 1
        recommendation_id = f"{round_id}_rec_{rank:03d}"
        candidate = dict(item["candidate"])
        descriptor_profile = dict(item.get("descriptor_profile") or {})
        descriptor_contrast = dict(item.get("descriptor_contrast_to_anchor") or {})
        descriptor_signal = dict(item.get("descriptor_signal") or {})
        evidence_refs = [card.card_id for card in evidence_cards if card.card_id]
        rationale = _build_lab_rationale(
            planner_name=planner_name,
            evidence_cards=evidence_cards,
            planner_error=planner_error,
        )
        trace = {
            "run_mode": "lab_ask",
            "round_id": round_id,
            "recommendation_id": recommendation_id,
            "rank": rank,
            "candidate": candidate,
            "controller_action": "keep_planner_batch",
            "controller_boundary": "recommend_only_wait_for_measured_result",
            "oracle_evaluation_disabled": True,
            "evidence_refs": evidence_refs,
            "planner_diagnostics": planner_diagnostics,
            "planner_error": planner_error,
            "proposed_by": str(item.get("proposed_by") or ""),
            "descriptor_profile": descriptor_profile,
            "descriptor_contrast_to_anchor": descriptor_contrast,
            "descriptor_signal": descriptor_signal,
            "created_at": created_at,
        }
        recommendation = {
            "round_id": round_id,
            "recommendation_id": recommendation_id,
            "rank": rank,
            "status": "pending",
            "candidate": candidate,
            "rationale": rationale,
            "planner_name": planner_name,
            "planner_error": planner_error,
            "planner_warnings": _planner_warnings(planner_diagnostics),
            "proposed_by": str(item.get("proposed_by") or ""),
            "controller_action": "keep_planner_batch",
            "descriptor_profile": descriptor_profile,
            "descriptor_contrast_to_anchor": descriptor_contrast,
            "descriptor_signal": descriptor_signal,
            "evidence_refs": evidence_refs,
            "created_at": created_at,
            "result": "",
            "failure_reason": "",
            "notes": "",
        }
        recommendations.append(recommendation)
        trace_records.append(trace)
    return recommendations, trace_records


def _build_agentic_rationale(
    *,
    controller_plan: dict[str, Any],
    action_package: dict[str, Any],
    semantic_assessment: dict[str, Any],
    evidence_refs: list[str],
    planner_error: str,
    batch_role: str = "",
    batch_role_reason: str = "",
    batch_contract: dict[str, Any] | None = None,
    batch_slot: dict[str, Any] | None = None,
    descriptor_contrast: dict[str, Any] | None = None,
    descriptor_signal: dict[str, Any] | None = None,
) -> str:
    action = (
        action_package.get("executed_execution_action")
        or controller_plan.get("executed_execution_action")
        or "direct_bo_pick"
    )
    role_label = batch_role.replace("_", " ") if batch_role else "batch member"
    role_article = "an" if role_label[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    contract = dict(batch_contract or {})
    slot = dict(batch_slot or {})
    pieces = [
        f"TRACE selected this condition as {role_article} {role_label} using {action} under lab ask/tell mode."
    ]

    def _append_unique(text: str, *, prefix: str = "") -> None:
        text = str(text or "").strip()
        if not text:
            return
        sentence = prefix + text.rstrip(".") + "."
        normalized = " ".join(sentence.lower().split())
        existing = {" ".join(piece.lower().split()) for piece in pieces}
        if normalized not in existing:
            pieces.append(sentence)

    batch_strategy = str(contract.get("batch_strategy") or "").strip()
    _append_unique(batch_strategy, prefix="Batch strategy: ")
    slot_purpose = str(slot.get("batch_slot_purpose") or "").strip()
    _append_unique(slot_purpose)
    _append_unique(batch_role_reason)
    slot_rationale = str(slot.get("batch_slot_rationale") or "").strip()
    _append_unique(slot_rationale)
    slot_risk = str(slot.get("batch_slot_risk_note") or "").strip()
    _append_unique(slot_risk, prefix="Risk note: ")
    descriptor_sentence = _descriptor_contrast_sentence(
        descriptor_contrast or {},
        descriptor_signal or {},
    )
    if descriptor_sentence:
        pieces.append(descriptor_sentence)
    if not slot_rationale:
        reasoning = str(controller_plan.get("reasoning") or "").strip()
        if reasoning:
            pieces.append(reasoning)
    soft_comment = str(semantic_assessment.get("soft_comment") or "").strip()
    if soft_comment:
        pieces.append(soft_comment)
    if evidence_refs:
        pieces.append("Scoped evidence refs: " + ", ".join(evidence_refs) + ".")
    if planner_error:
        pieces.append(f"Planner fallback was used after: {planner_error}.")
    return " ".join(pieces)


def _descriptor_signal_summary(
    descriptor_profile: dict[str, Any],
    descriptor_contrast: dict[str, Any],
) -> dict[str, Any]:
    contrasted_variables = [
        name
        for name, payload in descriptor_contrast.items()
        if isinstance(payload, dict) and payload.get("top_descriptor_deltas")
    ]
    profiled_variables = [
        name
        for name, payload in descriptor_profile.items()
        if isinstance(payload, dict) and payload.get("numeric_descriptor_count", 0)
    ]
    return {
        "profiled_variables": profiled_variables,
        "contrasted_variables": contrasted_variables,
        "has_descriptor_contrast": bool(contrasted_variables),
    }


def _descriptor_contrast_sentence(
    descriptor_contrast: dict[str, Any],
    descriptor_signal: dict[str, Any],
) -> str:
    if not descriptor_signal.get("has_descriptor_contrast"):
        return ""
    phrases: list[str] = []
    for variable, payload in descriptor_contrast.items():
        if not isinstance(payload, dict):
            continue
        deltas = payload.get("top_descriptor_deltas") or {}
        if not deltas:
            continue
        names = list(deltas.keys())[:2]
        from_value = payload.get("from", "")
        to_value = payload.get("to", "")
        if names:
            phrases.append(
                f"{variable} {from_value} -> {to_value} shifts {', '.join(names)}"
            )
        if len(phrases) >= 2:
            break
    if not phrases:
        return ""
    return "Descriptor contrast relative to the planner anchor: " + "; ".join(phrases) + "."


def _campaign_from_observations(
    *,
    observations: ObservationTable,
    design_space: DesignSpace,
    param_space: ParameterSpace,
    value_space: ParameterSpace,
    variable_names: list[str],
    objective_name: str,
) -> Campaign:
    campaign = Campaign()
    campaign.set_param_space(param_space)
    campaign.set_value_space(value_space)
    for row in observations.completed_rows(objective_name):
        candidate = _optimizer_candidate_values(
            design_space,
            row,
            variable_names=variable_names,
        )
        value = float(row[objective_name])
        vector = ParameterVector().from_dict(candidate, param_space=param_space)
        campaign.add_observation(vector, value)
    return campaign


def _value_space(objective_name: str) -> ParameterSpace:
    space = ParameterSpace()
    space.add(ParameterContinuous(name=objective_name, low=0.0, high=100.0))
    return space


def _candidate_to_dict(sample: Any, param_space: ParameterSpace) -> dict[str, Any]:
    if hasattr(sample, "to_dict"):
        try:
            data = sample.to_dict()
            return {param.name: _json_safe(data[param.name]) for param in param_space}
        except Exception:  # noqa: BLE001
            pass
    if isinstance(sample, dict):
        return {param.name: _json_safe(sample.get(param.name)) for param in param_space}
    return {param.name: _json_safe(getattr(sample, param.name)) for param in param_space}


def _lab_display_candidate(
    design_space: DesignSpace,
    candidate: dict[str, Any],
    *,
    condition_names: list[str],
) -> dict[str, str]:
    display: dict[str, str] = {}
    for name in condition_names:
        variable = _find_design_variable(design_space, name)
        value = candidate.get(name, "")
        if variable is None:
            display[name] = _clean(value)
            continue
        if variable.kind == "continuous":
            numeric = _parse_numeric_value(value)
            if numeric is None:
                display[name] = _clean(value)
                continue
            rounded = _round_numeric_for_variable(variable, numeric)
            display[name] = _format_numeric_with_unit(rounded, variable.unit)
            continue
        if variable.kind == "discrete_numeric":
            option = _match_design_option(variable, _clean(value))
            if option is not None:
                display[name] = _display_design_option(variable, option)
            else:
                numeric = _parse_numeric_value(value)
                display[name] = (
                    _format_numeric_with_unit(numeric, variable.unit)
                    if numeric is not None
                    else _clean(value)
                )
            continue
        option = _match_design_option(variable, _clean(value))
        if option is not None:
            display[name] = str(option.value)
        else:
            display[name] = _clean(value)
    return display


def _optimizer_candidate_values(
    design_space: DesignSpace,
    candidate: dict[str, Any],
    *,
    variable_names: list[str],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in variable_names:
        variable = _find_design_variable(design_space, name)
        raw_value = candidate.get(name, "")
        if variable is None:
            values[name] = raw_value
            continue
        if variable.kind == "continuous":
            numeric = _parse_numeric_value(raw_value)
            if numeric is None:
                raise ValueError(f"Continuous variable `{name}` requires a numeric value.")
            values[name] = _round_numeric_for_variable(variable, numeric)
            continue
        option = _match_design_option(variable, _clean(raw_value))
        values[name] = str(option.value) if option is not None else _clean(raw_value)
    return values


def _optimization_key(
    design_space: DesignSpace,
    candidate: dict[str, Any],
    variable_names: list[str],
) -> tuple[str, ...]:
    values = _optimizer_candidate_values(
        design_space,
        candidate,
        variable_names=variable_names,
    )
    return tuple(_normalize_condition_value(values.get(name, "")) for name in variable_names)


def _observation_candidate_keys(
    observations: ObservationTable,
    design_space: DesignSpace,
    variable_names: list[str],
) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    for row in observations.rows:
        if str(row.get("status", "")).strip().lower() in {"failed", "skipped"}:
            continue
        try:
            keys.add(_optimization_key(design_space, row, variable_names))
        except ValueError:
            continue
    return keys


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return value


def _pending_recommendation_keys(
    project: LabProject,
    design_space: DesignSpace,
    variable_names: list[str],
) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    for batch in project.load_batches():
        for recommendation in batch.recommendations:
            if str(recommendation.get("status") or "pending").lower() != "pending":
                continue
            candidate = recommendation.get("candidate") or {}
            try:
                keys.add(_optimization_key(design_space, candidate, variable_names))
            except ValueError:
                continue
    return keys


def _exclude_candidate_keys(
    variable_names: list[str],
    banned_keys: set[tuple[str, ...]],
    *,
    design_space: DesignSpace,
):
    banned = set(banned_keys)

    def _constraint(values: Any) -> bool:
        if hasattr(values, "to_dict"):
            data = values.to_dict()
        elif isinstance(values, dict):
            data = values
        elif hasattr(values, "tolist"):
            raw_values = values.tolist()
            if raw_values and isinstance(raw_values[0], list):
                raw_values = raw_values[0]
            data = {
                name: raw_values[index] if index < len(raw_values) else ""
                for index, name in enumerate(variable_names)
            }
        elif isinstance(values, (list, tuple)):
            data = {
                name: values[index] if index < len(values) else ""
                for index, name in enumerate(variable_names)
            }
        else:
            data = {name: getattr(values, name) for name in variable_names}
        try:
            key = _optimization_key(design_space, data, variable_names)
        except ValueError:
            return False
        return key not in banned

    return ExclusionConstraint(
        variable_names=variable_names,
        excluded_keys=banned,
        normalize_value=_normalize_condition_value,
        check=_constraint,
    )


def _build_lab_rationale(
    *,
    planner_name: str,
    evidence_cards: list[Any],
    planner_error: str,
) -> str:
    base = (
        f"{planner_name} proposed this condition under lab ask/tell mode; "
        "TRACE records the recommendation and waits for a measured result."
    )
    if planner_error:
        base += f" Planner fallback was used after: {planner_error}."
    if evidence_cards:
        refs = ", ".join(card.card_id for card in evidence_cards if card.card_id)
        if refs:
            base += f" Scoped evidence available: {refs}."
    return base


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_for_path() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _unique_backup_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(2, 1000):
        candidate = base.parent / f"{base.name}_{index}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique backup directory near {base}.")


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


class _LabPlannerEnv:
    def __init__(self, *, goal: str) -> None:
        self.goal = goal
