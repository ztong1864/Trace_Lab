"""Optional FastAPI app for local real-lab TRACE projects."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from chem_agent_bo.lab.design_space import DescriptorTableError
from chem_agent_bo.lab.evidence import (
    BLOCKED_ALLOWED_USES,
    BLOCKED_LEAKAGE_RISKS,
    EvidenceCard,
    EvidenceImportError,
    EvidenceStore,
)
from chem_agent_bo.lab.project import LabProject, ProjectConfig
from chem_agent_bo.lab.service import LabBOService, ProjectBusyError, ProjectCheckError


def create_app(projects_root: str | Path):
    """Create a FastAPI app.

    FastAPI is an optional lab-mode dependency, so imports live inside this
    factory and do not affect benchmark runners.
    """

    global BackgroundTasks, FastAPI, HTTPException, FileResponse, StaticFiles
    try:
        from fastapi import BackgroundTasks, FastAPI, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI is required for the lab web API. Install fastapi and uvicorn, "
            "or use experiments/run_lab_bo.py for CLI ask/tell."
        ) from exc

    root = Path(projects_root)
    root.mkdir(parents=True, exist_ok=True)
    # URLs name projects by handle only; raw folder paths are never accepted here.
    service = LabBOService(projects_root=root, allow_paths=False)
    web_dir = Path(__file__).resolve().parent / "web"
    app = FastAPI(title="TRACE Lab API", version="0.1")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "projects_root": str(root), "registry_path": str(service.registry.path)}

    @app.get("/api/projects")
    def list_projects() -> dict[str, Any]:
        return {"projects": service.list_projects()}

    @app.post("/api/projects")
    def create_project(payload: dict[str, Any]) -> dict[str, Any]:
        """Create a project from uploaded parts, all or nothing.

        Design space as `design_records` or `design_csv`; optional `descriptor_tables`
        ([{variable, value_column, csv or rows, aliases?, replace?}]), `observations` or
        `observations_csv`, and `evidence_cards`, `evidence_jsonl` or `evidence_csv`.
        """
        try:
            project_id = str(payload.get("project_id") or payload.get("config", {}).get("project_id") or "")
            config_payload = dict(payload.get("config") or {})
            config_payload["project_id"] = project_id
            return service.upload_project(
                project_id,
                config=ProjectConfig(**config_payload),
                design_records=payload.get("design_records"),
                design_csv=payload.get("design_csv"),
                descriptor_tables=list(payload.get("descriptor_tables") or []),
                observations=payload.get("observations"),
                observations_csv=payload.get("observations_csv"),
                observation_source=str(payload.get("observation_source") or "historical"),
                allow_duplicate_observations=bool(payload.get("allow_duplicate_observations", False)),
                evidence_cards=payload.get("evidence_cards"),
                evidence_jsonl=payload.get("evidence_jsonl"),
                evidence_csv=payload.get("evidence_csv"),
                overwrite=bool(payload.get("overwrite", False)),
            )
        except DescriptorTableError as exc:
            raise HTTPException(status_code=400, detail={"message": str(exc), "report": exc.report}) from exc
        except ProjectCheckError as exc:
            raise HTTPException(status_code=400, detail={"message": str(exc), "check": exc.report}) from exc
        except EvidenceImportError as exc:
            raise HTTPException(status_code=400, detail={"message": str(exc), "problems": exc.problems}) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/projects/check")
    def check_folder(payload: dict[str, Any]) -> dict[str, Any]:
        """Check a project folder on the server's disk before registering it."""
        try:
            return service.check_folder(payload.get("path"))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/projects/register")
    def register_project(payload: dict[str, Any]) -> dict[str, Any]:
        """Use an existing project folder on the server's disk in place, by handle."""
        try:
            return service.register_project(
                payload.get("path"),
                handle=payload.get("handle") or None,
                drop_config_keys=list(payload.get("drop_config_keys") or []),
                set_config=payload.get("set_config"),
            )
        except ProjectCheckError as exc:
            raise HTTPException(status_code=400, detail={"message": str(exc), "check": exc.report}) from exc
        except ProjectBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/projects/{project_id}/descriptors")
    def add_descriptor_table(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Attach a per-option descriptor table (CSV text or rows) to one design variable."""
        try:
            return service.add_descriptor_table(
                project_id,
                variable=str(payload.get("variable") or ""),
                value_column=str(payload.get("value_column") or ""),
                rows=payload.get("rows"),
                csv_text=payload.get("csv"),
                aliases=dict(payload.get("aliases") or {}),
                replace=bool(payload.get("replace", False)),
            )
        except DescriptorTableError as exc:
            raise HTTPException(status_code=400, detail={"message": str(exc), "report": exc.report}) from exc
        except ProjectBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/projects/{project_id}/evidence/import")
    def import_evidence(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Add or replace evidence cards (cards, JSONL or CSV), validated strictly."""
        try:
            return service.import_evidence(
                project_id,
                cards=payload.get("cards"),
                jsonl=payload.get("jsonl"),
                csv_text=payload.get("csv"),
                mode=str(payload.get("mode") or "append"),
            )
        except EvidenceImportError as exc:
            raise HTTPException(status_code=400, detail={"message": str(exc), "problems": exc.problems}) from exc
        except ProjectBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}/evidence/preview")
    def evidence_preview(project_id: str) -> dict[str, Any]:
        """Which cards the controller would be shown, and why each of the others never is."""
        try:
            return service.evidence_preview(project_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/projects/{project_id}/registration")
    def unregister_project(project_id: str) -> dict[str, Any]:
        """Forget a registered project; its files are not touched."""
        try:
            return service.unregister_project(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0] if exc.args else exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}/check")
    def check_project(project_id: str) -> dict[str, Any]:
        try:
            return service.check_project(project_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}")
    def project_summary(project_id: str) -> dict[str, Any]:
        try:
            return service.project_summary(project_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/projects/{project_id}/ask")
    def ask(project_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            payload = payload or {}
            return service.ask(
                project_id,
                batch_size=payload.get("batch_size"),
                planner_name=payload.get("planner_name"),
                controller_mode=payload.get("controller_mode"),
                agent_config_path=payload.get("agent_config_path") or payload.get("agent_config"),
                planner_use_descriptors=payload.get("planner_use_descriptors"),
                # {api_key, base_url?, model?}: the caller's own LLM key for the agentic steps, this request only.
                llm=payload.get("llm"),
            )
        except ProjectBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.patch("/api/projects/{project_id}/config")
    def update_config(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Change editable project.yaml settings; observations and batches are untouched."""
        try:
            return service.update_project_config(project_id, dict(payload or {}))
        except ProjectBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/projects/{project_id}/reset")
    def reset_project(project_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            payload = payload or {}
            return service.reset_project(
                project_id,
                backup=bool(payload.get("backup", True)),
                keep_historical=bool(payload.get("keep_historical", False)),
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/projects/{project_id}/tell")
    def tell(
        project_id: str,
        payload: dict[str, Any],
        background_tasks: BackgroundTasks,
    ) -> dict[str, Any]:
        try:
            defer_reflection = bool(payload.get("defer_reflection", False))
            llm = payload.get("llm")
            response = service.tell(
                project_id,
                results=list(payload.get("results") or []),
                reflect_results=not defer_reflection,
                llm=llm,
            )
            reflection_ids = list(response.get("reflection_recommendation_ids") or [])
            if defer_reflection and reflection_ids:
                background_tasks.add_task(
                    service.reflect_completed_recommendations,
                    project_id,
                    recommendation_ids=reflection_ids,
                    llm=llm,
                )
            return response
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}/recommendations")
    def recommendations(project_id: str) -> dict[str, Any]:
        try:
            project = LabProject(service.project_path(project_id))
            return {"batches": [batch.to_dict() for batch in project.load_batches()]}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}/evidence")
    def evidence(project_id: str) -> dict[str, Any]:
        try:
            project = LabProject(service.project_path(project_id))
            store = EvidenceStore.load(project.evidence_path)
            cards = []
            for card in store.cards:
                if not _is_ui_visible_evidence(card):
                    continue
                payload = asdict(card)
                cards.append(
                    {
                        key: payload.get(key)
                        for key in (
                            "card_id",
                            "source",
                            "summary",
                            "reaction_scope",
                            "variable_scope",
                            "target_nodes",
                            "mapping_status",
                            "confidence",
                            "allowed_use",
                            "source_type",
                            "doi",
                            "supporting_excerpt",
                            "transferability_note",
                            "leakage_risk",
                            "notes",
                        )
                    }
                )
            return {
                "evidence_path": str(project.evidence_path),
                "count": len(cards),
                "cards": cards,
            }
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}/observations")
    def observations(project_id: str) -> dict[str, Any]:
        try:
            project = LabProject(service.project_path(project_id))
            rows = project.load_observations().rows
            return {"observations": rows}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/projects/{project_id}/observations/import")
    def import_observations(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return service.import_observations(
                project_id,
                rows=list(payload.get("rows") or payload.get("observations") or []),
                source=str(payload.get("source") or "historical"),
                allow_duplicates=bool(payload.get("allow_duplicates", False)),
            )
        except ProjectBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}/trace/{round_id}")
    def trace(project_id: str, round_id: str) -> dict[str, Any]:
        try:
            project = LabProject(service.project_path(project_id))
            trace_path = project.trace_path(round_id)
            records = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            return {"round_id": round_id, "trace": records}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/")
        def index():
            return FileResponse(web_dir / "index.html")

    return app


def _is_ui_visible_evidence(card: EvidenceCard) -> bool:
    if card.mapping_status == "out_of_scope":
        return False
    if str(card.allowed_use or "").strip().lower() in BLOCKED_ALLOWED_USES:
        return False
    if _norm_policy_value(card.leakage_risk) in BLOCKED_LEAKAGE_RISKS:
        return False
    return True


def _norm_policy_value(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
