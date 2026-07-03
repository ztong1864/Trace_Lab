"""Optional FastAPI app for local real-lab TRACE projects."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from chem_agent_bo.lab.design_space import DesignSpace
from chem_agent_bo.lab.evidence import (
    BLOCKED_ALLOWED_USES,
    BLOCKED_LEAKAGE_RISKS,
    EvidenceCard,
    EvidenceStore,
)
from chem_agent_bo.lab.project import LabProject, ProjectConfig
from chem_agent_bo.lab.service import LabBOService


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
    service = LabBOService(projects_root=root)
    web_dir = Path(__file__).resolve().parent / "web"
    app = FastAPI(title="TRACE Lab API", version="0.1")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "projects_root": str(root)}

    @app.get("/api/projects")
    def list_projects() -> dict[str, Any]:
        projects = []
        for path in sorted(root.iterdir()):
            if path.is_dir() and (path / "project.yaml").exists():
                try:
                    projects.append(LabProject(path).summary())
                except Exception as exc:  # noqa: BLE001
                    projects.append({"project_dir": str(path), "error": str(exc)})
        return {"projects": projects}

    @app.post("/api/projects")
    def create_project(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            project_id = str(payload.get("project_id") or payload.get("config", {}).get("project_id") or "")
            config_payload = dict(payload.get("config") or {})
            config_payload["project_id"] = project_id
            config = ProjectConfig(**config_payload)
            design_records = list(payload.get("design_records") or [])
            design_space = DesignSpace.from_long_records(design_records)
            project = service.create_project(
                project_id,
                config=config,
                design_space=design_space,
                overwrite=bool(payload.get("overwrite", False)),
            )
            return project.summary()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}")
    def project_summary(project_id: str) -> dict[str, Any]:
        try:
            return LabProject(service.project_path(project_id)).summary()
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
            )
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
            response = service.tell(
                project_id,
                results=list(payload.get("results") or []),
                reflect_results=not defer_reflection,
            )
            reflection_ids = list(response.get("reflection_recommendation_ids") or [])
            if defer_reflection and reflection_ids:
                background_tasks.add_task(
                    service.reflect_completed_recommendations,
                    project_id,
                    recommendation_ids=reflection_ids,
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
