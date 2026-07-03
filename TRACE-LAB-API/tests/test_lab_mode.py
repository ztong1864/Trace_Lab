from pathlib import Path
import tempfile
import unittest

from chem_agent_bo.lab.api import create_app
from chem_agent_bo.lab.design_space import DesignSpace
from chem_agent_bo.lab.evidence import EvidenceCard, EvidenceStore
from chem_agent_bo.lab.project import LabProject, ProjectConfig
from chem_agent_bo.lab.service import LabBOService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PublicLabModeSmokeTests(unittest.TestCase):
    def test_demo_project_loads(self):
        project = LabProject(PROJECT_ROOT / "runs/lab_projects/oxidative_esterification_demo")
        summary = project.summary()
        self.assertEqual(summary["project"]["project_id"], "oxidative_esterification_demo")
        self.assertIn("Additive", summary["design_space"]["active_variables"])
        self.assertTrue(project.evidence_path.exists())

    def test_lab_api_exposes_scoped_evidence(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:  # pragma: no cover
            self.skipTest("FastAPI test client is not installed.")

        client = TestClient(create_app(PROJECT_ROOT / "runs/lab_projects"))
        response = client.get("/api/projects/oxidative_esterification_demo/evidence")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(payload["count"], 1)
        self.assertNotIn("source_path", {key for key, value in payload["cards"][0].items() if value})

    def test_ask_tell_roundtrip_without_oracle_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = LabBOService(projects_root=tmp)
            design = DesignSpace.from_long_records(
                [
                    {"variable": "Catalyst", "type": "categorical", "value": "cat_a"},
                    {"variable": "Catalyst", "type": "categorical", "value": "cat_b"},
                    {"variable": "Solvent", "type": "categorical", "value": "dce"},
                    {"variable": "Solvent", "type": "categorical", "value": "meoh"},
                    {"variable": "Temperature", "type": "discrete_numeric", "value": "25", "unit": "C"},
                    {"variable": "Temperature", "type": "discrete_numeric", "value": "50", "unit": "C"},
                ]
            )
            project = service.create_project(
                "smoke",
                config=ProjectConfig(
                    project_id="smoke",
                    objective_name="yield",
                    planner_name="random",
                    controller_mode="bo_only",
                    batch_size=2,
                ),
                design_space=design,
            )
            EvidenceStore(
                [
                    EvidenceCard(
                        card_id="ev_smoke",
                        source="example note",
                        summary="Scoped advisory context for smoke testing.",
                        variable_scope=["Catalyst"],
                        target_nodes=["lab_batch_composition"],
                        mapping_status="background",
                    )
                ]
            ).write_jsonl(project.evidence_path)

            batch = service.ask("smoke", batch_size=2, planner_name="random", controller_mode="bo_only")
            self.assertEqual(len(batch["recommendations"]), 2)
            results = [
                {
                    "recommendation_id": batch["recommendations"][0]["recommendation_id"],
                    "yield": "10.0",
                    "status": "completed",
                }
            ]
            told = service.tell("smoke", results=results, reflect_results=False)
            self.assertEqual(len(told["appended"]), 1)
            trace_text = project.trace_path(batch["round_id"]).read_text(encoding="utf-8")
            self.assertIn("oracle_evaluation_disabled", trace_text)
            self.assertNotIn("candidate_true_value", trace_text)


if __name__ == "__main__":
    unittest.main()

