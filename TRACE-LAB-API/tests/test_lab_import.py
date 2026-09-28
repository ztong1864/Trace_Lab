"""Importing outside projects: handles, registry, checks, registration, uploads."""

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from chem_agent_bo.lab.api import create_app
from chem_agent_bo.lab.registry import ProjectRegistry, default_handle, validate_handle
from chem_agent_bo.lab.service import LabBOService

from test_lab_mode import _create_small_project


class HandleTests(unittest.TestCase):
    def test_handles_can_never_be_paths(self):
        for bad in ("", "..", ".", "a/b", "a\\b", "C:", "../x", "a b", "_hidden", "x" * 129):
            with self.subTest(handle=bad), self.assertRaises(ValueError):
                validate_handle(bad)
        self.assertEqual(validate_handle("photoredox_ei-2"), "photoredox_ei-2")
        self.assertEqual(validate_handle(" padded "), "padded")

    def test_default_handle_comes_from_the_folder_name(self):
        self.assertEqual(default_handle(Path("x") / "my folder.v2"), "my_folder_v2")
        self.assertEqual(default_handle("03_agentic_without_cr_salts"), "03_agentic_without_cr_salts")


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.root = base / "root"
        self.outside = base / "outside"
        self.root.mkdir()
        self.outside.mkdir()

    def _outside_project(self, name: str) -> Path:
        service = LabBOService()
        _create_small_project(service, str(self.outside / name))
        return self.outside / name

    def test_add_resolve_and_remove(self):
        registry = ProjectRegistry(self.root)
        folder = self._outside_project("lactonization")
        registry.add("lacto", folder)
        self.assertEqual(registry.resolve("lacto"), folder.resolve())
        self.assertEqual(registry.handle_for(folder), "lacto")
        registry.remove("lacto")
        self.assertIsNone(registry.resolve("lacto"))
        with self.assertRaises(KeyError):
            registry.remove("lacto")

    def test_conflicts_are_refused(self):
        registry = ProjectRegistry(self.root)
        first, second = self._outside_project("a"), self._outside_project("b")
        registry.add("proj", first)
        with self.assertRaisesRegex(ValueError, "already registered for"):
            registry.add("proj", second)
        with self.assertRaisesRegex(ValueError, "already registered as `proj`"):
            registry.add("other", first)
        (self.root / "inside").mkdir()
        with self.assertRaisesRegex(ValueError, "projects root"):
            registry.add("inside", second)

    def test_service_resolves_handles_and_rejects_paths_when_paths_are_off(self):
        folder = self._outside_project("lactonization")
        ProjectRegistry(self.root).add("lacto", folder)
        service = LabBOService(projects_root=self.root, allow_paths=False)
        self.assertEqual(service.project_path("lacto"), folder.resolve())
        self.assertEqual(service.project_path("inside"), self.root / "inside")
        for bad in ("..", str(folder), "../outside/lactonization"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                service.project_path(bad)

    def test_listing_includes_root_and_registered_projects(self):
        service = LabBOService(projects_root=self.root)
        _create_small_project(service, "inside")
        (self.root / "_backups" / "old").mkdir(parents=True)
        (self.root / "_backups" / "old" / "project.yaml").write_text("project_id: old\n", encoding="utf-8")
        ProjectRegistry(self.root).add("lacto", self._outside_project("lactonization"))
        ProjectRegistry(self.root).add("gone", self._outside_project("temporary"))
        for path in sorted((self.outside / "temporary").iterdir()):
            path.unlink()
        (self.outside / "temporary").rmdir()

        listing = {item["handle"]: item for item in service.list_projects()}
        self.assertEqual(set(listing), {"inside", "lacto", "gone"})
        self.assertFalse(listing["inside"]["registered"])
        self.assertTrue(listing["lacto"]["registered"])
        self.assertEqual(listing["lacto"]["observation_count"], 6)
        self.assertFalse(listing["gone"]["available"])
        self.assertIn("No project.yaml", listing["gone"]["error"])

    def test_create_refuses_a_registered_handle(self):
        ProjectRegistry(self.root).add("lacto", self._outside_project("lactonization"))
        with self.assertRaisesRegex(FileExistsError, "registered project"):
            _create_small_project(LabBOService(projects_root=self.root), "lacto")


class HandleApiTests(unittest.TestCase):
    def test_registered_project_is_listed_summarized_and_asked_by_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, outside = Path(tmp) / "root", Path(tmp) / "outside"
            root.mkdir()
            _create_small_project(LabBOService(), str(outside / "screening"), batch_size=2)
            ProjectRegistry(root).add("screening", outside / "screening")
            client = TestClient(create_app(root))

            listed = client.get("/api/projects").json()["projects"]
            self.assertEqual([item["handle"] for item in listed], ["screening"])

            summary = client.get("/api/projects/screening").json()
            self.assertEqual(summary["handle"], "screening")
            self.assertEqual(Path(summary["project_dir"]), (outside / "screening").resolve())
            self.assertEqual(summary["pending_recommendation_count"], 0)

            batch = client.post("/api/projects/screening/ask", json={"controller_mode": "bo_only"})
            self.assertEqual(batch.status_code, 200, batch.text)
            self.assertTrue((outside / "screening" / "recommendations_round_001.json").exists())
            self.assertEqual(client.get("/api/projects/screening").json()["pending_recommendation_count"], 2)

    def test_paths_in_urls_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(Path(tmp) / "root"))
            for url in ("/api/projects/..", "/api/projects/%2E%2E", "/api/projects/C:", "/api/projects/..%5Csecret"):
                with self.subTest(url=url):
                    self.assertIn(client.get(url).status_code, {400, 404})
            response = client.post("/api/projects/../ask", json={})
            self.assertIn(response.status_code, {400, 404, 405})


class ProjectCheckTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = Path(self._tmp.name) / "proj"
        _create_small_project(LabBOService(), str(self.folder), batch_size=2)

    def _check(self):
        from chem_agent_bo.lab.project_check import check_project

        return check_project(self.folder)

    def _messages(self, report, kind="errors"):
        return " | ".join(f"{item['file']}:{item.get('line', '')}:{item['message']}" for item in report[kind])

    def test_clean_project_passes_with_facts(self):
        LabBOService().ask(str(self.folder))
        report = self._check()
        self.assertTrue(report["ok"], self._messages(report))
        self.assertEqual(report["facts"]["combination_count"], 27)
        self.assertEqual(report["facts"]["observation_count"], 6)
        self.assertEqual(report["facts"]["completed_observation_count"], 6)
        self.assertEqual(report["facts"]["recommendation_rounds"], 1)
        self.assertEqual(report["facts"]["pending_recommendation_count"], 2)
        self.assertEqual(report["facts"]["variables"], {"Catalyst": 3, "Solvent": 3, "Base": 3})

    def test_unknown_config_key_is_an_error(self):
        config = self.folder / "project.yaml"
        config.write_text(config.read_text(encoding="utf-8") + "searching_strategy: balanced\n", encoding="utf-8")
        report = self._check()
        self.assertFalse(report["ok"])
        self.assertIn("`searching_strategy`", self._messages(report))
        self.assertIn("drop_config_keys", self._messages(report))

    def test_observation_value_outside_the_design_space_names_the_line(self):
        path = self.folder / "observations.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("cat_c", "cat_z", 1), encoding="utf-8")
        report = self._check()
        self.assertFalse(report["ok"])
        (error,) = [item for item in report["errors"] if item["file"] == "observations.csv"]
        self.assertIn("`cat_z`", error["message"])
        self.assertIn("line", error)

    def test_completed_row_without_a_number_is_an_error(self):
        path = self.folder / "observations.csv"
        path.write_text(path.read_text(encoding="utf-8").replace(",12.0,", ",n.d.,", 1), encoding="utf-8")
        self.assertIn("numeric `yield`", self._messages(self._check()))

    def test_duplicate_descriptors_are_an_error_when_descriptors_are_requested(self):
        from test_lab_mode import _descriptor_design_space

        _descriptor_design_space(duplicate=True).write_csv(self.folder / "design_space.csv")
        config = self.folder / "project.yaml"
        config.write_text(
            config.read_text(encoding="utf-8").replace("planner_use_descriptors: false", "planner_use_descriptors: true"),
            encoding="utf-8",
        )
        self.assertIn("cat_b = cat_c", self._messages(self._check()))

    def test_agentic_without_an_agent_config_is_an_error(self):
        config = self.folder / "project.yaml"
        text = config.read_text(encoding="utf-8").replace("controller_mode: bo_only", "controller_mode: agentic")
        text = text.replace("agent_config_path: configs/agent_bo.yaml", "agent_config_path: missing_agent.yaml")
        config.write_text(text, encoding="utf-8")
        self.assertIn("missing_agent.yaml", self._messages(self._check()))

    def test_round_gap_is_an_error(self):
        LabBOService().ask(str(self.folder))
        for suffix in ("json", "csv"):
            (self.folder / f"recommendations_round_001.{suffix}").rename(self.folder / f"recommendations_round_002.{suffix}")
        messages = self._messages(self._check())
        self.assertIn("without gaps", messages)
        self.assertIn("expected `round_002`", messages)

    def test_evidence_problems(self):
        cards = [
            {"card_id": "c1", "source": "s", "summary": "x", "mapping_status": "direct", "variable_scope": ["Catalyst"]},
            {"card_id": "c1", "source": "s", "summary": "y", "mapping_status": "same_redox_manifold"},
            {"card_id": "c2", "source": "s", "summary": "z", "variable_scope": ["Ligand"]},
        ]
        import json

        (self.folder / "evidence_cards.jsonl").write_text(
            "\n".join(json.dumps(card) for card in cards) + "\n", encoding="utf-8"
        )
        report = self._check()
        self.assertIn("card_id `c1` appears 2 times", self._messages(report))
        warnings = self._messages(report, "warnings")
        self.assertIn("`same_redox_manifold`", warnings)
        self.assertIn("`Ligand`", warnings)
        self.assertEqual(report["facts"]["evidence_card_count"], 3)

    def test_check_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            _create_small_project(LabBOService(projects_root=root), "inside")
            client = TestClient(create_app(root))
            self.assertTrue(client.get("/api/projects/inside/check").json()["ok"])
            self.assertEqual(client.get("/api/projects/nothing/check").status_code, 404)
            by_path = client.post("/api/projects/check", json={"path": str(self.folder)})
            self.assertEqual(by_path.status_code, 200, by_path.text)
            self.assertTrue(by_path.json()["ok"])
            self.assertEqual(client.post("/api/projects/check", json={"path": "relative/folder"}).status_code, 400)
            self.assertEqual(client.post("/api/projects/check", json={"path": str(Path(tmp) / "none")}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
