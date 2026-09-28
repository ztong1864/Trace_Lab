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


def _load_cli():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "experiments" / "run_lab_bo.py"
    spec = importlib.util.spec_from_file_location("run_lab_bo_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.root = base / "root"
        self.outside = base / "lab data"
        self.root.mkdir()
        self.service = LabBOService(projects_root=self.root, allow_paths=False)

    def _project(self, name: str) -> Path:
        _create_small_project(LabBOService(), str(self.outside / name))
        return self.outside / name

    def _add_config_line(self, folder: Path, line: str) -> None:
        config = folder / "project.yaml"
        config.write_text(config.read_text(encoding="utf-8") + line, encoding="utf-8")

    def test_register_uses_the_folder_in_place(self):
        folder = self._project("screening v2")
        result = self.service.register_project(folder)
        self.assertEqual(result["handle"], "screening_v2")
        self.assertTrue(result["check"]["ok"])
        self.service.ask("screening_v2", controller_mode="bo_only")
        self.assertTrue((folder / "recommendations_round_001.json").exists())
        self.assertEqual(self.service.register_project(self._project("other"), handle="mine")["handle"], "mine")

    def test_failing_check_registers_nothing(self):
        folder = self._project("broken")
        path = folder / "observations.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("cat_c", "cat_z", 1), encoding="utf-8")
        from chem_agent_bo.lab.service import ProjectCheckError

        with self.assertRaises(ProjectCheckError) as caught:
            self.service.register_project(folder)
        self.assertFalse(caught.exception.report["ok"])
        self.assertEqual(self.service.list_projects(), [])

    def test_drop_config_keys_backs_up_and_fixes_the_config(self):
        folder = self._project("collab")
        self._add_config_line(folder, "searching_strategy: balanced\n")
        original = (folder / "project.yaml").read_text(encoding="utf-8")
        result = self.service.register_project(folder, drop_config_keys=["searching_strategy"])
        self.assertTrue(result["check"]["ok"])
        self.assertEqual(Path(result["config_backup_path"]).read_text(encoding="utf-8"), original)
        fixed = (folder / "project.yaml").read_text(encoding="utf-8")
        self.assertNotIn("searching_strategy", fixed)
        self.assertIn("planner_name: atlas", fixed)

    def test_config_is_restored_when_the_check_still_fails(self):
        folder = self._project("collab")
        self._add_config_line(folder, "searching_strategy: balanced\n")
        path = folder / "observations.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("cat_c", "cat_z", 1), encoding="utf-8")
        original = (folder / "project.yaml").read_text(encoding="utf-8")
        from chem_agent_bo.lab.service import ProjectCheckError

        with self.assertRaises(ProjectCheckError):
            self.service.register_project(folder, drop_config_keys=["searching_strategy"])
        self.assertEqual((folder / "project.yaml").read_text(encoding="utf-8"), original)

    def test_refusals(self):
        folder = self._project("p")
        with self.assertRaisesRegex(ValueError, "TRACE setting"):
            self.service.register_project(folder, drop_config_keys=["planner_name"])
        with self.assertRaisesRegex(ValueError, "absolute"):
            self.service.register_project("relative/p")
        with self.assertRaises(FileNotFoundError):
            self.service.register_project(self.outside)  # no project.yaml
        _create_small_project(LabBOService(projects_root=self.root), "inside")
        with self.assertRaisesRegex(ValueError, "already in the projects root"):
            self.service.register_project(self.root / "inside")
        from chem_agent_bo.lab.service import ProjectBusyError, _project_ask_lock

        lock = _project_ask_lock(folder)
        lock.acquire()
        try:
            with self.assertRaises(ProjectBusyError):
                self.service.register_project(folder)
        finally:
            lock.release()

    def test_api_register_and_unregister(self):
        folder = self._project("screening")
        broken = self._project("broken")
        self._add_config_line(broken, "searching_strategy: balanced\n")
        client = TestClient(create_app(self.root))

        ok = client.post("/api/projects/register", json={"path": str(folder)})
        self.assertEqual(ok.status_code, 200, ok.text)
        refused = client.post("/api/projects/register", json={"path": str(broken)})
        self.assertEqual(refused.status_code, 400)
        self.assertIn("searching_strategy", str(refused.json()["detail"]["check"]["errors"]))
        fixed = client.post(
            "/api/projects/register", json={"path": str(broken), "drop_config_keys": ["searching_strategy"]}
        )
        self.assertEqual(fixed.status_code, 200, fixed.text)
        self.assertEqual(
            sorted(item["handle"] for item in client.get("/api/projects").json()["projects"]),
            ["broken", "screening"],
        )

        self.assertEqual(client.delete("/api/projects/screening/registration").status_code, 200)
        self.assertEqual(client.delete("/api/projects/screening/registration").status_code, 404)
        self.assertTrue((folder / "project.yaml").exists())
        self.assertEqual([item["handle"] for item in client.get("/api/projects").json()["projects"]], ["broken"])

    def test_cli_scan(self):
        cli = _load_cli()
        good_a, good_b = self._project("a"), self._project("b")
        collab = self._project("collab")
        self._add_config_line(collab, "searching_strategy: balanced\n")
        _create_small_project(LabBOService(), str(self.outside / "x" / "same"))
        _create_small_project(LabBOService(), str(self.outside / "y" / "same"))
        _create_small_project(LabBOService(), str(self.outside / "_backups" / "old"))

        dry = cli._register_scan(LabBOService(projects_root=self.root), self.outside, dry_run=True)
        statuses = {
            Path(item["project_dir"]).relative_to(self.outside.resolve()).as_posix(): item["status"]
            for item in dry["projects"]
        }
        self.assertEqual(
            statuses,
            {
                "a": "would_register",
                "b": "would_register",
                "collab": "check_failed",
                "x/same": "handle_conflict",
                "y/same": "handle_conflict",
            },
        )
        self.assertEqual(self.service.list_projects(), [])

        real = cli._register_scan(LabBOService(projects_root=self.root), self.outside, dry_run=False)
        self.assertEqual(real["counts"], {"registered": 2, "check_failed": 1, "handle_conflict": 2})
        again = cli._register_scan(LabBOService(projects_root=self.root), self.outside, dry_run=False)
        self.assertEqual(again["counts"]["already_registered"], 2)
        self.assertEqual(sorted(item["handle"] for item in self.service.list_projects()), ["a", "b"])
        self.assertTrue(good_a.exists() and good_b.exists())


_CATALYST_TABLE = [
    {"name": "cat_a", "d1": "1.0", "d2": "0.2"},
    {"name": "cat_b", "d1": "2.0", "d2": "0.1"},
    {"name": "cat_c", "d1": "3.5", "d2": "0.9"},
    {"name": "cat_unused", "d1": "9.0", "d2": "9.0"},
]
_SOLVENT_TABLE = [
    {"solvent": "dce", "eps": "10.4"},
    {"solvent": "meoh", "eps": "32.7"},
    {"solvent": "thf", "eps": "7.6"},
]
_BASE_TABLE = [
    {"base": "k2co3", "pka": "10.3"},
    {"base": "cs2co3", "pka": "10.0"},
    {"base": "et3n", "pka": "10.8"},
]


class DescriptorTableTests(unittest.TestCase):
    def setUp(self):
        from test_lab_mode import _small_design_space

        self.design = _small_design_space()

    def _attach(self, rows, **kwargs):
        kwargs.setdefault("value_column", "name")
        return self.design.with_descriptor_table("Catalyst", rows, **kwargs)

    def _error(self, rows, **kwargs):
        from chem_agent_bo.lab.design_space import DescriptorTableError

        with self.assertRaises(DescriptorTableError) as caught:
            self._attach(rows, **kwargs)
        return str(caught.exception), caught.exception.report

    def test_exact_match(self):
        updated, report = self._attach(_CATALYST_TABLE)
        self.assertEqual(report["numeric_columns"], ["d1", "d2"])
        self.assertEqual(report["unused_rows"], ["cat_unused"])
        self.assertEqual(updated.numeric_descriptor_matrix("Catalyst")["matrix"], [[1.0, 0.2], [2.0, 0.1], [3.5, 0.9]])
        # The original design space is untouched.
        self.assertEqual(self.design.numeric_descriptor_matrix("Catalyst")["descriptor_keys"], [])

    def test_aliases_and_prefixed_columns(self):
        rows = [
            {"name": "CatA", "descriptor__d1": "1.0"},
            {"name": "cat_b", "descriptor__d1": "2.0"},
            {"name": "cat_c", "descriptor__d1": "3.0"},
        ]
        updated, report = self._attach(rows, aliases={"cat_a": "CatA"})
        self.assertEqual(report["matched_by_alias"], {"cat_a": "CatA"})
        self.assertEqual(updated.numeric_descriptor_matrix("Catalyst")["descriptor_keys"], ["d1"])

    def test_every_option_needs_a_row(self):
        message, report = self._error(_CATALYST_TABLE[:2])
        self.assertIn("['cat_c']", message)
        self.assertIn("aliases", message)

    def test_table_problems(self):
        self.assertIn("No `formula` column", self._error(_CATALYST_TABLE, value_column="formula")[0])
        self.assertIn("more than once", self._error(_CATALYST_TABLE + [_CATALYST_TABLE[0]])[0])
        self.assertIn("not options", self._error(_CATALYST_TABLE, aliases={"cat_x": "cat_a"})[0])
        same = [dict(row, d1="1.0", d2="0.2") if row["name"] == "cat_b" else row for row in _CATALYST_TABLE]
        self.assertIn("cat_a = cat_b", self._error(same)[0])
        text = [dict(row, d1="high", d2="low") for row in _CATALYST_TABLE]
        message, report = self._error(text)
        self.assertIn("no numeric descriptor", message)
        self.assertEqual(report["non_numeric_columns"], ["d1", "d2"])
        with self.assertRaisesRegex(ValueError, "No variable"):
            self.design.with_descriptor_table("Ligand", _CATALYST_TABLE, value_column="name")

    def test_existing_columns_need_replace(self):
        updated, _ = self._attach(_CATALYST_TABLE)
        self.design = updated
        self.assertIn("replace=true", self._error(_CATALYST_TABLE)[0])
        replacement = [{"name": row["name"], "d1": str(float(row["d1"]) * 10)} for row in _CATALYST_TABLE]
        replaced, _ = self._attach(replacement, replace=True)
        self.assertEqual(replaced.numeric_descriptor_matrix("Catalyst")["descriptor_keys"], ["d1"])
        self.assertEqual(replaced.numeric_descriptor_matrix("Catalyst")["matrix"][0], [10.0])

    def test_service_writes_backs_up_and_enables_descriptor_asks(self):
        import csv
        import io

        def as_csv(rows):
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            return out.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            service = LabBOService(projects_root=root, allow_paths=False)
            project = _create_small_project(
                LabBOService(projects_root=root), "desc", planner_name="chunked_gp", planner_use_descriptors=True
            )
            client = TestClient(create_app(root))
            original = project.design_space_path.read_text(encoding="utf-8")

            bad = client.post(
                "/api/projects/desc/descriptors",
                json={"variable": "Catalyst", "value_column": "name", "csv": as_csv(_CATALYST_TABLE[:2])},
            )
            self.assertEqual(bad.status_code, 400)
            self.assertIn("cat_c", bad.json()["detail"]["message"])
            self.assertEqual(project.design_space_path.read_text(encoding="utf-8"), original)

            first = client.post(
                "/api/projects/desc/descriptors",
                json={"variable": "Catalyst", "value_column": "name", "csv": as_csv(_CATALYST_TABLE)},
            )
            self.assertEqual(first.status_code, 200, first.text)
            self.assertEqual(Path(first.json()["backup_path"]).read_text(encoding="utf-8"), original)
            self.assertFalse(first.json()["descriptor_eligibility"]["eligible"])  # Solvent, Base still missing
            with self.assertRaisesRegex(ValueError, "no complete numeric descriptors"):
                service.ask("desc")

            service.add_descriptor_table("desc", variable="Solvent", value_column="solvent", rows=_SOLVENT_TABLE)
            result = service.add_descriptor_table("desc", variable="Base", value_column="base", csv_text=as_csv(_BASE_TABLE))
            self.assertTrue(result["descriptor_eligibility"]["eligible"])
            batch = service.ask("desc")
            self.assertTrue(batch["planner_use_descriptors"])
            self.assertEqual(len(batch["recommendations"]), 2)

            with self.assertRaisesRegex(ValueError, "exactly one"):
                service.add_descriptor_table("desc", variable="Base", value_column="base")

    def test_busy_project_is_refused(self):
        from chem_agent_bo.lab.service import _project_ask_lock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            project = _create_small_project(LabBOService(projects_root=root), "desc")
            lock = _project_ask_lock(project.project_dir)
            lock.acquire()
            try:
                response = TestClient(create_app(root)).post(
                    "/api/projects/desc/descriptors",
                    json={"variable": "Catalyst", "value_column": "name", "rows": _CATALYST_TABLE},
                )
            finally:
                lock.release()
            self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
