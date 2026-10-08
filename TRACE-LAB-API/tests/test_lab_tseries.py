"""Failed runs, register set_config, evidence status aliases, numeric descriptors, the planner
default and bo_only model estimates."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

import yaml
from fastapi.testclient import TestClient

from chem_agent_bo.lab import service as lab_service
from chem_agent_bo.lab.api import create_app
from chem_agent_bo.lab.design_space import DesignSpace
from chem_agent_bo.lab.evidence import EvidenceCard, EvidenceImportError, EvidenceStore
from chem_agent_bo.lab.project import LabProject, ProjectConfig
from chem_agent_bo.lab.project_check import check_project
from chem_agent_bo.lab.service import LabBOService, ProjectCheckError

from test_lab_mode import _create_small_project

VARIABLES = ("Catalyst", "Solvent", "Base")


def _key(candidate: dict) -> tuple[str, ...]:
    return tuple(str(candidate[name]) for name in VARIABLES)


class _TempService(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.root = self.base / "root"
        self.root.mkdir()
        self.service = LabBOService(projects_root=self.root)


class FailedRunTests(_TempService):
    """T-A: a failed or skipped condition is not proposed again; a measured failure is learned from."""

    def _project_with_failures(self, project_id: str, **overrides):
        _create_small_project(self.service, project_id, planner_name="chunked_gp", **overrides)
        batch = self.service.ask(project_id)
        first, second = batch["recommendations"][:2]
        self.service.tell(project_id, results=[
            {"recommendation_id": first["recommendation_id"], "yield": 0, "status": "failed",
             "failure_reason": "no product"},
            {"recommendation_id": second["recommendation_id"], "status": "skipped"},
        ], reflect_results=False)
        return _key(first["candidate"]), _key(second["candidate"])

    def test_a_measured_failure_is_kept_and_used_by_the_model_only(self):
        failed, skipped = self._project_with_failures("p", goal="minimize")
        observations = LabProject(self.root / "p").load_observations()
        rows = {_key(row): row for row in observations.rows if row.get("source") == "recommendation"}
        self.assertEqual(rows[failed]["yield"], "0.0")
        self.assertEqual(rows[skipped]["yield"], "")
        model_keys = {_key(row) for row in observations.model_rows("yield")}
        self.assertIn(failed, model_keys)
        self.assertNotIn(skipped, model_keys)
        # A failure never becomes the best result, even when lower is better.
        self.assertEqual(observations.best_so_far(objective_name="yield", goal="minimize"), 8.0)
        batch = LabProject(self.root / "p").load_batches()[0]
        self.assertEqual(batch.recommendations[0]["result"], "0.0")

    def test_failed_and_skipped_conditions_are_not_proposed_again(self):
        failed, skipped = self._project_with_failures("p")
        # 27 combinations - 6 completed - 2 failed/skipped = 19 left; ask for more than that.
        batch = self.service.ask("p", batch_size=25)
        keys = {_key(item["candidate"]) for item in batch["recommendations"]}
        self.assertEqual(len(keys), 19)
        self.assertNotIn(failed, keys)
        self.assertNotIn(skipped, keys)

    def test_repeat_failed_conditions_allows_unmeasured_ones_again(self):
        _create_small_project(self.service, "p", planner_name="chunked_gp")
        measured, unmeasured, skipped = self.service.ask("p", batch_size=3)["recommendations"]
        self.service.tell("p", results=[
            {"recommendation_id": measured["recommendation_id"], "yield": 0, "status": "failed"},
            {"recommendation_id": unmeasured["recommendation_id"], "status": "failed", "failure_reason": "stirrer broke"},
            {"recommendation_id": skipped["recommendation_id"], "status": "skipped"},
        ], reflect_results=False)
        self.service.update_project_config("p", {"repeat_failed_conditions": True})
        batch = self.service.ask("p", batch_size=25)
        keys = {_key(item["candidate"]) for item in batch["recommendations"]}
        # A failure measured as yield 0 is a result like any other, so it is not repeated.
        self.assertEqual(len(keys), 20)
        self.assertNotIn(_key(measured["candidate"]), keys)
        self.assertIn(_key(unmeasured["candidate"]), keys)
        self.assertIn(_key(skipped["candidate"]), keys)
        with self.assertRaisesRegex(ValueError, "true or false"):
            self.service.update_project_config("p", {"repeat_failed_conditions": "yes"})

    def test_a_retry_of_a_failed_condition_can_still_be_imported(self):
        failed, _ = self._project_with_failures("p")
        result = self.service.import_observations(
            "p", rows=[{**dict(zip(VARIABLES, failed)), "yield": "22", "status": "completed"}])
        self.assertEqual(result["imported_count"], 1)

    def test_a_failed_result_with_a_non_number_is_refused(self):
        _create_small_project(self.service, "p", planner_name="chunked_gp")
        rec = self.service.ask("p")["recommendations"][0]
        with self.assertRaisesRegex(ValueError, "must be a number or left out"):
            self.service.tell("p", results=[{"recommendation_id": rec["recommendation_id"], "yield": "n/a",
                                              "status": "failed"}], reflect_results=False)

    def test_the_agent_is_told_about_failed_conditions(self):
        failed, skipped = self._project_with_failures("p")
        project = LabProject(self.root / "p")
        context = lab_service._lab_reaction_context(
            project.load_config(), observations=project.load_observations(), variable_names=list(VARIABLES))
        listed = {_key(item["candidate"]): item for item in context["failed_conditions"]}
        self.assertEqual(set(listed), {failed, skipped})
        self.assertEqual(listed[failed]["failure_reason"], "no product")
        self.assertEqual(listed[failed]["yield"], "0.0")
        self.assertIn("excluded", context["failed_conditions_policy"])
        self.assertNotIn("failed_conditions", lab_service._lab_reaction_context(project.load_config()))


class RegisterSetConfigTests(_TempService):
    """T-B: a folder from another server registers once its stale settings are changed."""

    def _outside(self, name: str = "moved") -> Path:
        folder = self.base / "outside" / name
        _create_small_project(LabBOService(), str(folder))
        path = folder / "project.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload.update(controller_mode="agentic", agent_config_path="/srv/other_server/agent_bo.yaml")
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return folder

    def test_stale_agent_config_path_is_refused_with_a_hint(self):
        folder = self._outside()
        with self.assertRaises(ProjectCheckError) as caught:
            self.service.register_project(folder)
        self.assertIn("set_config", json.dumps(caught.exception.report["errors"]))

    def test_set_config_repairs_it_with_a_backup(self):
        folder = self._outside()
        original = (folder / "project.yaml").read_text(encoding="utf-8")
        result = self.service.register_project(folder, set_config={"agent_config_path": "configs/agent_bo.yaml"})
        self.assertTrue(result["check"]["ok"])
        self.assertEqual(result["config_changes"]["agent_config_path"],
                         {"old": "/srv/other_server/agent_bo.yaml", "new": "configs/agent_bo.yaml"})
        self.assertEqual(Path(result["config_backup_path"]).read_text(encoding="utf-8"), original)
        self.assertEqual(LabProject(folder).load_config().agent_config_path, "configs/agent_bo.yaml")

    def test_bad_values_and_locked_settings_are_refused_and_nothing_changes(self):
        folder = self._outside()
        original = (folder / "project.yaml").read_text(encoding="utf-8")
        for bad, message in (({"batch_size": "x"}, "integer"), ({"objective_name": "ee"}, "can't be changed"),
                             ({"searching_strategy": 1}, "non-editable")):
            with self.subTest(set_config=bad), self.assertRaisesRegex(ValueError, message):
                self.service.register_project(folder, set_config=bad)
            self.assertEqual((folder / "project.yaml").read_text(encoding="utf-8"), original)
        # Valid as a value, but the project still fails the check: the original file comes back.
        with self.assertRaises(ProjectCheckError):
            self.service.register_project(folder, set_config={"planner_name": "bogus"})
        self.assertEqual((folder / "project.yaml").read_text(encoding="utf-8"), original)
        self.assertEqual(self.service.list_projects(), [])

    def test_api_register_takes_set_config(self):
        folder = self._outside()
        client = TestClient(create_app(self.root))
        response = client.post("/api/projects/register", json={
            "path": str(folder), "set_config": {"agent_config_path": "configs/agent_bo.yaml"}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("agent_config_path", response.json()["config_changes"])


class MappingStatusAliasTests(_TempService):
    """T-C: words from the group's card guide are read as same_reaction_family everywhere."""

    def test_load_reads_aliases(self):
        for alias in ("same_redox_manifold", "same_reaction", "Same_Redox_Manifold"):
            card = EvidenceCard(card_id="c", source="s", summary="x", mapping_status=alias).normalized()
            self.assertEqual(card.mapping_status, "same_reaction_family")
        self.assertEqual(EvidenceCard(card_id="c", source="s", summary="x", mapping_status="maybe").normalized()
                         .mapping_status, "background")

    def test_import_converts_aliases_and_still_refuses_unknown_words(self):
        _create_small_project(self.service, "p")
        result = self.service.import_evidence("p", cards=[
            {"card_id": "c1", "summary": "x", "mapping_status": "same_redox_manifold"},
            {"card_id": "c2", "summary": "y", "mapping_status": "direct"},
        ])
        self.assertEqual(result["status_notes"], ["c1: mapping_status `same_redox_manifold` stored as `same_reaction_family`."])
        stored = {card.card_id: card.mapping_status
                  for card in EvidenceStore.load(LabProject(self.root / "p").evidence_path).cards}
        self.assertEqual(stored, {"c1": "same_reaction_family", "c2": "direct"})
        with self.assertRaises(EvidenceImportError) as caught:
            self.service.import_evidence("p", cards=[{"card_id": "c3", "summary": "z", "mapping_status": "maybe_related"}])
        self.assertIn("maybe_related", " ".join(caught.exception.problems))

    def test_check_reports_aliases_as_a_fact_not_a_warning(self):
        project = _create_small_project(self.service, "p")
        project.evidence_path.write_text("\n".join(json.dumps(card) for card in (
            {"card_id": "c1", "source": "s", "summary": "x", "mapping_status": "same_redox_manifold"},
            {"card_id": "c2", "source": "s", "summary": "y", "mapping_status": "same_redox_manifold"},
            {"card_id": "c3", "source": "s", "summary": "z", "mapping_status": "maybe_related"},
        )) + "\n", encoding="utf-8")
        report = check_project(project.project_dir)
        self.assertEqual(report["facts"]["mapping_status_aliases"],
                         {"same_redox_manifold": {"read_as": "same_reaction_family", "card_count": 2}})
        warnings = json.dumps(report["warnings"])
        self.assertNotIn("same_redox_manifold", warnings)
        self.assertIn("maybe_related", warnings)


def _solvent_temperature_design(*, base_descriptors: bool = True) -> DesignSpace:
    records = [
        {"variable": "Solvent", "type": "categorical", "value": value, "descriptor__eps": eps}
        for value, eps in (("dce", 10.4), ("meoh", 32.7), ("thf", 7.6))
    ]
    records += [{"variable": "Temperature", "type": "discrete_numeric", "value": value, "unit": "C"}
                for value in ("25", "50", "70")]
    records += [
        {"variable": "Base", "type": "categorical", "value": value, **({"descriptor__pka": pka} if base_descriptors else {})}
        for value, pka in (("k2co3", 10.3), ("cs2co3", 10.0), ("et3n", 10.8))
    ]
    return DesignSpace.from_long_records(records)


class NumericDescriptorTests(_TempService):
    """T-E: numeric variables need no descriptor table; categorical ones still do."""

    def test_numeric_variable_uses_its_own_values(self):
        design = _solvent_temperature_design()
        eligibility = design.planner_descriptor_eligibility()
        self.assertTrue(eligibility["eligible"], eligibility)
        self.assertEqual(eligibility["own_value_variables"], ["Temperature"])
        matrix = design.numeric_descriptor_matrix("Temperature")
        self.assertEqual(matrix["descriptor_keys"], ["value"])
        self.assertEqual(matrix["matrix"], [[25.0], [50.0], [70.0]])
        self.assertIn("Temperature", design.descriptor_overview()["enabled_variables"])

    def test_a_categorical_variable_without_a_table_is_still_refused_clearly(self):
        eligibility = _solvent_temperature_design(base_descriptors=False).planner_descriptor_eligibility()
        self.assertFalse(eligibility["eligible"])
        self.assertEqual(eligibility["missing_descriptor_variables"], ["Base"])
        message = lab_service._descriptor_ineligibility_error(eligibility)
        self.assertIn("`Base`", message)
        self.assertIn("categorical one needs a descriptor table", message)

    def test_ask_with_descriptors_runs(self):
        design = _solvent_temperature_design()
        self.service.create_project("d", config=ProjectConfig(
            project_id="d", planner_name="chunked_gp", controller_mode="bo_only", batch_size=2,
            planner_use_descriptors=True), design_space=design)
        self.service.import_observations("d", rows=[
            {"Solvent": solvent, "Temperature": temp, "Base": base, "yield": value, "status": "completed"}
            for solvent, temp, base, value in (("dce", "25", "k2co3", "12"), ("meoh", "50", "cs2co3", "35"),
                                               ("thf", "70", "et3n", "20"), ("dce", "70", "cs2co3", "28"))
        ])
        batch = self.service.ask("d")
        self.assertTrue(batch["planner_use_descriptors"])
        self.assertEqual(len(batch["recommendations"]), 2)


def _large_design(options: int = 20) -> DesignSpace:
    records = []
    for name in ("A", "B", "C", "D"):
        records.extend({"variable": name, "type": "categorical", "value": f"{name.lower()}{i}"} for i in range(options))
    return DesignSpace.from_long_records(records)


class PlannerDefaultTests(_TempService):
    """T-F: chunked_gp by default; Atlas from project.yaml is switched on large spaces."""

    def test_new_projects_default_to_chunked_gp(self):
        self.assertEqual(ProjectConfig(project_id="x").normalized().planner_name, "chunked_gp")
        self.assertEqual(ProjectConfig(project_id="x", planner_name="").normalized().planner_name, "chunked_gp")

    def _atlas_project(self):
        self.service.create_project("big", config=ProjectConfig(
            project_id="big", planner_name="atlas", controller_mode="bo_only", batch_size=2), design_space=_large_design())
        self.service.import_observations("big", rows=[
            {"A": f"a{i}", "B": f"b{i}", "C": f"c{i}", "D": f"d{i}", "yield": str(10 + 5 * i), "status": "completed"}
            for i in range(4)
        ])

    def test_atlas_on_a_large_space_runs_chunked_gp_for_the_round(self):
        self._atlas_project()
        batch = self.service.ask("big")
        self.assertEqual(batch["planner_switched"]["from"], "atlas")
        self.assertEqual(batch["planner_switched"]["combinations"], 160_000)
        self.assertIn("chunked_gp", batch["planner_warnings"][0])
        self.assertTrue(all(item["planner_name"] == "chunked_gp" for item in batch["recommendations"]))
        self.assertEqual(LabProject(self.root / "big").load_config().planner_name, "atlas")
        report = check_project(self.root / "big")
        self.assertTrue(any("instead" in item["message"] for item in report["warnings"]), report["warnings"])

    def test_an_ask_that_names_atlas_keeps_it(self):
        self._atlas_project()
        built = []

        def refuse_atlas(name, **kwargs):
            built.append(name)
            raise RuntimeError("stop before running atlas")

        with mock.patch.object(lab_service, "build_planner", side_effect=refuse_atlas):
            with self.assertRaisesRegex(RuntimeError, "stop before"):
                self.service.ask("big", planner_name="atlas")
        self.assertEqual(built, ["atlas"])


class BoOnlyEstimateTests(_TempService):
    """bo_only: each recommendation carries the model's prediction, so "why this one?" has an answer."""

    def test_recommendations_carry_a_model_estimate(self):
        for goal in ("maximize", "minimize"):
            with self.subTest(goal=goal):
                _create_small_project(self.service, f"p_{goal}", planner_name="chunked_gp", goal=goal)
                batch = self.service.ask(f"p_{goal}")
                for item in batch["recommendations"]:
                    estimate = item["model_estimate"]
                    # Observed yields are 8-40: a sign or scale slip would land far outside.
                    self.assertTrue(-10 < estimate["predicted"] < 60, estimate)
                    self.assertGreaterEqual(estimate["uncertainty"], 0.0)
                    self.assertIn("Model prediction for this condition: yield", item["rationale"])
                stored = LabProject(self.root / f"p_{goal}").load_batches()[0].recommendations[0]
                self.assertIn("model_estimate", stored)

    def test_a_random_initial_design_has_no_estimate(self):
        _create_small_project(self.service, "empty", planner_name="chunked_gp", observation_count=0)
        batch = self.service.ask("empty")
        self.assertTrue(all("model_estimate" not in item for item in batch["recommendations"]))
        self.assertIn("random initial design", batch["recommendations"][0]["rationale"])

    def test_estimates_follow_the_candidate_not_the_position(self):
        space = _create_small_project(self.service, "p", observation_count=0).load_design_space()
        pool = [{"candidate": {"Catalyst": "cat_a", "Solvent": "dce", "Base": "k2co3"},
                 "model_estimate": {"predicted": 1.0, "uncertainty": 0.1}},
                {"candidate": {"Catalyst": "cat_b", "Solvent": "dce", "Base": "k2co3"},
                 "model_estimate": {"predicted": 2.0, "uncertainty": 0.2}}]
        recommendations = [{"candidate": {"Catalyst": "cat_b", "Solvent": "dce", "Base": "k2co3"}},
                           {"candidate": {"Catalyst": "cat_c", "Solvent": "dce", "Base": "k2co3"}}]
        lab_service._attach_model_estimates(recommendations, pool, space, list(VARIABLES))
        self.assertEqual(recommendations[0]["model_estimate"]["predicted"], 2.0)
        self.assertNotIn("model_estimate", recommendations[1])


if __name__ == "__main__":
    unittest.main()
