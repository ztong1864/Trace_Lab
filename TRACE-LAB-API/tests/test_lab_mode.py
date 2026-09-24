import collections
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from chem_agent_bo.bo.atlas_bo import AtlasBOTool
from chem_agent_bo.bo.base import ExclusionConstraint
from chem_agent_bo.bo.chunked_gp import ChunkedGPPlanner, _LatentPosterior, _merge_top_k
from chem_agent_bo.config.schema import AgenticBOConfig
from chem_agent_bo.lab import service as lab_service
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


def _small_design_space() -> DesignSpace:
    records = []
    for name, values in (
        ("Catalyst", ("cat_a", "cat_b", "cat_c")),
        ("Solvent", ("dce", "meoh", "thf")),
        ("Base", ("k2co3", "cs2co3", "et3n")),
    ):
        records.extend({"variable": name, "type": "categorical", "value": value} for value in values)
    return DesignSpace.from_long_records(records)


def _create_small_project(
    service: LabBOService,
    project_id: str,
    *,
    observation_count: int = 6,
    **config_overrides,
) -> LabProject:
    config = {
        "project_id": project_id,
        "objective_name": "yield",
        "planner_name": "atlas",
        "controller_mode": "bo_only",
        "batch_size": 2,
        **config_overrides,
    }
    project = service.create_project(
        project_id,
        config=ProjectConfig(**config),
        design_space=_small_design_space(),
    )
    rows = [
        {"Catalyst": cat, "Solvent": solvent, "Base": base, "yield": value, "status": "completed"}
        for cat, solvent, base, value in (
            ("cat_a", "dce", "k2co3", "12"),
            ("cat_b", "meoh", "cs2co3", "35"),
            ("cat_c", "thf", "et3n", "20"),
            ("cat_a", "thf", "cs2co3", "28"),
            ("cat_b", "dce", "et3n", "40"),
            ("cat_c", "meoh", "k2co3", "8"),
        )
    ][:observation_count]
    if rows:
        service.import_observations(project_id, rows=rows, source="historical")
    return project


class _FailingPlanner:
    def suggest_shortlist(self, **_kwargs):
        raise RuntimeError("simulated planner crash")

    def planner_diagnostics(self):
        return {"planner_name": "atlas"}


_REAL_BUILD_PLANNER = lab_service.build_planner


def _atlas_always_fails(name, **kwargs):
    if name == "atlas":
        return _FailingPlanner()
    return _REAL_BUILD_PLANNER(name, **kwargs)


class PlannerReliabilityTests(unittest.TestCase):
    def test_unsupported_acquisition_fails_the_ask_but_not_the_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = LabBOService(projects_root=tmp)
            project = _create_small_project(service, "bad_acq", acquisition_function=" EI_UCB ")
            self.assertEqual(project.load_config().acquisition_function, "ei_ucb")
            self.assertEqual(project.summary()["project"]["project_id"], "bad_acq")
            with self.assertRaisesRegex(ValueError, "atlas planner does not support acquisition function `ei_ucb`"):
                service.ask("bad_acq")
            self.assertEqual(project.load_batches(), [])

    def test_boolean_settings_parse_strings(self):
        config = ProjectConfig(
            project_id="p",
            allow_random_fallback="false",
            planner_use_descriptors="true",
        ).normalized()
        self.assertFalse(config.allow_random_fallback)
        self.assertTrue(config.planner_use_descriptors)
        self.assertTrue(ProjectConfig(project_id="p").normalized().allow_random_fallback)

    def test_batches_written_before_planner_error_field_get_it_from_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = LabBOService(projects_root=tmp)
            project = _create_small_project(service, "legacy")
            payload = {
                "round_id": "round_001",
                "created_at": "",
                "recommendations": [{"recommendation_id": "round_001_rec_001", "planner_name": "random"}],
                "trace_records": [{"recommendation_id": "round_001_rec_001", "planner_error": "TypeError: boom"}],
            }
            project.recommendation_json_path("round_001").write_text(json.dumps(payload), encoding="utf-8")
            recommendation = project.load_batches()[0].recommendations[0]
            self.assertEqual(recommendation["planner_error"], "TypeError: boom")

    def test_decision_engine_receives_llm_runtime_settings(self):
        try:
            from chem_agent_bo.agent.decision_engine import DecisionEngine
        except ImportError:  # pragma: no cover
            self.skipTest("DecisionEngine dependencies are not installed.")
        config = AgenticBOConfig()
        config.runtime.use_responses_api = True
        config.runtime.reasoning_effort = "xhigh"
        config.runtime.disable_response_storage = True
        captured: dict = {}

        def capture(model_kwargs):
            captured.update(model_kwargs)
            return collections.defaultdict(mock.MagicMock)

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), mock.patch.object(
            DecisionEngine, "_build_agent_bundle", side_effect=capture
        ):
            lab_service._build_decision_engine(config)
        self.assertTrue(captured["use_responses_api"])
        self.assertEqual(captured["reasoning_effort"], "xhigh")
        self.assertIs(captured["store"], False)

    def test_atlas_tool_never_uses_atlas_batch_mode(self):
        tool = AtlasBOTool(acquisition_type="ucb")
        self.assertEqual(tool.planner.batch_size, 1)
        self.assertEqual(tool.planner_diagnostics()["acquisition_name"], "ucb")

    def test_atlas_ask_uses_requested_acquisition_without_fallback(self):
        for acquisition in ("ei", "ucb"):
            with self.subTest(acquisition=acquisition), tempfile.TemporaryDirectory() as tmp:
                service = LabBOService(projects_root=tmp)
                _create_small_project(service, "atlas_smoke", acquisition_function=acquisition)
                batch = service.ask("atlas_smoke")
                self.assertEqual(batch["planner_fallback"], {"used": False})
                self.assertEqual(len(batch["recommendations"]), 2)
                for item in batch["recommendations"]:
                    self.assertEqual(item["planner_name"], "atlas")
                    self.assertEqual(item["planner_error"], "")

    def test_planner_failure_falls_back_with_visible_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = LabBOService(projects_root=tmp)
            _create_small_project(service, "fallback")
            with mock.patch.object(
                lab_service,
                "build_planner",
                side_effect=_atlas_always_fails,
            ):
                batch = service.ask("fallback")
            fallback = batch["planner_fallback"]
            self.assertTrue(fallback["used"])
            self.assertEqual(fallback["requested_planner"], "atlas")
            self.assertIn("simulated planner crash", fallback["error"])
            for item in batch["recommendations"]:
                self.assertEqual(item["planner_name"], "random")
                self.assertIn("simulated planner crash", item["planner_error"])

    def test_planner_failure_without_fallback_writes_no_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = LabBOService(projects_root=tmp)
            project = _create_small_project(service, "strict", allow_random_fallback=False)
            with mock.patch.object(
                lab_service,
                "build_planner",
                side_effect=_atlas_always_fails,
            ), self.assertRaisesRegex(RuntimeError, "allow_random_fallback"):
                service.ask("strict")
            self.assertEqual(project.load_batches(), [])


def _planner_inputs(project: LabProject, *, include_descriptors: bool = False):
    config = project.load_config()
    design = project.load_design_space()
    param_space = design.param_space(include_descriptors=include_descriptors)
    campaign = lab_service._campaign_from_observations(
        observations=project.load_observations(),
        design_space=design,
        param_space=param_space,
        value_space=lab_service._value_space(config.objective_name),
        variable_names=design.variable_names,
        objective_name=config.objective_name,
    )
    return design, param_space, campaign


def _keys(samples, param_space) -> list[tuple[str, ...]]:
    return [tuple(str(sample.to_dict()[param.name]) for param in param_space) for sample in samples]


class ChunkedGPPlannerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.service = LabBOService(projects_root=self._tmp.name)
        self.project = _create_small_project(self.service, "chunked", planner_name="chunked_gp")
        self.design, self.param_space, self.campaign = _planner_inputs(self.project)
        self.observed = {
            (row["Catalyst"], row["Solvent"], row["Base"]) for row in self.project.load_observations().rows
        }

    def tearDown(self):
        self._tmp.cleanup()

    def _shortlist(self, size: int, **planner_kwargs):
        planner = ChunkedGPPlanner(seed=3, **planner_kwargs)
        samples = planner.suggest_shortlist(
            observations=self.campaign.observations,
            subspace=self.param_space,
            shortlist_size=size,
        )
        return _keys(samples, self.param_space), planner

    def test_chunked_scan_matches_single_chunk_scan(self):
        # finalist_count=3 < 21 legal candidates, so the running top-k is truncated
        # across chunks and the cross-chunk merge is exercised.
        for acquisition in ("ei", "ucb"):
            with self.subTest(acquisition=acquisition):
                small, planner = self._shortlist(
                    4, acquisition_type=acquisition, chunk_size=4, finalist_count=3
                )
                whole, _ = self._shortlist(
                    4, acquisition_type=acquisition, chunk_size=10_000, finalist_count=3
                )
                self.assertEqual(small, whole)
                diagnostics = planner.planner_diagnostics()
                self.assertEqual(diagnostics["space_size"], 27)
                self.assertEqual(diagnostics["scanned_count"], 27 - len(self.observed))
                self.assertEqual(diagnostics["scan_mode"], "full")

    def test_running_top_k_merge_matches_one_shot_top_k(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=1_000)
        indices = np.arange(1_000, dtype=np.int64)
        expected = set(indices[np.argsort(-scores)[:25]].tolist())
        for dedupe in (False, True):
            with self.subTest(dedupe=dedupe):
                top_scores, top_idx = np.empty(0), np.empty(0, dtype=np.int64)
                for start in range(0, 1_000, 64):
                    stop = start + 64
                    top_scores, top_idx = _merge_top_k(
                        top_scores, top_idx, scores[start:stop], indices[start:stop], keep=25, dedupe=dedupe
                    )
                    if dedupe:  # re-deliver a chunk, as random subsampling can
                        top_scores, top_idx = _merge_top_k(
                            top_scores, top_idx, scores[start:stop], indices[start:stop], keep=25, dedupe=True
                        )
                self.assertEqual(set(top_idx.tolist()), expected)
                self.assertEqual(len(top_idx), 25)

    def test_first_pick_is_the_global_acquisition_maximum(self):
        planner = ChunkedGPPlanner(seed=3, score_dtype=torch.double)
        space = planner._encode_space(self.param_space)
        train_idx, train_y, _ = planner._training_data(self.campaign.observations, space)
        posterior = _LatentPosterior(planner._fit(space, train_idx, train_y), torch.double)
        legal = np.setdiff1d(np.arange(space["size"]), train_idx)
        mean, var = posterior.mean_var(planner._features(planner._decode(legal, space), space["blocks"]))
        brute_force_best = int(legal[int(torch.argmax(planner._acquisition(mean, var, posterior.best_f)))])
        keys, _ = self._shortlist(1, chunk_size=4, finalist_count=2)
        self.assertEqual(keys[0], tuple(planner._candidate(brute_force_best, space).values()))

    def test_latent_posterior_matches_gpytorch_prediction(self):
        planner = ChunkedGPPlanner(seed=3)
        space = planner._encode_space(self.param_space)
        train_idx, train_y, _ = planner._training_data(self.campaign.observations, space)
        model = planner._fit(space, train_idx, train_y)
        x = planner._features(planner._decode(np.arange(space["size"]), space), space["blocks"])
        with torch.no_grad():
            expected = model(x)
            posterior = _LatentPosterior(model, torch.double)
            mean, var = posterior.mean_var(x)
            joint_mean, cov = posterior.mean_cov(x)
            mean32, var32 = _LatentPosterior(model, torch.float32).mean_var(x.float())
        torch.testing.assert_close(mean, expected.mean, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(var, expected.variance, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(joint_mean, expected.mean, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(cov, expected.covariance_matrix, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(mean32.double(), expected.mean, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(var32.double(), expected.variance, rtol=1e-3, atol=1e-4)

    def test_spaces_over_the_scan_limit_use_a_labelled_subsample(self):
        keys, planner = self._shortlist(3, chunk_size=4, max_scan_size=10)
        diagnostics = planner.planner_diagnostics()
        self.assertEqual(diagnostics["scan_mode"], "subsample")
        self.assertTrue(any("scan limit" in note for note in diagnostics["warnings"]))
        self.assertEqual(len(set(keys)), len(keys))
        self.assertFalse(self.observed.intersection(keys))

    def test_batch_picks_differ_in_at_least_min_changed_variables(self):
        keys, planner = self._shortlist(4, min_changed_variables=2)
        self.assertFalse(planner.planner_diagnostics()["diversity_relaxed"])
        for i, first in enumerate(keys):
            for second in keys[i + 1 :]:
                self.assertGreaterEqual(sum(a != b for a, b in zip(first, second)), 2)

    def test_shortlist_is_unique_skips_observed_and_caps_at_remaining(self):
        keys, _ = self._shortlist(50, chunk_size=5)
        self.assertEqual(len(keys), 27 - len(self.observed))
        self.assertEqual(len(set(keys)), len(keys))
        self.assertFalse(self.observed.intersection(keys))

    def test_exclusion_constraint_is_applied_per_chunk(self):
        top, _ = self._shortlist(1, chunk_size=5)
        constraint = lab_service._exclude_candidate_keys(
            self.design.variable_names,
            {tuple(value.lower() for value in top[0])},
            design_space=self.design,
        )
        keys, _ = self._shortlist(3, chunk_size=5, known_constraints=[constraint])
        self.assertNotIn(top[0], keys)

    def test_exclusion_keys_are_matched_by_variable_name(self):
        top, _ = self._shortlist(1)
        names = self.design.variable_names
        reordered = list(reversed(names))
        constraint = ExclusionConstraint(
            variable_names=reordered,
            excluded_keys={tuple(reversed(top[0]))},
            normalize_value=str,
            check=lambda _values: True,
        )
        keys, _ = self._shortlist(3, known_constraints=[constraint])
        self.assertNotIn(top[0], keys)

        mismatched = ExclusionConstraint(
            variable_names=names[:2],
            excluded_keys=set(),
            normalize_value=str,
            check=lambda _values: True,
        )
        with self.assertRaisesRegex(ValueError, "matched by variable name"):
            self._shortlist(2, known_constraints=[mismatched])

    def test_generic_constraints_filter_the_candidate_pool(self):
        keys, planner = self._shortlist(4, known_constraints=[lambda candidate: candidate["Catalyst"] != "cat_b"])
        self.assertTrue(keys)
        self.assertTrue(all(key[0] != "cat_b" for key in keys))
        self.assertEqual(planner.planner_diagnostics()["post_filter_constraint_count"], 1)

    def test_cold_start_uses_a_random_initial_design(self):
        for observation_count in (0, 1):
            with self.subTest(observation_count=observation_count), tempfile.TemporaryDirectory() as tmp:
                service = LabBOService(projects_root=tmp)
                _create_small_project(
                    service,
                    "cold",
                    planner_name="chunked_gp",
                    allow_random_fallback=False,
                    observation_count=observation_count,
                )
                batch = service.ask("cold")
                self.assertEqual(batch["planner_fallback"], {"used": False})
                self.assertTrue(any("initial design" in note for note in batch["planner_warnings"]))
                keys = [tuple(item["candidate"].values()) for item in batch["recommendations"]]
                self.assertEqual(len(keys), 2)
                self.assertEqual(len(set(keys)), 2)
                for item in batch["recommendations"]:
                    self.assertEqual(item["planner_name"], "chunked_gp")
                    self.assertTrue(item["planner_warnings"])

    def test_uses_descriptor_encoding_when_available(self):
        records = []
        for name, values in (
            ("Catalyst", (("cat_a", 1.0, 0.2), ("cat_b", 2.0, 0.1), ("cat_c", 3.5, 0.9))),
            ("Solvent", (("dce", 10.4, 0.0), ("meoh", 32.7, 1.0), ("thf", 7.6, 0.5))),
            ("Base", (("k2co3", 10.3, 1.0), ("cs2co3", 10.0, 2.0), ("et3n", 10.8, 0.0))),
        ):
            records.extend(
                {
                    "variable": name,
                    "type": "categorical",
                    "value": value,
                    "descriptor__d1": d1,
                    "descriptor__d2": d2,
                }
                for value, d1, d2 in values
            )
        design = DesignSpace.from_long_records(records)
        param_space = design.param_space(include_descriptors=True)
        campaign = lab_service._campaign_from_observations(
            observations=self.project.load_observations(),
            design_space=design,
            param_space=param_space,
            value_space=lab_service._value_space("yield"),
            variable_names=design.variable_names,
            objective_name="yield",
        )
        planner = ChunkedGPPlanner(use_descriptors=True)
        planner.suggest_shortlist(observations=campaign.observations, subspace=param_space, shortlist_size=2)
        diagnostics = planner.planner_diagnostics()
        self.assertEqual(set(diagnostics["encodings"].values()), {"descriptor"})
        self.assertEqual(diagnostics["feature_dim"], 6)

    def test_service_ask_with_chunked_gp(self):
        for acquisition in ("ei", "ucb"):
            with self.subTest(acquisition=acquisition), tempfile.TemporaryDirectory() as tmp:
                service = LabBOService(projects_root=tmp)
                project = _create_small_project(
                    service, "chunked_ask", planner_name="chunked_gp", acquisition_function=acquisition
                )
                first = service.ask("chunked_ask")
                second = service.ask("chunked_ask")
                self.assertEqual(first["planner_fallback"], {"used": False})
                for item in first["recommendations"] + second["recommendations"]:
                    self.assertEqual(item["planner_name"], "chunked_gp")
                first_keys = {tuple(item["candidate"][name] for name in ("Catalyst", "Solvent", "Base")) for item in first["recommendations"]}
                second_keys = {tuple(item["candidate"][name] for name in ("Catalyst", "Solvent", "Base")) for item in second["recommendations"]}
                self.assertFalse(first_keys & second_keys, "pending recommendations must be excluded")
                trace = project.trace_path(first["round_id"]).read_text(encoding="utf-8")
                self.assertIn('"acquisition_name": "%s"' % acquisition, trace)


if __name__ == "__main__":
    unittest.main()

