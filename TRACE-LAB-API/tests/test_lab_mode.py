import collections
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
import torch

from chem_agent_bo.bo.atlas_bo import AtlasBOTool
from chem_agent_bo.bo.base import ExclusionConstraint
from chem_agent_bo.bo.chunked_gp import ChunkedGPPlanner, _LatentPosterior, _merge_top_k
from chem_agent_bo.bo.random_bo import RandomPlanner
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

    def test_agent_config_path_resolves_against_project_folder_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            (project_dir / "agent_bo.yaml").write_text("runtime: {}\n", encoding="utf-8")
            self.assertEqual(
                lab_service._resolve_agent_config_path("agent_bo.yaml", project_dir=project_dir),
                project_dir / "agent_bo.yaml",
            )
            self.assertEqual(
                lab_service._resolve_agent_config_path("configs/agent_bo.yaml", project_dir=project_dir),
                PROJECT_ROOT / "configs/agent_bo.yaml",
            )
            with self.assertRaisesRegex(FileNotFoundError, "missing.yaml"):
                lab_service._resolve_agent_config_path("missing.yaml", project_dir=project_dir)

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
        self.assertNotIn("temperature", captured)  # unset by default: gpt-6-luna rejects it

        captured.clear()
        config.runtime.temperature = 0.2
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), mock.patch.object(
            DecisionEngine, "_build_agent_bundle", side_effect=capture
        ):
            lab_service._build_decision_engine(config)
        self.assertEqual(captured["temperature"], 0.2)

    def test_repo_agent_config_sends_no_temperature(self):
        config = lab_service._load_agentic_config("configs/agent_bo.yaml", project_dir=PROJECT_ROOT)
        self.assertIsNone(config.runtime.temperature)

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
        brute_force_best = int(legal[int(torch.argmax(planner._acquisition(mean, var, posterior.best_f, "ei")))])
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

    def test_ei_ucb_merges_both_batches_and_ranks_by_predicted_mean(self):
        size = 4
        ei_keys, _ = self._shortlist(size, acquisition_type="ei")
        ucb_keys, _ = self._shortlist(size, acquisition_type="ucb")
        keys, planner = self._shortlist(size, acquisition_type="ei_ucb")
        diagnostics = planner.planner_diagnostics()
        self.assertEqual(diagnostics["acquisition_name"], "ei_ucb")
        self.assertEqual(diagnostics["selection_mode"], "gp_ei_ucb_merge_rank_by_mean")

        union = list(dict.fromkeys(ei_keys + ucb_keys))
        space = planner._encode_space(self.param_space)
        train_idx, train_y, _ = planner._training_data(self.campaign.observations, space)
        posterior = _LatentPosterior(planner._fit(space, train_idx, train_y), torch.double)
        index_of = {
            key: planner._index_of([space["lookup"][v][value] for v, value in enumerate(key)], space)
            for key in union
        }
        indices = np.asarray([index_of[key] for key in union], dtype=np.int64)
        means = posterior.mean_cov(planner._features(planner._decode(indices, space), space["blocks"]))[0]
        mean_of = dict(zip(union, means.tolist()))
        expected = sorted(union, key=lambda key: -mean_of[key])[:size]
        self.assertEqual(keys, expected)

        expected_tags = [
            "+".join(kind for kind, picks in (("ei", ei_keys), ("ucb", ucb_keys)) if key in picks)
            for key in keys
        ]
        self.assertEqual(diagnostics["pick_acquisitions"], expected_tags)

    def test_single_acquisition_picks_are_tagged(self):
        _, planner = self._shortlist(3, acquisition_type="ucb")
        self.assertEqual(planner.planner_diagnostics()["pick_acquisitions"], ["ucb"] * 3)

    def test_atlas_rejects_ei_ucb(self):
        with self.assertRaisesRegex(ValueError, "atlas planner does not support acquisition function `ei_ucb`"):
            AtlasBOTool(acquisition_type="ei_ucb")

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
                self.assertEqual({item["proposed_by"] for item in first["recommendations"]}, {acquisition})

    def test_service_ask_with_ei_ucb_tags_each_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = LabBOService(projects_root=tmp)
            _create_small_project(
                service, "mixed", planner_name="chunked_gp", acquisition_function="ei_ucb", batch_size=4
            )
            batch = service.ask("mixed")
            self.assertEqual(batch["planner_fallback"], {"used": False})
            tags = [item["proposed_by"] for item in batch["recommendations"]]
            self.assertEqual(len(tags), 4)
            self.assertTrue(set(tags) <= {"ei", "ucb", "ei+ucb"})
            self.assertTrue(all(tags))


class _ScriptedReflectionRuntime:
    """Stand-in agentic runtime: reflection fails for listed candidates, succeeds otherwise."""

    def __init__(self, failing_catalysts: set[str]) -> None:
        self.failing_catalysts = failing_catalysts

    def reflect_after_result(self, *, candidate, result, **_kwargs):
        if candidate.get("Catalyst") in self.failing_catalysts:
            raise RuntimeError("Error code: 502 - GR_UPSTREAM_REJECTED")
        return {"insight": f"result {result} noted"}


class ReflectionErrorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.service = LabBOService(projects_root=self._tmp.name)
        self.project = _create_small_project(
            self.service, "reflect", planner_name="random", controller_mode="bo_only", batch_size=2
        )
        self.service.ask("reflect")
        config = self.project.load_config()
        config.controller_mode = "agentic"  # reflection only runs in agentic mode
        self.project.write_config(config)
        self.recs = self.project.load_batches()[-1].recommendations

    def tearDown(self):
        self._tmp.cleanup()

    def _runtime(self, failing_catalysts):
        runtime = _ScriptedReflectionRuntime(failing_catalysts)
        return mock.patch.multiple(
            lab_service,
            _build_decision_engine=mock.MagicMock(return_value=None),
            _build_lab_controller_runtime=mock.MagicMock(return_value=runtime),
        )

    def _results(self):
        return [
            {"recommendation_id": rec["recommendation_id"], "yield": "12", "status": "completed"}
            for rec in self.recs
        ]

    def test_partial_reflection_failure_is_reported_and_kept_on_the_recommendation(self):
        failing = self.recs[0]["candidate"]["Catalyst"]
        if failing == self.recs[1]["candidate"]["Catalyst"]:
            self.skipTest("both random picks share a catalyst")
        with self._runtime({failing}):
            told = self.service.tell("reflect", results=self._results())
        self.assertEqual(told["reflection_status"], "partial")
        self.assertEqual(
            [item["recommendation_id"] for item in told["reflection_errors"]],
            [self.recs[0]["recommendation_id"]],
        )
        self.assertIn("GR_UPSTREAM_REJECTED", told["reflection_errors"][0]["error"])
        self.assertEqual(len(self.project.load_observations().rows), 8)  # results saved anyway

        first, second = self.project.load_batches()[-1].recommendations
        self.assertIn("GR_UPSTREAM_REJECTED", first["reflection_error"])
        self.assertNotIn("reflection", first)
        self.assertNotIn("reflection_error", second)
        self.assertEqual(second["reflection"], {"insight": "result 12.0 noted"})

        # A later retry that succeeds clears the error.
        with self._runtime(set()):
            retried = self.service.reflect_completed_recommendations(
                "reflect", recommendation_ids=[first["recommendation_id"]]
            )
        self.assertEqual(retried["reflection_errors"], [])
        first = self.project.load_batches()[-1].recommendations[0]
        self.assertNotIn("reflection_error", first)
        self.assertEqual(first["reflection"], {"insight": "result 12.0 noted"})

    def test_all_reflections_failing_reports_failed(self):
        catalysts = {rec["candidate"]["Catalyst"] for rec in self.recs}
        with self._runtime(catalysts):
            told = self.service.tell("reflect", results=self._results())
        self.assertEqual(told["reflection_status"], "failed")
        self.assertEqual(len(told["reflection_errors"]), 2)


class ProjectConfigApiTests(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:  # pragma: no cover
            self.skipTest("FastAPI test client is not installed.")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.service = LabBOService(projects_root=self.root)
        self.project = _create_small_project(self.service, "cfg", planner_name="random", batch_size=2)
        self.service.ask("cfg")
        self.client = TestClient(create_app(self.root))

    def tearDown(self):
        self._tmp.cleanup()

    def _patch(self, payload):
        return self.client.patch("/api/projects/cfg/config", json=payload)

    def test_changes_settings_backs_up_and_keeps_project_data(self):
        design_before = self.project.design_space_path.read_bytes()
        observations_before = self.project.observations_path.read_bytes()
        batches_before = [batch.to_dict() for batch in self.project.load_batches()]
        options = {"chunked_gp": {"min_changed_variables": 1, "finalist_count": 50}}

        response = self._patch(
            {"planner_name": "chunked_gp", "acquisition_function": "ucb", "planner_options": options}
        )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(set(body["changed"]), {"planner_name", "acquisition_function", "planner_options"})
        config = self.project.load_config()
        self.assertEqual((config.planner_name, config.acquisition_function), ("chunked_gp", "ucb"))
        self.assertEqual(config.planner_options, options)
        self.assertIn("planner_name: random", Path(body["backup_path"]).read_text(encoding="utf-8"))
        self.assertEqual(self.project.design_space_path.read_bytes(), design_before)
        self.assertEqual(self.project.observations_path.read_bytes(), observations_before)
        self.assertEqual([batch.to_dict() for batch in self.project.load_batches()], batches_before)

    def test_unchanged_values_write_nothing(self):
        response = self._patch({"batch_size": 2})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["changed"], {})
        self.assertFalse((self.root / "_backups").exists())

    def test_planner_options_reach_chunked_gp(self):
        self._patch(
            {
                "planner_name": "chunked_gp",
                "planner_options": {"chunked_gp": {"min_changed_variables": 1, "max_scan_size": 5}},
            }
        )
        batch = self.service.ask("cfg")
        trace = json.loads(self.project.trace_path(batch["round_id"]).read_text(encoding="utf-8").splitlines()[0])
        diagnostics = trace["planner_diagnostics"]
        self.assertEqual(diagnostics["min_changed_variables"], 1)
        self.assertEqual(diagnostics["scan_mode"], "subsample")  # 19 legal combinations > max_scan_size 5

    def test_planner_options_merge_and_null_removes(self):
        self._patch({"planner_options": {"chunked_gp": {"min_changed_variables": 3, "finalist_count": 50}}})
        self._patch({"planner_options": {"chunked_gp": {"finalist_count": None, "max_scan_size": 1000}}})
        self.assertEqual(
            self.project.load_config().planner_options,
            {"chunked_gp": {"min_changed_variables": 3, "max_scan_size": 1000}},
        )
        self._patch({"planner_options": {"chunked_gp": None}})
        self.assertEqual(self.project.load_config().planner_options, {})

    def test_invalid_changes_are_rejected_without_writing(self):
        before = self.project.config_path.read_bytes()
        cases = {
            "locked field": ({"objective_name": "ee"}, "can't be changed"),
            "unknown field": ({"colour": "blue"}, "Unknown or non-editable"),
            "acquisition for planner": (
                {"planner_name": "atlas", "acquisition_function": "ei_ucb"},
                "does not support acquisition function `ei_ucb`",
            ),
            "mode typo": ({"controller_mode": "bo-only"}, "controller_mode must be"),
            "option value": ({"planner_options": {"chunked_gp": {"min_changed_variables": 0}}}, "positive integer"),
            "option name": ({"planner_options": {"chunked_gp": {"min_changed": 2}}}, "Unknown option"),
            "planner without options": ({"planner_options": {"atlas": {"x": 1}}}, "no configurable options"),
            "batch size type": ({"batch_size": "5"}, "must be an integer"),
            "missing agent config": (
                {"controller_mode": "agentic", "agent_config_path": "missing.yaml"},
                "not found",
            ),
            "nothing to change": ({}, "at least one setting"),
        }
        for label, (payload, message) in cases.items():
            with self.subTest(label):
                response = self._patch(payload)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn(message, response.json()["detail"])
        self.assertEqual(self.project.config_path.read_bytes(), before)
        self.assertFalse((self.root / "_backups").exists())

    def test_changes_and_asks_are_refused_while_an_ask_runs(self):
        with ConcurrentAskTests._slow_planners({"slow_ask": 0.8}):
            running = threading.Thread(target=self.service.ask, args=("cfg",), name="slow_ask")
            running.start()
            time.sleep(0.2)
            config_response = self._patch({"batch_size": 3})
            ask_response = self.client.post("/api/projects/cfg/ask", json={})
            running.join()
        self.assertEqual(config_response.status_code, 409, config_response.text)
        self.assertIn("ask is running", config_response.json()["detail"])
        self.assertEqual(ask_response.status_code, 409, ask_response.text)
        self.assertEqual(self._patch({"batch_size": 3}).status_code, 200)

    def test_overwrite_refuses_a_project_with_data(self):
        payload = {
            "project_id": "cfg",
            "config": {"project_id": "cfg"},
            "design_records": _small_design_space().to_long_records(),
            "overwrite": True,
        }
        response = self.client.post("/api/projects", json=payload)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("Refusing to overwrite", response.json()["detail"])
        self.assertEqual(len(self.project.load_observations().rows), 6)

        _create_small_project(self.service, "fresh", observation_count=0)
        fresh = dict(payload, project_id="fresh", config={"project_id": "fresh"})
        self.assertEqual(self.client.post("/api/projects", json=fresh).status_code, 200)

    def test_invalid_hand_edited_options_fail_the_ask(self):
        config = self.project.load_config()
        config.planner_name = "chunked_gp"
        config.planner_options = {"chunked_gp": {"finalist_count": 0}}
        self.project.write_config(config)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.service.ask("cfg")


class _SlowPrintingPlanner:
    """Prints steadily while it works (as Atlas does), then returns random candidates."""

    def __init__(self, tag: str, seconds: float) -> None:
        self.tag = tag
        self.seconds = seconds
        self.inner = RandomPlanner(seed=1)

    def suggest_shortlist(self, **kwargs):
        end = time.time() + self.seconds
        while time.time() < end:
            print(f"planner progress {self.tag}")
            time.sleep(0.02)
        return self.inner.suggest_shortlist(**kwargs)

    def planner_diagnostics(self):
        return {"planner_name": "atlas"}


class ConcurrentAskTests(unittest.TestCase):
    """The web server runs requests in a thread pool, so asks can overlap."""

    @staticmethod
    def _slow_planners(seconds_by_thread: dict[str, float]):
        def build(_name, **_kwargs):
            tag = threading.current_thread().name
            return _SlowPrintingPlanner(tag, seconds_by_thread[tag])

        return mock.patch.object(lab_service, "build_planner", side_effect=build)

    def test_overlapping_asks_keep_output_in_their_own_logs(self):
        stdout_before, stderr_before = sys.stdout, sys.stderr
        with tempfile.TemporaryDirectory() as tmp:
            service = LabBOService(projects_root=tmp)
            for project_id in ("p1", "p2"):
                _create_small_project(service, project_id, batch_size=1)
            outcomes: dict = {}

            def ask(project_id):
                try:
                    outcomes[project_id] = service.ask(project_id)["planner_fallback"]
                except Exception as exc:  # noqa: BLE001
                    outcomes[project_id] = f"{type(exc).__name__}: {exc}"

            # p1 starts first and finishes first -- the order that used to leave the
            # process-wide stdout pointing at p1's closed log file. The old code only
            # crashed a planner on the second overlap, so overlap twice.
            for _ in range(2):
                with self._slow_planners({"p1": 0.6, "p2": 1.0}):
                    first = threading.Thread(target=ask, args=("p1",), name="p1")
                    second = threading.Thread(target=ask, args=("p2",), name="p2")
                    first.start()
                    time.sleep(0.2)
                    second.start()
                    first.join()
                    second.join()
                self.assertEqual(outcomes, {"p1": {"used": False}, "p2": {"used": False}})
                self.assertIs(sys.stdout, stdout_before)
                self.assertIs(sys.stderr, stderr_before)
                self.assertFalse(sys.stdout.closed)

            for project_id, other in (("p1", "p2"), ("p2", "p1")):
                log = (Path(tmp) / project_id / "logs" / "third_party.log").read_text(encoding="utf-8")
                self.assertIn(f"planner progress {project_id}", log)
                self.assertNotIn(f"planner progress {other}", log)

    def test_second_ask_on_same_project_is_rejected_while_one_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = LabBOService(projects_root=tmp)
            project = _create_small_project(service, "p1", batch_size=1)
            with self._slow_planners({"first_ask": 0.8, "MainThread": 0.0}):
                first = threading.Thread(target=service.ask, args=("p1",), name="first_ask")
                first.start()
                time.sleep(0.2)
                with self.assertRaisesRegex(RuntimeError, "already running for project `p1`"):
                    service.ask("p1")
                first.join()
                self.assertEqual([batch.round_id for batch in project.load_batches()], ["round_001"])
                service.ask("p1")  # the lock is released once the first ask finishes
            self.assertEqual(
                [batch.round_id for batch in project.load_batches()], ["round_001", "round_002"]
            )


if __name__ == "__main__":
    unittest.main()

