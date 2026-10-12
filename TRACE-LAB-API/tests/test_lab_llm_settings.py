"""Model settings with a caller's own model, and the retry without a rejected parameter. A project's agent config
is used as it is (FounDryClaw's import_project rewrites it to the user's model and endpoint); only a parameter the
model rejects with a 400 is left out."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from chem_agent_bo.config import load_agentic_bo_config
from chem_agent_bo.lab import service as lab_service
from chem_agent_bo.lab.project_check import check_project
from chem_agent_bo.lab.service import LabBOService, _build_decision_engine, _llm_report

from test_lab_mode import _create_small_project

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REJECTED = RuntimeError(
    "Error code: 400 - {'error': {'message': '`temperature` is not supported for this model request. Omit it and "
    "retry.', 'type': 'invalid_request_error', 'code': 'GR_UNSUPPORTED_TEMPERATURE', 'param': 'temperature'}}"
)


def _group_config():
    config = load_agentic_bo_config(PROJECT_ROOT / "configs" / "agent_bo.yaml")
    config.runtime.model_name = "gpt-5.6-terra"
    config.runtime.temperature = 0.0
    return config


class CallersModelTests(unittest.TestCase):
    def test_callers_model_gets_the_configs_other_settings(self):
        # No request-time adjustment any more: the file's temperature is sent; a model that rejects it is retried
        # without it (RejectedParameterTests).
        engine = _build_decision_engine(_group_config(), {"api_key": "k", "model": "gpt-6-luna"})
        self.assertEqual(engine.model_name, "gpt-6-luna")
        self.assertEqual(engine._model_kwargs.get("temperature"), 0.0)
        report = _llm_report(engine)
        self.assertEqual(report["llm_model"], "gpt-6-luna")
        self.assertEqual(report["llm_adjustments"], [])

    def test_configured_model_keeps_its_temperature(self):
        engine = _build_decision_engine(_group_config(), {"api_key": "k"})
        self.assertEqual(engine.model_name, "gpt-5.6-terra")
        self.assertEqual(engine._model_kwargs.get("temperature"), 0.0)
        self.assertEqual(_llm_report(engine)["llm_adjustments"], [])

    def test_no_engine_no_report(self):
        self.assertEqual(_llm_report(None), {})


class RejectedParameterTests(unittest.TestCase):
    def setUp(self):
        self.engine = _build_decision_engine(_group_config(), {"api_key": "k"})

    def test_parameter_is_found_in_the_error(self):
        find = type(self.engine)._rejected_param
        self.assertEqual(find(REJECTED), "temperature")
        self.assertEqual(find(RuntimeError("Error code: 400 - Unsupported parameter: 'reasoning_effort'")), "reasoning_effort")
        self.assertIsNone(find(RuntimeError("Error code: 400 - context length exceeded")))
        self.assertIsNone(find(RuntimeError("Error code: 500 - `temperature` is not supported")))
        self.assertIsNone(find(None))

    def test_retry_without_the_parameter(self):
        answer = object()
        calls = []

        def fake_invoke(agent, message_text, **kwargs):
            calls.append(dict(self.engine._model_kwargs))
            return (None, REJECTED) if len(calls) == 1 else (answer, None)

        with mock.patch.object(self.engine, "_invoke_with_retries", side_effect=fake_invoke):
            result = self.engine._invoke_agent_structured(
                self.engine.diagnosis_agent, "prompt", object(), call_name="stagnation_diagnosis")
        self.assertIs(result, answer)
        self.assertEqual(len(calls), 2)
        self.assertIn("temperature", calls[0])
        self.assertNotIn("temperature", calls[1])
        self.assertEqual(self.engine.dropped_params, ["temperature"])
        self.assertTrue(any("temperature" in line for line in _llm_report(self.engine)["llm_adjustments"]))

    def test_other_errors_still_fail_at_once(self):
        other = RuntimeError("Error code: 400 - {'error': {'message': 'Invalid schema'}}")
        with mock.patch.object(self.engine, "_invoke_with_retries", return_value=(None, other)) as invoke:
            with self.assertRaises(RuntimeError):
                self.engine._invoke_agent_structured(self.engine.diagnosis_agent, "prompt", object(),
                                                     call_name="stagnation_diagnosis")
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(self.engine.dropped_params, [])


class ImportNoteTests(unittest.TestCase):
    def test_project_config_for_another_model_is_a_fact(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = LabBOService(projects_root=Path(tmp))
            project = _create_small_project(service, "group_project", controller_mode="agentic")
            agent = yaml.safe_load((PROJECT_ROOT / "configs" / "agent_bo.yaml").read_text(encoding="utf-8"))
            agent["runtime"].update({"model_name": "gpt-5.6-terra", "temperature": 0.0,
                                     "api_base": "https://endpoint.greatrouter.com/v1"})
            (project.project_dir / "agent_bo.yaml").write_text(yaml.safe_dump(agent), encoding="utf-8")
            config = yaml.safe_load((project.project_dir / "project.yaml").read_text(encoding="utf-8"))
            config["agent_config_path"] = "agent_bo.yaml"
            (project.project_dir / "project.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            report = check_project(project.project_dir)
            note = report["facts"].get("agent_llm_note", "")
            self.assertIn("gpt-5.6-terra", note)
            self.assertIn("temperature 0.0", note)
            self.assertNotIn("not sent", note)
            self.assertNotIn("agent_llm_note", " ".join(str(item) for item in report["warnings"]))

    def test_default_config_has_no_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = LabBOService(projects_root=Path(tmp))
            project = _create_small_project(service, "plain_project")
            self.assertNotIn("agent_llm_note", check_project(project.project_dir)["facts"])


if __name__ == "__main__":
    unittest.main()
