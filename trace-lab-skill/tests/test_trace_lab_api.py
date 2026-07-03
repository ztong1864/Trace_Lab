import importlib.util
from argparse import Namespace
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "trace_lab_api.py"
SPEC = importlib.util.spec_from_file_location("trace_lab_api", SCRIPT_PATH)
trace_lab_api = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trace_lab_api)


def test_ask_defaults_to_agentic_controller(monkeypatch):
    captured = {}

    def fake_request_json(base_url, method, path, payload=None):
        captured["payload"] = payload
        return {"ok": True}

    monkeypatch.setattr(trace_lab_api, "request_json", fake_request_json)
    args = Namespace(
        project_id="demo",
        batch_size=None,
        planner_name=None,
        controller_mode=None,
        agent_config_path=None,
        planner_use_descriptors=None,
    )

    trace_lab_api.ask("http://127.0.0.1:8788", args)

    assert captured["payload"]["controller_mode"] == "agentic"
