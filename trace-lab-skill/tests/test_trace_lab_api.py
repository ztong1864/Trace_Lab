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


def test_config_settings_nest_dotted_keys_and_parse_json_values():
    updates = trace_lab_api.parse_config_settings(
        [
            "planner_name=chunked_gp",
            "planner_options.chunked_gp.min_changed_variables=3",
            "planner_options.chunked_gp.finalist_count=null",
            "allow_random_fallback=false",
        ],
        '{"batch_size": 4}',
    )

    assert updates == {
        "batch_size": 4,
        "planner_name": "chunked_gp",
        "planner_options": {"chunked_gp": {"min_changed_variables": 3, "finalist_count": None}},
        "allow_random_fallback": False,
    }


def test_config_sends_patch_and_shows_settings_without_changes(monkeypatch):
    calls = []

    def fake_request_json(base_url, method, path, payload=None):
        calls.append((method, path, payload))
        return {"project": {"planner_name": "chunked_gp"}}

    monkeypatch.setattr(trace_lab_api, "request_json", fake_request_json)
    shown = trace_lab_api.dispatch(
        Namespace(command="config", project_id="demo", settings=[], settings_json=None), "http://x"
    )
    trace_lab_api.dispatch(
        Namespace(command="config", project_id="demo", settings=["seed=11"], settings_json=None), "http://x"
    )

    assert shown == {"planner_name": "chunked_gp"}
    assert calls == [
        ("GET", "/api/projects/demo", None),
        ("PATCH", "/api/projects/demo/config", {"seed": 11}),
    ]
