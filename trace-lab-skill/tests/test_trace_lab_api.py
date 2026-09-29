import importlib.util
from argparse import Namespace
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "trace_lab_api.py"
SPEC = importlib.util.spec_from_file_location("trace_lab_api", SCRIPT_PATH)
trace_lab_api = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(trace_lab_api)


def test_ask_uses_the_project_controller_unless_one_is_given(monkeypatch):
    captured = []

    def fake_request_json(base_url, method, path, payload=None):
        captured.append(payload)
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
    trace_lab_api.ask("http://127.0.0.1:8788", Namespace(**{**vars(args), "controller_mode": "bo_only"}))

    # Not sending controller_mode lets a project configured as bo_only stay bo_only.
    assert "controller_mode" not in captured[0]
    assert captured[1]["controller_mode"] == "bo_only"


def _create_args(tmp_path, **overrides):
    design = tmp_path / "design.csv"
    design.write_text("variable,type,value\nCatalyst,categorical,cat_a\n", encoding="utf-8")
    values = dict(
        project_id="demo",
        design_csv=str(design),
        objective_name="yield",
        planner_name="chunked_gp",
        acquisition_function=None,
        controller_mode="bo_only",
        batch_size=4,
        planner_use_descriptors=True,
        descriptor=[],
        alias=[],
        observations_csv=None,
        evidence=None,
        overwrite=False,
    )
    values.update(overrides)
    return Namespace(**values)


def test_create_payload_uploads_all_parts(tmp_path):
    table = tmp_path / "additives.csv"
    table.write_text("formula,d1\nCeCl3,1\n", encoding="utf-8")
    results = tmp_path / "results.csv"
    results.write_text("Catalyst,yield\ncat_a,12\n", encoding="utf-8")
    cards = tmp_path / "cards.csv"
    cards.write_text("card_id,summary\nc1,x\n", encoding="utf-8")
    args = _create_args(
        tmp_path,
        descriptor=[["Additive", "formula", str(table)]],
        alias=[["Additive", "CeCl3·7H2O", "CeCl3"]],
        observations_csv=str(results),
        evidence=str(cards),
        acquisition_function="ei_ucb",
    )

    payload = trace_lab_api.create_payload(args)

    assert payload["design_csv"].startswith("variable,type,value")
    assert payload["config"]["acquisition_function"] == "ei_ucb"
    assert payload["config"]["planner_use_descriptors"] is True
    assert payload["descriptor_tables"] == [
        {"variable": "Additive", "value_column": "formula", "csv": "formula,d1\nCeCl3,1\n", "aliases": {"CeCl3·7H2O": "CeCl3"}}
    ]
    assert payload["observations_csv"] == "Catalyst,yield\ncat_a,12\n"
    assert payload["evidence_csv"] == "card_id,summary\nc1,x\n"


def test_create_payload_rejects_aliases_without_a_table(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="without a --descriptor table"):
        trace_lab_api.create_payload(_create_args(tmp_path, alias=[["Solvent", "DCE", "dce"]]))


def test_check_summary_lists_errors_warnings_and_facts():
    text = trace_lab_api.format_check(
        {
            "ok": False,
            "errors": [{"file": "observations.csv", "line": 4, "message": "value `cat_z` is not in design-space options"}],
            "warnings": [{"file": "evidence_cards.jsonl", "message": "2 card(s) have mapping_status `x`"}],
            "facts": {"observation_count": 6, "pending_recommendation_count": 0},
        }
    )
    assert text.splitlines()[0] == "Check: FAILED (1 error(s), 1 warning(s))"
    assert "ERROR: observations.csv line 4: value `cat_z`" in text
    assert "WARNING: evidence_cards.jsonl: 2 card(s)" in text
    assert "Facts: observation_count=6, pending_recommendation_count=0" in text


def test_check_needs_a_handle_or_a_path():
    import pytest

    for project_id, path in (("demo", "D:/x"), (None, None)):
        with pytest.raises(ValueError, match="handle or --path"):
            trace_lab_api.dispatch(Namespace(command="check", project_id=project_id, path=path), "http://x")


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


def test_evidence_preview_calls_the_endpoint_and_summarizes_it(monkeypatch):
    preview = {
        "card_count": 3,
        "top_k": 1,
        "retrieved": [{"rank": 1, "score": 79.7, "mapping_status": "same_reaction_family", "card_id": "card_a"}],
        "not_shown": [{"rank": 2, "score": 75.8, "mapping_status": "background", "card_id": "card_b"}],
        "never_retrieved": [
            {"card_id": "card_c", "mapping_status": "same_start_end", "reason_text": "reaction_scope does not match."}
        ],
    }
    calls = []
    monkeypatch.setattr(
        trace_lab_api, "request_json", lambda base_url, method, path, payload=None: calls.append((method, path)) or preview
    )
    data = trace_lab_api.dispatch(Namespace(command="evidence-preview", project_id="demo"), "http://x")
    text = trace_lab_api.format_summary(data, command="evidence-preview")

    assert calls == [("GET", "/api/projects/demo/evidence/preview")]
    assert "3 card(s); each ask shows the top 1" in text
    assert "| 1 | 79.7 | same_reaction_family | card_a |" in text
    assert "the best of them ranks 2 with score 75.8" in text
    assert "Never shown: card_c (same_start_end): reaction_scope does not match." in text
