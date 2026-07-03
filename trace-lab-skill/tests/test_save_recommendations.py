import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "save_recommendations.py"
SPEC = importlib.util.spec_from_file_location("save_recommendations", SCRIPT_PATH)
save_recommendations = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(save_recommendations)


def test_markdown_table_expands_candidate_fields():
    rows = [
        {
            "rank": "1",
            "recommendation_id": "round_001_rec_001",
            "candidate": "Additive=MnBr2; Solvent=Benzene",
            "batch_role": "explore",
            "rationale": "Try this condition.",
            "candidate__Additive": "MnBr2",
            "candidate__Solvent": "Benzene",
        }
    ]

    table = save_recommendations.format_markdown_table(rows)

    assert "| rank | Additive | Solvent | rationale |" in table
    assert "| 1 | MnBr2 | Benzene | Try this condition. |" in table
    assert "| rank | recommendation_id | candidate |" not in table
    assert "round_001_rec_001" not in table
    assert "explore" not in table
