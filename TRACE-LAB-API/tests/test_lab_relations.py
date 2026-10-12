"""Relation of a candidate to the measured runs, and the check on the LLM's "exact repeat" comments (L1, round_002)."""

from types import SimpleNamespace
import unittest

from chem_agent_bo.agent.relations import claims_repeat, describe_differences, relation_sentence, relation_to_history
from chem_agent_bo.lab.service import _recommendations_from_decision
from chem_agent_bo.prompts.templates import build_semantic_assessment_prompt

VARIABLES = ["Fe loading", "TEMPO derivative", "Additive", "Solvent"]
BEST = {"Fe loading": "4 mol%", "TEMPO derivative": "AZADO", "Additive": "NaCl", "Solvent": "tBuOH"}
HISTORY = [
    {"iteration": 1, "result": 0.0, "active_variables": VARIABLES,
     "candidate": {"Fe loading": "4 mol%", "TEMPO derivative": "4-OBz-TEMPO", "Additive": "Co(OAc)2", "Solvent": "PhCl"}},
    {"iteration": 2, "result": 2.0, "active_variables": VARIABLES,
     "candidate": {"Fe loading": "5 mol%", "TEMPO derivative": "4-OBz-TEMPO", "Additive": "La(OTf)3", "Solvent": "EtOAc"}},
    {"iteration": 3, "result": 3.0, "active_variables": VARIABLES, "candidate": dict(BEST)},
]
L1_COMMENTS = [
    "However, this exactly repeats the iteration-3 best condition, already measured at 3% yield, so it adds little.",
    "The condition is chemically plausible and is the current measured best, but it exactly duplicates an observed experiment.",
    "Chemically plausible, but this is an exact repeat of the current best observation.",
]


class RelationTests(unittest.TestCase):
    def test_one_variable_probe_is_not_a_duplicate(self):
        relation = relation_to_history({**BEST, "Additive": "LiCl"}, HISTORY)
        self.assertIsNone(relation["exact_duplicate_of"])
        self.assertEqual(relation["closest_measured"]["iteration"], 3)
        self.assertEqual(describe_differences(relation), "Additive: NaCl -> LiCl")
        self.assertIn("1 variable differs: Additive: NaCl -> LiCl.", relation_sentence(relation))
        two = relation_to_history({**BEST, "Additive": "LiCl", "Solvent": "DCE"}, HISTORY)
        self.assertIn("2 variables differ: ", relation_sentence(two))

    def test_exact_duplicate_is_found(self):
        relation = relation_to_history(dict(BEST), HISTORY)
        self.assertEqual(relation["exact_duplicate_of"], {"iteration": 3, "result": 3.0})
        self.assertIn("repeat measurement of iteration 3", relation_sentence(relation))

    def test_numbers_compare_by_value_and_ties_prefer_the_best_run(self):
        history = [{"iteration": 1, "result": 1.0, "active_variables": ["T"], "candidate": {"T": "60"}},
                   {"iteration": 2, "result": 9.0, "active_variables": ["T"], "candidate": {"T": "80"}}]
        self.assertEqual(relation_to_history({"T": "60.0"}, history)["exact_duplicate_of"]["iteration"], 1)
        self.assertEqual(relation_to_history({"T": "70"}, history)["closest_measured"]["iteration"], 2)
        self.assertEqual(relation_to_history({"T": "70"}, history, goal="minimize")["closest_measured"]["iteration"], 1)

    def test_no_history(self):
        self.assertEqual(relation_to_history(BEST, []), {})
        self.assertEqual(relation_sentence({}), "")

    def test_repeat_claims(self):
        for comment in L1_COMMENTS:
            self.assertTrue(claims_repeat(comment), comment)
        self.assertFalse(claims_repeat("It is intentionally near-duplicate rather than broad coverage."))
        self.assertFalse(claims_repeat("Retains the best measured AZADO condition and changes only NaCl loading."))

    def test_prompt_carries_the_relation(self):
        relation = relation_to_history({**BEST, "Additive": "LiCl"}, HISTORY)
        prompt = build_semantic_assessment_prompt({**BEST, "Additive": "LiCl"}, {}, {}, relation=relation)
        self.assertIn("computed by code, authoritative", prompt)
        self.assertIn('"exact_duplicate_of": null', prompt)
        self.assertNotIn("computed by code", build_semantic_assessment_prompt(BEST, {}, {}, relation=None))


def _decision(candidates, comments):
    return SimpleNamespace(
        selected_candidates=candidates,
        trace_records=[{"relation_to_measured": relation_to_history(c, HISTORY), "batch_role": "probe"} for c in candidates],
        semantic_assessments=[{"soft_comment": comment, "risk_level": "low"} for comment in comments],
        verification_passes=[None] * len(candidates),
        controller_plan={"reasoning": "local probes"}, action_package={"executed_execution_action": "direct_bo_pick"},
        batch_contract={}, batch_validation_report={}, evidence_trace={}, skill_trace={}, diagnosis={},
        hypothesis_action={}, coverage_insight={}, rerank_action=None, action_capability=None,
    )


class RecommendationCheckTests(unittest.TestCase):
    def build(self, candidates, comments):
        recommendations, _ = _recommendations_from_decision(
            round_id="round_002", decision=_decision(candidates, comments), planner_name="chunked_gp",
            planner_diagnostics={}, planner_error="", created_at="2026-10-09T00:00:00+00:00")
        return recommendations

    def test_false_repeat_claim_is_flagged_and_left_out(self):
        recommendation = self.build([{**BEST, "Additive": "LiCl"}], [L1_COMMENTS[0]])[0]
        self.assertNotIn("exactly repeats", recommendation["rationale"])
        self.assertIn("Additive: NaCl -> LiCl", recommendation["rationale"])
        warnings = recommendation["semantic_assessment"]["consistency_warnings"]
        self.assertTrue(warnings and "Additive: NaCl -> LiCl" in warnings[0])
        self.assertEqual(recommendation["planner_warnings"], warnings)
        self.assertIsNone(recommendation["relation_to_measured"]["exact_duplicate_of"])

    def test_true_repeat_keeps_the_comment(self):
        recommendation = self.build([dict(BEST)], [L1_COMMENTS[2]])[0]
        self.assertIn("exact repeat", recommendation["rationale"])
        self.assertIn("repeat measurement of iteration 3", recommendation["rationale"])
        self.assertNotIn("consistency_warnings", recommendation["semantic_assessment"])

    def test_ordinary_comment_is_kept(self):
        comment = "Chemically plausible one-factor probe of the additive."
        recommendation = self.build([{**BEST, "Additive": "LiCl"}], [comment])[0]
        self.assertIn(comment, recommendation["rationale"])
        self.assertEqual(recommendation["planner_warnings"], [])


if __name__ == "__main__":
    unittest.main()
