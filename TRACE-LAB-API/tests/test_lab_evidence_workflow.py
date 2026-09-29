"""Evidence cards from papers: retrieval preview, drafting checks, review sheet, publish."""

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from chem_agent_bo.lab.api import create_app
from chem_agent_bo.lab.evidence import (
    LAB_EVIDENCE_TARGET_NODES,
    EvidenceCard,
    EvidenceStore,
    retrieval_preview,
)
from chem_agent_bo.lab.service import LabBOService

from test_lab_mode import _create_small_project


def _card(card_id: str, **fields) -> EvidenceCard:
    values = {"source": "A paper", "summary": f"summary of {card_id}", "variable_scope": ["Solvent"]}
    values.update(fields)
    return EvidenceCard(card_id=card_id, **values)


class RetrievalScreenTests(unittest.TestCase):
    """`explain` and `applicable` share one screen, so they cannot disagree."""

    variables = ["Catalyst", "Solvent", "Base"]
    scope = "oxidative lactonization of diols"

    def _store(self) -> EvidenceStore:
        return EvidenceStore(
            [
                _card("good_high", confidence="high", mapping_status="same_reaction_family"),
                _card("good_low", confidence="low", mapping_status="same_reaction_family"),
                _card("background_any", mapping_status="background", variable_scope=["Temperature"]),
                _card("other_variable", mapping_status="variable_level", variable_scope=["Temperature"]),
                _card("hidden", mapping_status="out_of_scope"),
                _card("blocked_use", allowed_use="do_not_use"),
                _card("oracle", leakage_risk="candidate_lookup"),
                _card("wrong_node", target_nodes=["some_other_node"]),
                _card(
                    "direct_wrong_scope",
                    mapping_status="direct",
                    reaction_scope="esterification of phenols",
                ),
                _card(
                    "direct_right_scope",
                    mapping_status="direct",
                    reaction_scope="oxidative lactonization",
                ),
            ]
        )

    def _explain(self, top_k=100):
        return {
            item["card_id"]: item
            for item in self._store().explain(
                variables=self.variables,
                reaction_scope=self.scope,
                target_nodes=list(LAB_EVIDENCE_TARGET_NODES),
                max_items=top_k,
            )
        }

    def test_each_screen_gives_its_own_reason(self):
        got = self._explain()
        reasons = {card_id: item["reason"] for card_id, item in got.items()}
        self.assertEqual(reasons["hidden"], "out_of_scope")
        self.assertEqual(reasons["blocked_use"], "allowed_use_blocked")
        self.assertEqual(reasons["oracle"], "leakage_risk_blocked")
        self.assertEqual(reasons["other_variable"], "no_variable_overlap")
        self.assertEqual(reasons["wrong_node"], "target_node_mismatch")
        self.assertEqual(reasons["direct_wrong_scope"], "reaction_scope_mismatch")
        for card_id in ("good_high", "good_low", "background_any", "direct_right_scope"):
            self.assertEqual(reasons[card_id], "", card_id)
            self.assertIsNotNone(got[card_id]["rank"])
        self.assertIn("reaction_scope", got["direct_wrong_scope"]["reason_text"])

    def test_explain_agrees_with_applicable_at_every_cut(self):
        store = self._store()
        for top_k in range(0, 8):
            with self.subTest(top_k=top_k):
                applicable = [
                    card.card_id
                    for card in store.applicable(
                        variables=self.variables,
                        reaction_scope=self.scope,
                        target_nodes=list(LAB_EVIDENCE_TARGET_NODES),
                        max_items=top_k,
                    )
                ]
                explained = sorted(
                    (
                        item
                        for item in store.explain(
                            variables=self.variables,
                            reaction_scope=self.scope,
                            target_nodes=list(LAB_EVIDENCE_TARGET_NODES),
                            max_items=top_k,
                        )
                        if item["retrieved"]
                    ),
                    key=lambda item: item["rank"],
                )
                self.assertEqual(applicable, [item["card_id"] for item in explained])

    def test_preview_splits_shown_not_shown_and_never(self):
        preview = retrieval_preview(
            self._store(), variables=self.variables, reaction_scope=self.scope, top_k=2
        )
        self.assertEqual(len(preview["retrieved"]), 2)
        self.assertEqual([item["rank"] for item in preview["retrieved"]], [1, 2])
        self.assertTrue(all(item["rank"] > 2 for item in preview["not_shown"]))
        self.assertEqual(
            {item["card_id"] for item in preview["never_retrieved"]},
            {"hidden", "blocked_use", "oracle", "other_variable", "wrong_node", "direct_wrong_scope"},
        )
        self.assertEqual(
            len(preview["retrieved"]) + len(preview["not_shown"]) + len(preview["never_retrieved"]),
            preview["card_count"],
        )


class EvidencePreviewProjectTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "root"
        self.service = LabBOService(projects_root=self.root)
        self.project = _create_small_project(
            self.service, "ev", reaction_scope="oxidative lactonization of diols"
        )
        self.client = TestClient(create_app(self.root))
        cards = [
            {"card_id": "ok_1", "summary": "a", "variable_scope": ["Solvent"], "confidence": "0.9"},
            {"card_id": "ok_2", "summary": "b", "variable_scope": ["Catalyst"], "confidence": "0.5"},
            {
                "card_id": "stranded",
                "summary": "c",
                "variable_scope": ["Solvent"],
                "mapping_status": "same_start_end",
                "reaction_scope": "esterification of phenols",
            },
            {"card_id": "hidden_on_purpose", "summary": "d", "mapping_status": "out_of_scope"},
        ]
        response = self.client.post("/api/projects/ev/evidence/import", json={"cards": cards})
        self.assertEqual(response.status_code, 200, response.text)

    def test_endpoint_lists_what_the_controller_gets(self):
        response = self.client.get("/api/projects/ev/evidence/preview")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["handle"], "ev")
        self.assertEqual(body["target_nodes"], list(LAB_EVIDENCE_TARGET_NODES))
        self.assertEqual([item["card_id"] for item in body["retrieved"]], ["ok_1", "ok_2"])
        never = {item["card_id"]: item for item in body["never_retrieved"]}
        self.assertEqual(never["stranded"]["reason"], "reaction_scope_mismatch")
        self.assertEqual(never["stranded"]["reaction_scope"], "esterification of phenols")
        self.assertEqual(never["hidden_on_purpose"]["reason"], "out_of_scope")

    def test_unknown_project_is_a_404(self):
        self.assertEqual(self.client.get("/api/projects/nope/evidence/preview").status_code, 404)

    def test_check_warns_about_stranded_cards_but_not_deliberately_hidden_ones(self):
        report = self.service.check_project("ev")
        messages = " | ".join(item["message"] for item in report["warnings"])
        self.assertIn("1 card(s) can never be shown", messages)
        self.assertIn("stranded", messages)
        self.assertNotIn("hidden_on_purpose", messages)
        self.assertTrue(report["ok"])


if __name__ == "__main__":
    unittest.main()
