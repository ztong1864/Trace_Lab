"""Evidence cards from papers: retrieval preview, drafting checks, review sheet, publish."""

from pathlib import Path
import shutil
import tempfile
import textwrap
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


def _minimal_pdf(page_texts: list[str]) -> bytes:
    """A small valid PDF with one line of Helvetica text per page (empty text gives a blank page)."""
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * index} 0 R" for index in range(len(page_texts)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>".encode())
    font_number = 3 + 2 * len(page_texts)
    for index, text in enumerate(page_texts):
        page_number = 3 + 2 * index
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {page_number + 1} 0 R "
            f"/Resources << /Font << /F1 {font_number} 0 R >> >> >>".encode()
        )
        lines = [line for paragraph in text.split("\n") for line in (textwrap.wrap(paragraph, 80) or [""])]
        shown = " T* ".join(
            "(" + line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ") Tj" for line in lines
        )
        stream = f"BT /F1 9 Tf 11 TL 20 700 Td {shown} ET".encode() if text else b""
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


_LONG = " ".join(["Table 1 entry 1 toluene gave 87% yield with Fe(NO3)3 and TEMPO at 25 C."] * 6)


class PrepareSourcesTests(unittest.TestCase):
    def setUp(self):
        from chem_agent_bo.lab import literature

        self.literature = literature
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.root = self.base / "root"
        self.project = _create_small_project(
            LabBOService(projects_root=self.root), "ev", reaction_scope="oxidative lactonization of diols"
        )
        self.pdf_dir = self.base / "papers"
        self.pdf_dir.mkdir()

    def _pdf(self, name: str, pages: list[str]) -> Path:
        path = self.pdf_dir / name
        path.write_bytes(_minimal_pdf(pages))
        return path

    def _prepare(self, *paths):
        return self.literature.prepare_sources(self.project.project_dir, list(paths))

    @unittest.skipUnless(shutil.which("pdftotext"), "pdftotext (poppler) is not installed")
    def test_prepare_real_pdfs_makes_text_packets_and_context(self):
        good = self._pdf("Fe(NO3)3+TEMPO-rt°.pdf", [_LONG + " doi 10.1002/cjoc.202200768.", _LONG, "References", _LONG])
        blank = self._pdf("scan.pdf", ["", ""])
        result = self._prepare(good, blank)
        by_file = {item["file"]: item for item in result["sources"]}
        ready, scanned = by_file[good.name], by_file[blank.name]
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["page_count"], 4)
        self.assertEqual(ready["doi"], "10.1002/cjoc.202200768")
        self.assertEqual(ready["references_from_page"], 3)
        self.assertEqual(scanned["status"], "needs_ocr")
        self.assertEqual(result["ready_count"], 1)

        pages = self.literature.load_pages(self.project.project_dir, ready["source_id"])
        self.assertIn("87% yield", pages[0])
        packet = (self.literature.work_dir(self.project.project_dir) / "packets" / ready["packets"][0]).read_text(
            encoding="utf-8"
        )
        self.assertIn("=== PAGE 1 ===", packet)
        self.assertIn("=== PAGE 3 ===", packet)
        self.assertNotIn("=== PAGE 4 ===", packet, "text after the references heading is left out of the packet")
        self.assertTrue(self.literature.work_dir(self.project.project_dir).joinpath("context.json").exists())

    def test_reprepare_is_idempotent_and_keeps_other_sources(self):
        first = self._pdf("a.pdf", [_LONG, _LONG])
        second = self._pdf("b.pdf", [_LONG + " unlike a", _LONG])
        self._prepare(first)
        self._prepare(first, second)
        self._prepare(second)
        manifest = self.literature.read_manifest(self.project.project_dir)
        self.assertEqual(len(manifest), 2)
        self.assertEqual(len({item["source_id"] for item in manifest}), 2)

    def test_failure_of_one_paper_does_not_stop_the_others(self):
        good = self._pdf("good.pdf", [_LONG, _LONG])
        broken = self.pdf_dir / "broken.pdf"
        broken.write_bytes(b"this is not a pdf")
        result = self._prepare(broken, good)
        statuses = {item["file"]: item["status"] for item in result["sources"]}
        self.assertEqual(statuses["good.pdf"], "ready")
        self.assertEqual(statuses["broken.pdf"], "failed")

    def test_context_carries_the_projects_variables_options_and_existing_cards(self):
        LabBOService(projects_root=self.root).import_evidence(
            "ev", cards=[{"card_id": "old_1", "summary": "x", "doi": "10.1/x"}]
        )
        path = self.literature.write_context(self.project.project_dir)
        import json

        context = json.loads(path.read_text(encoding="utf-8"))
        names = {item["name"]: item for item in context["variables"]}
        self.assertEqual(set(names), {"Catalyst", "Solvent", "Base"})
        self.assertEqual(sorted(names["Solvent"]["options"]), ["dce", "meoh", "thf"])
        self.assertEqual(context["reaction_scope"], "oxidative lactonization of diols")
        self.assertEqual(context["existing_cards"][0]["card_id"], "old_1")

    def test_missing_pdftotext_is_a_clear_error(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"PDFTOTEXT": ""}), mock.patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(self.literature.LiteratureError, "pdftotext"):
                self.literature.find_pdftotext()

    def test_prepare_needs_a_project(self):
        with self.assertRaisesRegex(self.literature.LiteratureError, "No project"):
            self.literature.prepare_sources(self.base / "nowhere", [])


if __name__ == "__main__":
    unittest.main()
