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
        self.assertIn("=== PAGE 1 (reading order) ===", packet)
        self.assertIn("=== PAGE 3 (reading order) ===", packet)
        self.assertNotIn("=== PAGE 4", packet, "text after the references heading is left out of the packet")
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
        self.assertEqual(context["prompt_card_chars"], 400, "the repo's default agent config")

    def test_context_takes_the_prompt_limit_from_the_projects_agent_config(self):
        (self.project.project_dir / "agent_bo.yaml").write_text(
            "prompt:\n  decision_engine_knowledge_max_chars: 250\n", encoding="utf-8"
        )
        LabBOService(projects_root=self.root).update_project_config("ev", {"agent_config_path": "agent_bo.yaml"})
        import json

        path = self.literature.write_context(self.project.project_dir)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["prompt_card_chars"], 250)
        LabBOService(projects_root=self.root).update_project_config("ev", {"agent_config_path": "gone.yaml"})
        path = self.literature.write_context(self.project.project_dir)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["prompt_card_chars"], 400, "unreadable -> default")

    def test_missing_pdftotext_is_a_clear_error(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"PDFTOTEXT": ""}), mock.patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(self.literature.LiteratureError, "pdftotext"):
                self.literature.find_pdftotext()

    def test_prepare_needs_a_project(self):
        with self.assertRaisesRegex(self.literature.LiteratureError, "No project"):
            self.literature.prepare_sources(self.base / "nowhere", [])


_LAYOUT = [
    "Fe(III)-Catalyzed Aerobic Oxidation of 1,4-Diols\nAbstract page without results.",
    (
        "Table 1 The effects of solvents and the loading of catalysts\n"
        "  Entry   x    y     Solvent      NMR yield of 2a/%\n"
        "  1       10   10    DCE          59\n"
        "  2       10   10    THF          17\n"
        "  7       10   10    Toluene      87 (81b)\n"
        "  11e     5    10    Toluene      88\n"
    ),
    "Scheme 2 substrate scope. Reactions ran at 25 °C for 24 h with 1.0 mmol substrate.",
]
_READING = [
    _LAYOUT[0],
    (
        "Study on the solvent effect led to the observation that the highest NMR yield of 87% for lactone 2a "
        "was realized upon using toluene as solvent (Table 1, entry 7). The role of Fe(NO₃)₃·9H₂O "
        "and TEMPO was found to be vital, since no product was formed without oxida-\ntion catalysts."
    ),
    _LAYOUT[2],
]
_CONTEXT = {
    "variables": [
        {"name": "Catalyst", "type": "categorical", "optimized": True, "options": ["cat_a", "cat_b"]},
        {"name": "Solvent", "type": "categorical", "optimized": True, "options": ["dce", "thf", "toluene"]},
        {"name": "Base", "type": "categorical", "optimized": False, "options": ["k2co3"]},
    ]
}
_GOOD = {
    "source_id": "paper_ab12cd34",
    "finding_id": "f1",
    "page": 2,
    "quote": "7 10 10 Toluene 87 (81b)",
    "summary": "Toluene gave 87% NMR yield of the lactone (81% isolated) with 10 mol% Fe and TEMPO at 25 °C.",
    "variable_scope": ["Solvent"],
    "proposed_mapping_status": "same_reaction_family",
    "confidence": 0.9,
    "transferability_note": "Butane-1,4-diol at 1 mmol; the project uses a different diol and scale.",
}


class VerifyDraftTests(unittest.TestCase):
    def setUp(self):
        from chem_agent_bo.lab import literature

        self.lit = literature
        self.texts = {"layout": _LAYOUT, "reading": _READING}

    def check(self, **changes):
        draft = {**_GOOD, **changes}
        return self.lit.check_draft(
            draft, texts=self.texts, context=_CONTEXT, taken_card_ids=set(), existing_excerpts=[]
        )

    def codes(self, record, kind="errors"):
        return {item["code"] for item in record[kind]}

    def test_a_faithful_table_row_passes(self):
        record = self.check()
        self.assertEqual(record["status"], "ok", record["errors"])
        self.assertEqual(record["card_id"], "paper_ab12cd34_f1")
        self.assertEqual(record["quote_match"]["mode"], "exact")
        self.assertEqual(record["quote_match"]["variant"], "layout")
        self.assertEqual(record["confidence"], "0.9")
        self.assertEqual(record["matched_options"], {"Solvent": ["toluene"]})

    def test_spacing_dashes_subscripts_and_hyphenation_do_not_matter(self):
        prose = self.check(
            quote="The role of Fe(NO3)3·9H2O and TEMPO was found to be vital, since no product was formed "
            "without oxidation catalysts.",
            summary="Both Fe(NO3)3·9H2O and TEMPO were required for the lactone to form in the screen here.",
        )
        self.assertEqual(prose["status"], "ok", prose["errors"])
        self.assertEqual(prose["quote_match"]["variant"], "reading")
        spaced = self.check(quote="7   10 10\nToluene    87 (81b)")
        self.assertEqual(spaced["status"], "ok", spaced["errors"])

    def test_a_fabricated_quote_is_rejected(self):
        record = self.check(quote="Toluene gave the highest yield of any solvent tested in this work.")
        self.assertEqual(record["status"], "rejected")
        self.assertIn("quote_not_found", self.codes(record))

    def test_a_changed_digit_is_not_a_typo(self):
        record = self.check(quote="7 10 10 Toluene 89 (81b)", summary=_GOOD["summary"].replace("87%", "89%"))
        self.assertIn("quote_not_found", self.codes(record))
        # An 11-entry row from the table with one digit altered must not pass as an approximate match either.
        altered = self.check(quote="Study on the solvent effect led to the observation that the highest NMR yield of 88% for lactone 2a")
        self.assertIn("quote_not_found", self.codes(altered))

    def test_a_small_wording_slip_is_accepted_but_flagged(self):
        quote = (
            "Study on the solvent effect led to the observation that the highest NMR yield of 87% for lactone 2a "
            "was realised upon using toluene as solvent"
        )
        record = self.check(
            quote=quote,
            page=2,
            summary="Toluene gave the highest NMR yield of the lactone, 87%, of the solvents in this screen.",
        )
        self.assertEqual(record["status"], "ok", record["errors"])
        self.assertIn("quote_approximate", self.codes(record, "warnings"))

    def test_wrong_page_is_corrected(self):
        record = self.check(page=1)
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["page"], 2)
        self.assertIn("page_corrected", self.codes(record, "warnings"))

    def test_joined_excerpts_must_all_be_on_one_page(self):
        joined = self.check(
            quote="1 10 10 DCE 59 [...] 2 10 10 THF 17",
            summary="DCE gave 59% and THF gave 17% NMR yield of the lactone in the solvent screen.",
        )
        self.assertEqual(joined["status"], "ok", joined["errors"])
        split_across_pages = self.check(quote="1 10 10 DCE 59 ... Scheme 2 substrate scope")
        self.assertIn("quote_not_found", self.codes(split_across_pages))

    def test_numbers_in_the_summary_must_come_from_the_paper(self):
        percent = self.check(summary="Toluene gave 92% NMR yield of the lactone (81% isolated) in this screen.")
        self.assertIn("number_not_in_quote", self.codes(percent))
        other = self.check(
            summary="Toluene gave 87% NMR yield of the lactone at 60 °C after 12 h in the reported screen.",
            quote="7 10 10 Toluene 87 (81b)",
        )
        self.assertEqual(other["status"], "ok")
        self.assertIn("number_not_on_page", self.codes(other, "warnings"))

    def test_scope_note_status_and_nodes_are_checked(self):
        bad_scope = self.check(variable_scope=["Solvnt"])
        self.assertIn("unknown_variable", self.codes(bad_scope))
        self.assertIn("Did you mean `Solvent`", bad_scope["errors"][0]["message"])
        self.assertIn("no_variable_scope", self.codes(self.check(variable_scope=[])))
        self.assertIn("no_transferability_note", self.codes(self.check(transferability_note=" ")))
        self.assertIn("bad_status", self.codes(self.check(proposed_mapping_status="same_redox_manifold")))
        self.assertIn("bad_confidence", self.codes(self.check(confidence="very")))
        self.assertIn("bad_target_node", self.codes(self.check(target_nodes=["nowhere"])))
        self.assertIn("quote_too_long", self.codes(self.check(quote="x" * 901)))

    def test_only_the_chemist_can_claim_a_direct_match(self):
        for claimed in ("direct", "same_start_end"):
            record = self.check(proposed_mapping_status=claimed)
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["mapping_status"], "same_reaction_family")
            self.assertIn("status_downgraded", self.codes(record, "warnings"))

    def test_scope_on_unoptimized_variables_only_would_never_be_shown(self):
        record = self.check(variable_scope=["Base"])
        self.assertIn("not_retrievable", self.codes(record, "warnings"))
        self.assertNotIn("not_retrievable", self.codes(self.check(variable_scope=["Base"], proposed_mapping_status="background"), "warnings"))

    def test_several_options_with_yields_are_flagged_as_lookup_like(self):
        record = self.check(
            quote="1 10 10 DCE 59 [...] 2 10 10 THF 17",
            summary="DCE gave 59% and THF gave 17% NMR yield of the lactone in the solvent screen.",
        )
        self.assertEqual(record["status"], "ok", record["errors"])
        self.assertEqual(record["matched_options"], {"Solvent": ["dce", "thf"]})
        self.assertIn("lookup_like", self.codes(record, "warnings"))

    def test_options_match_across_hyphens_underscores_and_case(self):
        context = {
            "variables": [
                {"name": "TEMPO derivative", "type": "categorical", "optimized": True, "options": ["4_OH_TEMPO", "TEMPO"]},
                {"name": "Solvent", "type": "categorical", "optimized": True, "options": ["dce"]},
            ]
        }
        texts = {"layout": ["The reaction using 4-OH-TEMPO afforded 2a in 38% yield, much lower than with TEMPO."]}
        record = self.lit.check_draft(
            dict(
                _GOOD,
                page=1,
                quote="The reaction using 4-OH-TEMPO afforded 2a in 38% yield, much lower than with TEMPO.",
                summary="4-OH-TEMPO in place of TEMPO gave a much lower 38% yield of the lactone here.",
                variable_scope=["TEMPO derivative"],
            ),
            texts=texts,
            context=context,
            taken_card_ids=set(),
            existing_excerpts=[],
        )
        self.assertEqual(record["status"], "ok", record["errors"])
        self.assertEqual(record["matched_options"], {"TEMPO derivative": ["4_OH_TEMPO", "TEMPO"]})

    def test_existing_cards_from_the_same_paper_are_pointed_out(self):
        record = self.lit.check_draft(
            dict(_GOOD),
            texts=self.texts,
            context=_CONTEXT,
            taken_card_ids=set(),
            existing_excerpts=[],
            same_paper_card_ids=["card_a", "card_b"],
        )
        self.assertEqual(record["status"], "ok")
        self.assertIn("paper_already_cited", self.codes(record, "warnings"))
        message = next(w["message"] for w in record["warnings"] if w["code"] == "paper_already_cited")
        self.assertIn("card_a, card_b", message)

    def test_findings_already_in_the_project_are_duplicates(self):
        same_id = self.lit.check_draft(
            dict(_GOOD),
            texts=self.texts,
            context=_CONTEXT,
            taken_card_ids={"paper_ab12cd34_f1"},
            existing_excerpts=[],
        )
        self.assertEqual(same_id["status"], "duplicate")
        old_excerpt = self.lit.normalize_text("Entry 7 was 7 10 10 Toluene 87 (81b) in the original Table 1, row seven.")
        same_quote = self.lit.check_draft(
            dict(_GOOD, finding_id="f9"),
            texts=self.texts,
            context=_CONTEXT,
            taken_card_ids=set(),
            existing_excerpts=[old_excerpt],
        )
        self.assertEqual(same_quote["status"], "duplicate")

    def test_a_summary_too_long_for_the_prompt_is_warned_about(self):
        padding = " More context that adds no new numbers." * 10
        long = self.check(summary=_GOOD["summary"] + padding)
        self.assertEqual(long["status"], "ok", long["errors"])
        self.assertIn("summary_long", self.codes(long, "warnings"))
        self.assertIn("first", next(w["message"] for w in long["warnings"] if w["code"] == "summary_long"))
        self.assertNotIn("summary_long", self.codes(self.check(), "warnings"))

    def test_the_budget_follows_the_projects_prompt_limit(self):
        small = dict(_CONTEXT, prompt_card_chars=150)
        record = self.lit.check_draft(
            dict(_GOOD), texts=self.texts, context=small, taken_card_ids=set(), existing_excerpts=[]
        )
        self.assertIn("summary_long", self.codes(record, "warnings"), "95 characters do not fit a 150 character card")
        roomy = dict(_CONTEXT, prompt_card_chars=900)
        record = self.lit.check_draft(
            dict(_GOOD, summary=_GOOD["summary"] + " Extra detail." * 30),
            texts=self.texts,
            context=roomy,
            taken_card_ids=set(),
            existing_excerpts=[],
        )
        self.assertNotIn("summary_long", self.codes(record, "warnings"))

    def test_a_source_without_text_is_rejected(self):
        record = self.lit.check_draft(
            dict(_GOOD), texts=None, context=_CONTEXT, taken_card_ids=set(), existing_excerpts=[]
        )
        self.assertEqual(record["status"], "rejected")
        self.assertIn("unknown_source", self.codes(record))


class _OnDiskProject(unittest.TestCase):
    def setUp(self):
        import json

        from chem_agent_bo.lab import literature

        self.json, self.lit = json, literature
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "root"
        self.project = _create_small_project(
            LabBOService(projects_root=self.root), "ev", reaction_scope="oxidative lactonization of diols"
        )
        self.dir = self.project.project_dir
        base = literature.work_dir(self.dir)
        (base / "text").mkdir(parents=True)
        (base / "drafts").mkdir()
        (base / "text" / "paper_ab12cd34.json").write_text(
            json.dumps({"pages": _LAYOUT, "pages_reading": _READING}), encoding="utf-8"
        )
        manifest = [
            {
                "source_id": "paper_ab12cd34",
                "status": "ready",
                "file": "Paper.pdf",
                "path": "C:/papers/Paper.pdf",
                "sha256": "ab12cd34" + "0" * 56,
                "doi": "10.1002/cjoc.202200768",
            },
            {"source_id": "scan_00000000", "status": "needs_ocr"},
        ]
        (base / "sources.jsonl").write_text("".join(json.dumps(item) + "\n" for item in manifest), encoding="utf-8")
        literature.write_context(self.dir)
        self.drafts = base / "drafts" / "drafts.jsonl"

    def _write(self, *records, extra=""):
        self.drafts.write_text(
            "".join(self.json.dumps(item) + "\n" for item in records) + extra, encoding="utf-8"
        )



class VerifyDraftsOnDiskTests(_OnDiskProject):
    def test_report_counts_and_verified_file(self):
        good = dict(_GOOD, variable_scope=["Solvent"])
        self._write(
            good,
            dict(good, finding_id="f2", quote="Toluene was the best solvent in every experiment we ran here."),
            dict(good, finding_id="f1", page=2),
            dict(good, source_id="scan_00000000", finding_id="f3"),
            extra="not json\n",
        )
        report = self.lit.verify_drafts(self.dir)
        self.assertEqual((report["checked"], report["ok"], report["rejected"]), (4, 1, 3))
        self.assertEqual(len(report["unreadable_lines"]), 1)
        self.assertEqual(report["unreadable_lines"][0]["line"], "5")
        codes = report["by_code"]
        for code in ("quote_not_found", "repeated_finding_id", "unknown_source"):
            self.assertIn(code, codes)
        written = [self.json.loads(line) for line in Path(report["verified_path"]).read_text(encoding="utf-8").splitlines()]
        self.assertEqual([item["status"] for item in written], ["ok", "rejected", "rejected", "rejected"])

    def test_cards_already_in_the_project_are_skipped(self):
        LabBOService(projects_root=self.root).import_evidence(
            "ev", cards=[{"card_id": "paper_ab12cd34_f1", "summary": "already here"}]
        )
        self._write(dict(_GOOD))
        report = self.lit.verify_drafts(self.dir)
        self.assertEqual((report["ok"], report["duplicate"]), (0, 1))

    def test_cards_citing_the_same_doi_are_reported(self):
        base = self.lit.work_dir(self.dir)
        manifest = [{"source_id": "paper_ab12cd34", "status": "ready", "doi": "10.1002/cjoc.202200768"}]
        (base / "sources.jsonl").write_text(self.json.dumps(manifest[0]) + "\n", encoding="utf-8")
        LabBOService(projects_root=self.root).import_evidence(
            "ev",
            cards=[
                {"card_id": "by_doi_field", "summary": "x", "doi": "10.1002/CJOC.202200768"},
                {"card_id": "by_source_text", "summary": "y", "source": "Li et al. DOI: 10.1002/cjoc.202200768."},
                {"card_id": "other_paper", "summary": "z", "doi": "10.1021/jacs.6b03948"},
            ],
        )
        self._write(dict(_GOOD))
        report = self.lit.verify_drafts(self.dir)
        self.assertEqual(report["ok"], 1)
        message = next(w["message"] for w in report["records"][0]["warnings"] if w["code"] == "paper_already_cited")
        self.assertIn("2 existing card(s)", message)
        self.assertIn("by_doi_field", message)
        self.assertNotIn("other_paper", message)

    def test_needs_prepare_first(self):
        import shutil as _shutil

        _shutil.rmtree(self.lit.work_dir(self.dir))
        with self.assertRaisesRegex(self.lit.LiteratureError, "evidence-prepare"):
            self.lit.verify_drafts(self.dir)


class SheetAndAcceptTests(_OnDiskProject):
    def setUp(self):
        super().setUp()
        import csv

        self.csv = csv
        self.service = LabBOService(projects_root=self.root)
        self.client = TestClient(create_app(self.root))
        self._write(
            dict(_GOOD),
            dict(
                _GOOD,
                finding_id="f2",
                quote="1 10 10 DCE 59 [...] 2 10 10 THF 17",
                summary="DCE gave 59% and THF gave 17% NMR yield of the lactone in the solvent screen.",
            ),
            dict(_GOOD, finding_id="f3", quote="This sentence is not in the paper at all, sorry."),
        )
        self.lit.verify_drafts(self.dir)
        self.sheet_path = self.lit.work_dir(self.dir) / "review_sheet.csv"

    def _build(self, **kwargs):
        return self.lit.build_sheet(self.dir, **kwargs)

    def _rows(self):
        with self.sheet_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(self.csv.DictReader(handle))

    def _save(self, rows):
        with self.sheet_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = self.csv.DictWriter(handle, fieldnames=self.lit.SHEET_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def _decide(self, decisions, edits=None):
        rows = self._rows()
        for row in rows:
            row["decision"] = decisions.get(row["card_id"], "")
            row.update((edits or {}).get(row["card_id"], {}))
        self._save(rows)

    def _card_count(self):
        return self.client.get("/api/projects/ev/evidence").json()["count"]

    def test_sheet_holds_only_checked_drafts_with_their_warnings(self):
        result = self._build()
        self.assertEqual((result["row_count"], result["skipped"]), (2, 1))
        raw = self.sheet_path.read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM so Excel opens it correctly")
        rows = {row["card_id"]: row for row in self._rows()}
        self.assertEqual(set(rows), {"paper_ab12cd34_f1", "paper_ab12cd34_f2"})
        first = rows["paper_ab12cd34_f1"]
        self.assertEqual(first["decision"], "")
        self.assertEqual(first["supporting_excerpt"], "7 10 10 Toluene 87 (81b)")
        self.assertEqual(first["variable_scope"], "Solvent")
        self.assertEqual(first["source"], "Paper.pdf (DOI 10.1002/cjoc.202200768)")
        self.assertIn("[lookup_like]", rows["paper_ab12cd34_f2"]["checks"])

    def test_a_sheet_with_decisions_is_not_overwritten_without_force(self):
        self._build()
        self._decide({"paper_ab12cd34_f1": "accept"})
        with self.assertRaisesRegex(self.lit.LiteratureError, "already has 1 decision"):
            self._build()
        self._build(force=True)
        self.assertEqual({row["decision"] for row in self._rows()}, {""})
        self.assertEqual(len(list(self.sheet_path.parent.glob("review_sheet_*.csv"))), 1, "old sheet kept")

    def test_accepted_rows_become_cards_and_everything_else_is_left_out(self):
        self._build()
        self._decide({"paper_ab12cd34_f1": "accept", "paper_ab12cd34_f2": "reject"})
        result = self.service.accept_evidence_sheet("ev")
        self.assertEqual((result["accepted"], result["rejected"], result["undecided"]), (1, 1, 0))
        self.assertEqual(result["card_ids"], ["paper_ab12cd34_f1"])
        self.assertEqual(self._card_count(), 1)
        card = self.client.get("/api/projects/ev/evidence").json()["cards"][0]
        self.assertEqual(card["mapping_status"], "same_reaction_family")
        self.assertEqual(card["doi"], "10.1002/cjoc.202200768")
        self.assertEqual(card["supporting_excerpt"], "7 10 10 Toluene 87 (81b)")
        self.assertIn("p. 2", card["notes"])
        self.assertIn("reviewed by the chemist", card["notes"])
        stored = self.project.evidence_path.read_text(encoding="utf-8")
        self.assertIn("C:/papers/Paper.pdf", stored)
        self.assertTrue(Path(result["backup_dir"]).exists())
        logged = (self.lit.work_dir(self.dir) / "accepted.jsonl").read_text(encoding="utf-8")
        self.assertIn("paper_ab12cd34_f1", logged)

    def test_nothing_decided_imports_nothing(self):
        self._build()
        result = self.service.accept_evidence_sheet("ev")
        self.assertEqual((result["imported_count"], result["undecided"]), (0, 2))
        self.assertEqual(self._card_count(), 0)

    def test_a_typo_in_decision_stops_everything(self):
        self._build()
        self._decide({"paper_ab12cd34_f1": "accept", "paper_ab12cd34_f2": "acept"})
        with self.assertRaises(self.lit.SheetError) as caught:
            self.service.accept_evidence_sheet("ev")
        self.assertIn("`acept`", " ".join(caught.exception.problems))
        self.assertEqual(self._card_count(), 0, "the valid accepted row was not imported either")

    def test_an_edited_quote_must_still_be_in_the_paper(self):
        self._build()
        self._decide(
            {"paper_ab12cd34_f1": "accept"},
            {"paper_ab12cd34_f1": {"supporting_excerpt": "7 10 10 Toluene 97 (81b)"}},
        )
        with self.assertRaises(self.lit.SheetError) as caught:
            self.service.accept_evidence_sheet("ev")
        self.assertIn("quote_not_found", " ".join(caught.exception.problems))
        self.assertEqual(self._card_count(), 0)

    def test_an_edited_summary_is_rechecked_too(self):
        self._build()
        self._decide(
            {"paper_ab12cd34_f1": "accept"},
            {"paper_ab12cd34_f1": {"summary": "Toluene gave 92% NMR yield of the lactone in this solvent screen."}},
        )
        with self.assertRaises(self.lit.SheetError) as caught:
            self.service.accept_evidence_sheet("ev")
        self.assertIn("number_not_in_quote", " ".join(caught.exception.problems))

    def test_the_chemist_may_upgrade_the_status_and_set_the_papers_scope(self):
        self._build()
        self._decide(
            {"paper_ab12cd34_f1": "accept"},
            {"paper_ab12cd34_f1": {"mapping_status": "direct", "reaction_scope": "esterification of phenols"}},
        )
        self.service.accept_evidence_sheet("ev")
        card = self.client.get("/api/projects/ev/evidence").json()["cards"][0]
        self.assertEqual((card["mapping_status"], card["reaction_scope"]), ("direct", "esterification of phenols"))
        # ...and the preview then shows the consequence of that choice.
        preview = self.client.get("/api/projects/ev/evidence/preview").json()
        never = {item["card_id"]: item for item in preview["never_retrieved"]}
        self.assertEqual(never["paper_ab12cd34_f1"]["reason"], "reaction_scope_mismatch")

    def test_a_chemist_cannot_smuggle_in_an_invalid_status(self):
        self._build()
        self._decide(
            {"paper_ab12cd34_f1": "accept"},
            {"paper_ab12cd34_f1": {"mapping_status": "same_redox_manifold"}},
        )
        with self.assertRaises(self.lit.SheetError) as caught:
            self.service.accept_evidence_sheet("ev")
        self.assertIn("bad_status", " ".join(caught.exception.problems))

    def test_accepting_the_same_sheet_twice_does_not_duplicate_cards(self):
        from chem_agent_bo.lab.evidence import EvidenceImportError

        self._build()
        self._decide({"paper_ab12cd34_f1": "accept"})
        self.service.accept_evidence_sheet("ev")
        with self.assertRaises(EvidenceImportError):
            self.service.accept_evidence_sheet("ev")
        self.assertEqual(self._card_count(), 1)

    def test_a_sheet_without_the_needed_columns_is_refused(self):
        self.sheet_path.write_text("decision,card_id\naccept,x\n", encoding="utf-8")
        with self.assertRaisesRegex(self.lit.SheetError, "missing column"):
            self.service.accept_evidence_sheet("ev")


class ScopeRuleTests(unittest.TestCase):
    """A card claiming direct precedent is hidden only when its reaction is unrelated to the project's."""

    project_scope = (
        "Intramolecular oxidative lactonization of a user-selected hydroxy-alcohol precursor to the "
        "corresponding lactone under an iron(III)-nitrate/nitroxyl oxidation manifold."
    )

    def _related(self, card_scope):
        from chem_agent_bo.lab.evidence import _reaction_related

        return _reaction_related(self.project_scope, card_scope)

    def test_the_same_kind_of_reaction_is_related_even_when_worded_differently(self):
        self.assertTrue(self._related("aerobic oxidative lactonization of 1,4-diols to γ-butyrolactones"))
        self.assertTrue(self._related("iron/nitroxyl aerobic oxidative lactonization"))
        self.assertTrue(self._related("oxidative lactonization"))

    def test_a_different_reaction_is_not(self):
        self.assertFalse(self._related("esterification of phenols"))
        self.assertFalse(self._related("palladium catalyzed cross coupling of aryl halides"))

    def test_known_limit_neighbouring_chemistry_can_share_two_general_words(self):
        """The rule reads words, not chemistry: 'oxidation' and 'alcohol' also occur in this project's
        scope, so a benzylic alcohol oxidation counts as related. Whether a card really is direct
        precedent stays the chemist's call (mapping_status); this only decides whether to hide it."""
        self.assertTrue(self._related("benzylic oxidation of alcohols"))

    def test_one_shared_general_word_is_not_enough(self):
        self.assertFalse(self._related("aerobic oxidation of sulfides"))

    def test_a_short_scope_needs_all_of_its_key_words(self):
        self.assertTrue(self._related("lactonization"))
        self.assertFalse(self._related("amidation"))

    def test_only_the_decision_to_hide_a_card_is_loosened_not_the_rank_bonus(self):
        from chem_agent_bo.lab.evidence import LAB_EVIDENCE_TARGET_NODES, _card_score

        card = _card(
            "li",
            mapping_status="same_start_end",
            reaction_scope="aerobic oxidative lactonization of 1,4-diols",
            variable_scope=["Solvent"],
            confidence="0.9",
        )
        score, reason = EvidenceStore._screen(
            card, variables=["Solvent"], reaction_scope=self.project_scope, target_nodes=list(LAB_EVIDENCE_TARGET_NODES)
        )
        self.assertEqual(reason, "")
        self.assertEqual(score, _card_score(card, variable_overlap=1, target_overlap=0, reaction_match=False))

    def test_an_unrelated_direct_card_is_still_hidden(self):
        card = _card("phenol", mapping_status="direct", reaction_scope="esterification of phenols")
        score, reason = EvidenceStore._screen(card, variables=["Solvent"], reaction_scope=self.project_scope)
        self.assertEqual((score, reason), (None, "reaction_scope_mismatch"))


class SelectionModeTests(unittest.TestCase):
    """`per_variable` gives each optimized variable its most specific card, then tops up by score."""

    variables = ["A", "B", "C"]

    def _store(self, *cards):
        return EvidenceStore(list(cards))

    def _ids(self, store, **kwargs):
        options = dict(variables=self.variables, max_items=10, selection="per_variable")
        options.update(kwargs)
        return [card.card_id for card in store.applicable(**options)]

    def test_a_specific_card_takes_the_slot_of_its_variable_ahead_of_a_broad_one(self):
        store = self._store(
            _card("broad", variable_scope=["A", "B", "C"], mapping_status="same_reaction_family", confidence="0.97"),
            _card("spec_a", variable_scope=["A"], mapping_status="same_reaction_family", confidence="0.8"),
            _card("spec_b", variable_scope=["B"], mapping_status="same_reaction_family", confidence="0.8"),
        )
        self.assertEqual(self._ids(store), ["spec_a", "spec_b", "broad"])
        self.assertEqual(self._ids(store, selection="score")[0], "broad", "score mode is untouched")

    def test_mapping_status_still_beats_specificity(self):
        store = self._store(
            _card("broad_family", variable_scope=["A", "B", "C"], mapping_status="same_reaction_family"),
            _card("spec_variable_level", variable_scope=["A"], mapping_status="variable_level"),
            _card("spec_b", variable_scope=["B"], mapping_status="same_reaction_family"),
            _card("spec_c", variable_scope=["C"], mapping_status="same_reaction_family"),
        )
        # A is contested by both; neither is otherwise forced, so the stronger status takes the slot.
        explained = {item["card_id"]: item for item in store.explain(variables=self.variables, selection="per_variable")}
        self.assertEqual(explained["broad_family"]["slots"], ["A"])
        self.assertEqual(explained["spec_variable_level"]["slots"], [], "it only tops up")

    def test_a_specific_card_takes_over_when_the_broad_one_is_needed_elsewhere(self):
        store = self._store(
            _card("broad_family", variable_scope=["A", "B", "C"], mapping_status="same_reaction_family"),
            _card("spec_variable_level", variable_scope=["A"], mapping_status="variable_level"),
        )
        # B and C have no other card, so broad_family must serve them; A then goes to its own card.
        explained = {item["card_id"]: item for item in store.explain(variables=self.variables, selection="per_variable")}
        self.assertEqual(explained["broad_family"]["slots"], ["B", "C"])
        self.assertEqual(explained["spec_variable_level"]["slots"], ["A"])

    def test_confidence_breaks_a_tie_between_equally_specific_cards(self):
        store = self._store(
            _card("low", variable_scope=["A"], confidence="0.6"),
            _card("high", variable_scope=["A"], confidence="0.9"),
        )
        self.assertEqual(self._ids(store)[0], "high")

    def test_one_card_can_fill_two_slots_and_is_listed_once(self):
        store = self._store(_card("ab", variable_scope=["A", "B"]), _card("c", variable_scope=["C"]))
        explained = {item["card_id"]: item for item in store.explain(variables=self.variables, selection="per_variable")}
        self.assertEqual(explained["ab"]["slots"], ["A", "B"])
        self.assertEqual(explained["c"]["slots"], ["C"])
        self.assertEqual(self._ids(store), ["ab", "c"])

    def test_a_broad_top_status_card_cannot_take_every_slot(self):
        """The chemist's project: one same_start_end card names every variable; it must not shut the rest out."""
        store = self._store(
            _card("broad_direct", variable_scope=["A", "B", "C"], mapping_status="same_start_end", confidence="0.97"),
            _card("spec_a", variable_scope=["A"]),
            _card("spec_b", variable_scope=["B"]),
            _card("spec_c", variable_scope=["C"]),
        )
        explained = {item["card_id"]: item for item in store.explain(variables=self.variables, selection="per_variable")}
        self.assertEqual(explained["broad_direct"]["slots"], ["A"])
        self.assertEqual(explained["spec_b"]["slots"], ["B"])
        self.assertEqual(explained["spec_c"]["slots"], ["C"])
        self.assertEqual(self._ids(store), ["broad_direct", "spec_b", "spec_c", "spec_a"])

    def test_a_variable_with_only_one_candidate_chooses_before_the_others(self):
        store = self._store(
            _card("x", variable_scope=["A", "B"], mapping_status="same_reaction_family"),
            _card("y", variable_scope=["B"], mapping_status="variable_level"),
        )
        # Design order puts B first, and B would take x on status; A has no other card, so A chooses first.
        explained = {
            item["card_id"]: item for item in store.explain(variables=["B", "A"], selection="per_variable")
        }
        self.assertEqual(explained["x"]["slots"], ["A"])
        self.assertEqual(explained["y"]["slots"], ["B"])

    def test_a_card_is_reused_only_where_it_is_the_sole_candidate(self):
        store = self._store(_card("only", variable_scope=["A", "B"]), _card("other", variable_scope=["C"]))
        explained = {item["card_id"]: item for item in store.explain(variables=self.variables, selection="per_variable")}
        self.assertEqual(explained["only"]["slots"], ["A", "B"])

    def test_cards_that_fill_no_slot_follow_in_score_order(self):
        store = self._store(
            _card("spec_a", variable_scope=["A"]),
            _card("background_b", variable_scope=["B"], mapping_status="background", confidence="0.99"),
            _card("nameless", variable_scope=[], mapping_status="same_reaction_family"),
        )
        # A -> spec_a; B -> background_b; C has no card; nameless has no scope so it only tops up.
        self.assertEqual(self._ids(store), ["spec_a", "background_b", "nameless"])

    def test_fewer_slots_than_variables_keeps_the_first_in_design_order(self):
        store = self._store(
            _card("spec_a", variable_scope=["A"]), _card("spec_b", variable_scope=["B"]), _card("spec_c", variable_scope=["C"])
        )
        self.assertEqual(self._ids(store, max_items=2), ["spec_a", "spec_b"])

    def test_applicable_and_explain_agree_at_every_cut(self):
        store = self._store(
            _card("broad", variable_scope=["A", "B", "C"], confidence="0.97"),
            _card("spec_a", variable_scope=["A"], confidence="0.8"),
            _card("spec_c", variable_scope=["C"], confidence="0.7"),
            _card("only_b", variable_scope=["B"], mapping_status="variable_level"),
            _card("hidden", mapping_status="out_of_scope"),
        )
        for k in range(0, 7):
            with self.subTest(k=k):
                applicable = self._ids(store, max_items=k)
                explained = sorted(
                    (e for e in store.explain(variables=self.variables, max_items=k, selection="per_variable") if e["retrieved"]),
                    key=lambda e: e["rank"],
                )
                self.assertEqual(applicable, [e["card_id"] for e in explained])

    def test_an_unknown_mode_is_an_error_not_a_silent_fallback(self):
        with self.assertRaisesRegex(ValueError, "per_variable"):
            self._store(_card("x")).applicable(variables=self.variables, selection="by_magic")

    def test_score_mode_is_the_default_and_is_unchanged(self):
        store = self._store(_card("broad", variable_scope=["A", "B", "C"]), _card("spec", variable_scope=["A"]))
        default = [c.card_id for c in store.applicable(variables=self.variables, max_items=10)]
        explicit = [c.card_id for c in store.applicable(variables=self.variables, max_items=10, selection="score")]
        self.assertEqual(default, explicit)
        self.assertEqual(default[0], "broad")


class SelectionSettingTests(unittest.TestCase):
    """The mode is a project setting: editable, validated, honoured by an ask, and shown by the preview."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "root"
        self.service = LabBOService(projects_root=self.root)
        self.project = _create_small_project(self.service, "sel", planner_name="random", batch_size=2)
        self.client = TestClient(create_app(self.root))
        cards = [
            {"card_id": "broad", "summary": "Broad card.", "variable_scope": ["Catalyst", "Solvent", "Base"], "confidence": "0.97"},
            {"card_id": "spec_solvent", "summary": "Solvent card.", "variable_scope": ["Solvent"], "confidence": "0.8"},
            {"card_id": "spec_base", "summary": "Base card.", "variable_scope": ["Base"], "confidence": "0.8"},
        ]
        response = self.client.post("/api/projects/sel/evidence/import", json={"cards": cards})
        self.assertEqual(response.status_code, 200, response.text)

    def _patch(self, value):
        return self.client.patch("/api/projects/sel/config", json={"evidence_selection": value})

    def test_default_is_score(self):
        self.assertEqual(self.project.load_config().evidence_selection, "score")

    def test_it_can_be_changed_and_is_saved(self):
        response = self._patch("per_variable")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("evidence_selection", response.json()["changed"])
        self.assertEqual(self.project.load_config().evidence_selection, "per_variable")

    def test_a_bad_value_is_refused_and_leaves_the_setting_alone(self):
        response = self._patch("by_magic")
        self.assertEqual(response.status_code, 400)
        self.assertIn("per_variable", response.text)
        self.assertEqual(self.project.load_config().evidence_selection, "score")

    def test_the_project_check_flags_a_bad_value_in_project_yaml(self):
        path = self.project.config_path
        path.write_text(path.read_text(encoding="utf-8") + "evidence_selection: by_magic\n", encoding="utf-8")
        report = self.service.check_project("sel")
        self.assertFalse(report["ok"])
        self.assertIn("evidence_selection", " ".join(item["message"] for item in report["errors"]))

    def test_an_ask_uses_the_mode_for_the_evidence_it_cites(self):
        def refs():
            batch = self.service.ask("sel", batch_size=2, planner_name="random", controller_mode="bo_only")
            return list(batch["recommendations"][0]["evidence_refs"])

        self.assertEqual(refs()[0], "broad")
        self._patch("per_variable")
        # per_variable: Catalyst -> broad (only card naming it), Solvent -> spec_solvent, Base -> spec_base
        self.assertEqual(refs(), ["broad", "spec_solvent", "spec_base"])

    def test_the_preview_reports_the_mode_and_each_cards_slots(self):
        self._patch("per_variable")
        body = self.client.get("/api/projects/sel/evidence/preview").json()
        self.assertEqual(body["selection"], "per_variable")
        slots = {item["card_id"]: item["slots"] for item in body["retrieved"]}
        self.assertEqual(slots["spec_solvent"], ["Solvent"])
        self.assertEqual(slots["spec_base"], ["Base"])


class PreviewPromptViewTests(unittest.TestCase):
    """The preview shows what the agent really reads, and where the settings and its limits disagree."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "root"
        self.service = LabBOService(projects_root=self.root)
        self.project = _create_small_project(self.service, "pv", planner_name="random", batch_size=2)
        self.client = TestClient(create_app(self.root))

    def _import(self, *cards):
        response = self.client.post("/api/projects/pv/evidence/import", json={"cards": list(cards)})
        self.assertEqual(response.status_code, 200, response.text)

    def _agent_config(self, top_k, max_items, max_chars):
        (self.project.project_dir / "agent_bo.yaml").write_text(
            "orchestrator:\n"
            f"  knowledge_top_k: {top_k}\n"
            "prompt:\n"
            f"  decision_engine_knowledge_max_items: {max_items}\n"
            f"  decision_engine_knowledge_max_chars: {max_chars}\n",
            encoding="utf-8",
        )
        self.service.update_project_config("pv", {"agent_config_path": "agent_bo.yaml"})

    def _preview(self):
        response = self.client.get("/api/projects/pv/evidence/preview")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_a_long_card_is_shown_cut_where_the_prompt_cuts_it(self):
        long_card = {
            "card_id": "long",
            "summary": "S" * 300,
            "variable_scope": ["Solvent"],
            "transferability_note": "T" * 200,
            "supporting_excerpt": "E" * 200,
        }
        short_card = {"card_id": "short", "summary": "Short finding.", "variable_scope": ["Catalyst"], "confidence": "0.5"}
        self._import(long_card, short_card)
        body = self._preview()
        by_id = {item["card_id"]: item for item in body["retrieved"]}
        self.assertEqual(body["prompt_card_chars"], 400)
        self.assertEqual(body["top_k_source"], "agent config")
        cut = by_id["long"]
        self.assertTrue(cut["prompt_text"].endswith("..."))
        self.assertEqual(len(cut["prompt_text"]), 403)
        self.assertGreater(cut["hidden_chars"], 0)
        self.assertIn("Transferability: T", cut["prompt_text"], "the note starts inside the limit...")
        self.assertNotIn("T" * 100, cut["prompt_text"], "...but only its first characters are seen")
        self.assertNotIn("Evidence excerpt", cut["prompt_text"], "the excerpt is past the cut")
        self.assertEqual(by_id["short"]["hidden_chars"], 0)
        self.assertTrue(by_id["short"]["prompt_text"].startswith("Short finding."))
        self.assertTrue(any("longer than the prompt keeps" in w for w in body["warnings"]))

    def test_cards_beyond_the_agents_read_limit_are_marked_and_warned(self):
        self._agent_config(top_k=4, max_items=2, max_chars=200)
        self._import(*[{"card_id": f"c{i}", "summary": f"Card {i}.", "variable_scope": ["Solvent"]} for i in range(4)])
        body = self._preview()
        self.assertEqual((body["top_k"], body["prompt_max_items"], body["prompt_card_chars"]), (4, 2, 200))
        self.assertEqual([item["read_by_agent"] for item in body["retrieved"]], [True, True, False, False])
        self.assertTrue(any("knowledge_top_k is 4 but decision_engine_knowledge_max_items is 2" in w for w in body["warnings"]))

    def test_per_variable_warns_when_there_are_more_variable_slots_than_cards_read(self):
        self._agent_config(top_k=5, max_items=2, max_chars=400)
        self._import(
            {"card_id": "cat", "summary": "Catalyst card.", "variable_scope": ["Catalyst"]},
            {"card_id": "sol", "summary": "Solvent card.", "variable_scope": ["Solvent"]},
            {"card_id": "base", "summary": "Base card.", "variable_scope": ["Base"]},
        )
        self.service.update_project_config("pv", {"evidence_selection": "per_variable"})
        body = self._preview()
        self.assertEqual([item["slots"] for item in body["retrieved"][:3]], [["Catalyst"], ["Solvent"], ["Base"]])
        warning = next(w for w in body["warnings"] if "optimized variables have a card of their own" in w)
        self.assertIn("3 of 3", warning)
        self.assertIn("knowledge_top_k", warning)
        self.assertIn("decision_engine_knowledge_max_items", warning)

    def test_no_warning_when_the_limits_fit(self):
        self._agent_config(top_k=3, max_items=3, max_chars=400)
        self._import(
            {"card_id": "cat", "summary": "Catalyst card.", "variable_scope": ["Catalyst"]},
            {"card_id": "sol", "summary": "Solvent card.", "variable_scope": ["Solvent"]},
            {"card_id": "base", "summary": "Base card.", "variable_scope": ["Base"]},
        )
        self.service.update_project_config("pv", {"evidence_selection": "per_variable"})
        self.assertEqual(self._preview()["warnings"], [])

    def test_an_unreadable_agent_config_falls_back_to_the_defaults(self):
        self.service.update_project_config("pv", {"agent_config_path": "missing_agent.yaml"})
        self._import({"card_id": "one", "summary": "One.", "variable_scope": ["Solvent"]})
        body = self._preview()
        self.assertEqual((body["top_k"], body["prompt_max_items"], body["prompt_card_chars"]), (5, 5, 400))
        self.assertEqual(body["top_k_source"], "default")

    def test_cards_with_the_same_summary_are_flagged_in_the_preview_and_the_check(self):
        twin = {"summary": "The very same finding.", "variable_scope": ["Solvent"]}
        self._import(dict(twin, card_id="twin_a"), dict(twin, card_id="twin_b"), {"card_id": "other", "summary": "Different."})
        preview_warning = next(w for w in self._preview()["warnings"] if "same summary" in w)
        self.assertIn("twin_a", preview_warning)
        self.assertIn("twin_b", preview_warning)
        report = self.service.check_project("pv")
        message = next(w["message"] for w in report["warnings"] if "same summary" in w["message"])
        self.assertIn("twin_a", message)
        self.assertNotIn("other", message)
        self.assertTrue(report["ok"], "duplicates are a warning, not an error")


if __name__ == "__main__":
    unittest.main()
