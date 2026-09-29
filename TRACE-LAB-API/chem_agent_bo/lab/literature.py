"""Turn papers into checkable text for drafting evidence cards.

The workflow is split so that the parts that can be wrong are checked by code:

1. `prepare_sources`  PDF -> per-page text, a manifest, the project's context and
                      page-tagged packets for whoever drafts the cards.
2. (drafting)         done outside TRACE, into `evidence_work/drafts/*.jsonl`.
3. `verify_drafts`    every quote and number is checked against the page text.
4. review sheet       the chemist accepts or rejects; see `evidence_sheet`.

Everything lives in `<project>/evidence_work/`; the project's own evidence file is
only touched when reviewed cards are imported.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from chem_agent_bo.lab.evidence import LAB_EVIDENCE_TARGET_NODES, EvidenceStore
from chem_agent_bo.lab.project import LabProject

WORK_DIR = "evidence_work"
PAGES_PER_PACKET = 12
# Below this many characters per page a PDF is treated as scanned.
MIN_CHARS_PER_PAGE = 200
MAX_OPTIONS_IN_CONTEXT = 300

_DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
_REFERENCES_HEADING = re.compile(
    r"^\s*(references?|references and notes|notes and references|literature cited|bibliography)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class LiteratureError(ValueError):
    """A paper can't be turned into text, or a work file is missing."""


def work_dir(project_dir: str | Path) -> Path:
    return Path(project_dir) / WORK_DIR


def find_pdftotext() -> str:
    """Locate poppler's pdftotext: $PDFTOTEXT, then PATH."""
    configured = os.environ.get("PDFTOTEXT", "").strip()
    found = configured if configured and Path(configured).exists() else shutil.which("pdftotext")
    if not found:
        raise LiteratureError(
            "pdftotext (poppler) was not found. Install poppler, or set the PDFTOTEXT environment "
            "variable to the full path of pdftotext. In Git Bash it is usually already on the PATH."
        )
    return found


def extract_pages(pdf_path: str | Path, *, layout: bool = True) -> list[str]:
    """Text of each page: `layout=True` keeps table columns, `False` gives reading order (better for prose)."""
    path = Path(pdf_path)
    if not path.is_file():
        raise LiteratureError(f"No such file: {path}")
    exe = find_pdftotext()
    with tempfile.TemporaryDirectory() as tmp:
        # Some pdftotext builds mishandle non-ASCII paths (paper names carry °, ·, ...), so work on a copy.
        source = Path(tmp) / "paper.pdf"
        shutil.copyfile(path, source)
        completed = subprocess.run(
            [exe, *(["-layout"] if layout else []), "-enc", "UTF-8", str(source), "-"],
            capture_output=True,
            timeout=180,
            check=False,
        )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()[:300]
        raise LiteratureError(f"pdftotext could not read {path.name}: {detail or 'exit ' + str(completed.returncode)}")
    text = completed.stdout.decode("utf-8", errors="replace")
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return pages


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:48] or "paper"


def source_id_for(path: str | Path, sha256: str) -> str:
    """Stable id: the file name plus a hash prefix, so a renamed copy of the same paper keeps its hash part."""
    return f"{slug(Path(path).stem)}_{sha256[:8]}"


def guess_doi(pages: list[str]) -> str:
    for page in pages[:2]:
        match = _DOI.search(page)
        if match:
            return match.group(0).rstrip(".,;)]}")
    return ""


def references_start_page(pages: list[str]) -> int | None:
    """1-based page where a References heading first appears after the first page, if any."""
    for number, page in enumerate(pages, start=1):
        if number > 1 and _REFERENCES_HEADING.search(page):
            return number
    return None


def prepare_sources(
    project_dir: str | Path,
    pdf_paths: list[str | Path],
    *,
    pages_per_packet: int = PAGES_PER_PACKET,
) -> dict[str, Any]:
    """Extract page text and packets for each PDF and refresh the project's context file.

    Safe to repeat: a paper is identified by its content hash, so it is never added twice.
    """
    project = LabProject(Path(project_dir))
    if not project.config_path.exists():
        raise LiteratureError(f"No project at {project.project_dir} (missing project.yaml).")
    base = work_dir(project.project_dir)
    (base / "text").mkdir(parents=True, exist_ok=True)
    (base / "packets").mkdir(exist_ok=True)
    (base / "drafts").mkdir(exist_ok=True)
    manifest = {item["source_id"]: item for item in read_manifest(project.project_dir)}
    results = []
    for raw in pdf_paths:
        path = Path(raw)
        try:
            record = _prepare_one(base, path, pages_per_packet=pages_per_packet)
        except LiteratureError as exc:
            record = {"file": path.name, "path": str(path.resolve()), "status": "failed", "error": str(exc)}
        else:
            manifest[record["source_id"]] = record
        results.append(record)
    _write_jsonl(base / "sources.jsonl", list(manifest.values()))
    context_path = write_context(project.project_dir)
    return {
        "project_dir": str(project.project_dir),
        "work_dir": str(base),
        "context": str(context_path),
        "sources": results,
        "ready_count": sum(1 for item in results if item.get("status") == "ready"),
    }


def _prepare_one(base: Path, path: Path, *, pages_per_packet: int) -> dict[str, Any]:
    pages = extract_pages(path)
    reading = extract_pages(path, layout=False)
    sha = file_sha256(path)
    source_id = source_id_for(path, sha)
    characters = sum(len(page.strip()) for page in pages)
    references_from = references_start_page(pages)
    record: dict[str, Any] = {
        "source_id": source_id,
        "file": path.name,
        "path": str(path.resolve()),
        "sha256": sha,
        "page_count": len(pages),
        "characters": characters,
        "doi": guess_doi(pages),
        "references_from_page": references_from,
        "packets": [],
    }
    if not pages or characters / max(1, len(pages)) < MIN_CHARS_PER_PAGE:
        record["status"] = "needs_ocr"
        record["note"] = "Almost no text was found; this looks like a scanned PDF. Run OCR first."
        return record
    record["status"] = "ready"
    (base / "text" / f"{source_id}.json").write_text(
        json.dumps(
            {"source_id": source_id, "file": path.name, "sha256": sha, "pages": pages, "pages_reading": reading},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    last_page = references_from if references_from else len(pages)
    for old in (base / "packets").glob(f"{source_id}_p*.txt"):
        old.unlink()
    for first in range(1, last_page + 1, pages_per_packet):
        last = min(first + pages_per_packet - 1, last_page)
        name = f"{source_id}_p{first:02d}-{last:02d}.txt"
        header = (
            f"SOURCE {source_id}\nfile: {path.name}\nsha256: {sha[:8]}\ndoi: {record['doi'] or 'not found'}\n"
            f"pages {first}-{last} of {len(pages)}"
            + (f" (references start on page {references_from}; later pages are left out)" if references_from else "")
            + "\n"
        )
        body = "\n".join(_packet_page(number, pages, reading) for number in range(first, last + 1))
        (base / "packets" / name).write_text(header + body + "\n", encoding="utf-8")
        record["packets"].append(name)
    return record


_TABLE_WORD = re.compile(r"\bTable\s+\d", re.IGNORECASE)
_TABLE_ROW = re.compile(r"\d\S*\s{2,}\S.*\s{2,}\S")


def _looks_tabular(layout_page: str) -> bool:
    return bool(_TABLE_WORD.search(layout_page)) or sum(bool(_TABLE_ROW.search(line)) for line in layout_page.splitlines()) >= 4


def _packet_page(number: int, layout: list[str], reading: list[str]) -> str:
    """One page for the drafter: prose in reading order, plus the column layout when the page has a table."""
    text = f"\n=== PAGE {number} (reading order) ===\n{(reading[number - 1] if number <= len(reading) else '').rstrip()}"
    if _looks_tabular(layout[number - 1]):
        text += f"\n--- PAGE {number} (table layout; quote table rows from here) ---\n{layout[number - 1].rstrip()}"
    return text


def write_context(project_dir: str | Path) -> Path:
    """What the project needs from a paper, taken from the project itself (variables, options, scope)."""
    project = LabProject(Path(project_dir))
    config = project.load_config()
    design = project.load_design_space()
    active = set(design.variable_names)
    variables = []
    for variable in design.variables:
        options = variable.option_values()
        entry: dict[str, Any] = {
            "name": variable.name,
            "type": variable.kind,
            "optimized": variable.name in active,
            "unit": variable.unit,
            "option_count": len(options),
            "options": options[:MAX_OPTIONS_IN_CONTEXT],
        }
        if variable.kind == "continuous":
            entry.update(low=variable.low, high=variable.high)
        variables.append(entry)
    store = EvidenceStore.load(project.evidence_path)
    context = {
        "project_id": config.project_id,
        "reaction_name": config.reaction_name,
        "reaction_scope": config.reaction_scope,
        "objective_name": config.objective_name,
        "goal": config.goal,
        "variables": variables,
        "existing_cards": [
            {"card_id": card.card_id, "doi": card.doi, "source": card.source[:160]} for card in store.cards
        ],
    }
    base = work_dir(project.project_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / "context.json"
    path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_manifest(project_dir: str | Path) -> list[dict[str, Any]]:
    return _read_jsonl(work_dir(project_dir) / "sources.jsonl")


def load_page_texts(project_dir: str | Path, source_id: str) -> dict[str, list[str]]:
    """Both extractions of a paper: `layout` (table columns) and `reading` (prose order)."""
    path = work_dir(project_dir) / "text" / f"{source_id}.json"
    if not path.exists():
        raise LiteratureError(f"No extracted text for source `{source_id}`; run evidence-prepare first.")
    stored = json.loads(path.read_text(encoding="utf-8"))
    layout = list(stored["pages"])
    return {"layout": layout, "reading": list(stored.get("pages_reading") or layout)}


def load_pages(project_dir: str | Path, source_id: str) -> list[str]:
    return load_page_texts(project_dir, source_id)["layout"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")


# ---------------------------------------------------------------------------
# Verifying drafts against the paper text
# ---------------------------------------------------------------------------

# Statuses a draft may propose. `direct` and `same_start_end` say the paper matches the
# project's own reaction, which needs the chemist's knowledge of both, so they are downgraded.
DRAFTABLE_STATUSES = ("same_reaction_family", "variable_level", "background", "out_of_scope")
CHEMIST_ONLY_STATUSES = ("direct", "same_start_end")
DEFAULT_TARGET_NODES = ("design_init_experiments", "hypothesis_action")
MAX_QUOTE_CHARS = 900
MIN_SUMMARY_CHARS = 40
FUZZY_THRESHOLD = 0.95
MIN_OPTION_CHARS = 3
MIN_DUPLICATE_CHARS = 20

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_OTHER_QUANTITY = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:mol\s*%|°\s*C|℃|h\b|hours?\b|mmol\b|mL\b|equiv\b|atm\b|bar\b)", re.IGNORECASE
)
_ELLIPSIS = re.compile(r"\[\s*(?:\.\.\.|…)\s*\]|\.\.\.|…")
_CONFIDENCE_WORDS = {"high", "medium", "low", "strong", "moderate", "weak"}


def normalize_text(text: str) -> str:
    """Make PDF text and quotes comparable: unicode forms, dashes, hyphenation, case and spacing."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("­", "")
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text))


def _has_number(text: str, number: str) -> bool:
    return bool(re.search(rf"(?<![\d.]){re.escape(number)}(?!\d|\.\d)", text))


def _fuzzy_score(part: str, text: str) -> float:
    """Best similarity of `part` to any stretch of `text`; 0 when the digits in it are not all present."""
    size = len(part)
    if size < 20 or len(text) < size // 2:
        return 0.0
    step = max(8, size // 6)
    best, best_start = 0.0, 0
    for start in range(0, max(1, len(text) - size + 1), step):
        matcher = SequenceMatcher(None, text[start : start + size], part, autojunk=False)
        if matcher.real_quick_ratio() <= best or matcher.quick_ratio() <= best:
            continue
        score = matcher.ratio()
        if score > best:
            best, best_start = score, start
    for start in range(max(0, best_start - step), min(len(text), best_start + step) + 1):
        score = SequenceMatcher(None, text[start : start + size], part, autojunk=False).ratio()
        if score > best:
            best, best_start = score, start
    # A near match must not differ in a digit: 87 vs 88 is a fabricated result, not a typo.
    window = text[max(0, best_start - step) : best_start + size + step]
    return best if _numbers(part) <= _numbers(window) else 0.0


def _match_on_page(parts: list[str], page_text: str) -> tuple[str, float] | None:
    """('exact' | 'fuzzy', score) when every part of the quote is on the page, else None."""
    worst, mode = 1.0, "exact"
    for part in parts:
        if part in page_text:
            continue
        score = _fuzzy_score(part, page_text)
        if score < FUZZY_THRESHOLD:
            return None
        worst, mode = min(worst, score), "fuzzy"
    return mode, worst


def locate_quote(quote: str, texts: dict[str, list[str]], page: int | None) -> dict[str, Any] | None:
    """Find a quote in the paper. Tries the stated page first, then every page.

    A quote may join excerpts with `...`; each piece must be found (on the same page).
    Returns {page, variant, mode, score} or None.
    """
    parts = [part.strip() for part in _ELLIPSIS.split(normalize_text(quote)) if len(part.strip()) >= 8]
    if not parts:
        return None
    normalized = {variant: [normalize_text(item) for item in pages] for variant, pages in texts.items()}
    page_count = max((len(pages) for pages in normalized.values()), default=0)
    order = list(range(1, page_count + 1))
    if page and 1 <= page <= page_count:
        order.remove(page)
        order.insert(0, page)
    best: dict[str, Any] | None = None
    for number in order:
        for variant, pages in normalized.items():
            if number > len(pages):
                continue
            found = _match_on_page(parts, pages[number - 1])
            if found is None:
                continue
            hit = {"page": number, "variant": variant, "mode": found[0], "score": round(found[1], 3)}
            if found[0] == "exact":
                return hit
            if best is None or hit["score"] > best["score"]:
                best = hit
    return best


def _loose(text: str) -> str:
    """Normalized text with `_`, `-` and spaces treated alike, so `4_OH_TEMPO` finds `4-OH-TEMPO`."""
    return re.sub(r"[_\-\s]+", " ", normalize_text(text))


def _option_hits(quote_norm: str, variables: list[dict[str, Any]]) -> dict[str, list[str]]:
    quote_norm = _loose(quote_norm)
    hits: dict[str, list[str]] = {}
    for variable in variables:
        if variable.get("type") == "continuous":
            continue
        found = []
        for option in variable.get("options", []):
            normalized = _loose(option)
            if len(normalized) < MIN_OPTION_CHARS:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", quote_norm):
                found.append(str(option))
        if found:
            hits[variable["name"]] = found
    return hits


def _confidence_text(value: Any) -> str | None:
    """Normalize a confidence to '0.85'-style text or a word; None when it is neither."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return f"{float(value):g}" if 0 <= float(value) <= 1 else None
    text = str(value or "").strip().lower()
    if text in _CONFIDENCE_WORDS:
        return text
    try:
        return f"{float(text):g}" if 0 <= float(text) <= 1 else None
    except ValueError:
        return None


class _Findings:
    def __init__(self) -> None:
        self.errors: list[dict[str, str]] = []
        self.warnings: list[dict[str, str]] = []

    def error(self, code: str, message: str) -> None:
        self.errors.append({"code": code, "message": message})

    def warn(self, code: str, message: str) -> None:
        self.warnings.append({"code": code, "message": message})


def check_draft(
    draft: dict[str, Any],
    *,
    texts: dict[str, list[str]] | None,
    context: dict[str, Any],
    taken_card_ids: set[str],
    existing_excerpts: list[str],
    same_paper_card_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Check one drafted finding against the paper. Returns the draft plus `status`, `errors`, `warnings`.

    status is `ok`, `rejected` (has errors) or `duplicate` (already in the project).
    What this cannot prove: that a yield belongs to the condition the summary names. The
    chemist checks that by reading the quote beside the summary in the review sheet.
    """
    found = _Findings()
    source_id = str(draft.get("source_id") or "").strip()
    finding_id = str(draft.get("finding_id") or "").strip()
    quote = str(draft.get("quote") or "").strip()
    summary = str(draft.get("summary") or "").strip()
    result: dict[str, Any] = {
        **draft,
        "source_id": source_id,
        "finding_id": finding_id,
        "card_id": f"{source_id}_{slug(finding_id)}" if source_id and finding_id else "",
    }
    if not source_id or not finding_id:
        found.error("missing_id", "Every draft needs a source_id and a finding_id.")
    if texts is None:
        found.error("unknown_source", f"Source `{source_id}` has no extracted text (not prepared, or it needs OCR).")
    if not quote:
        found.error("no_quote", "A draft needs a verbatim quote from the paper.")
    elif len(quote) > MAX_QUOTE_CHARS:
        found.error("quote_too_long", f"The quote has {len(quote)} characters; the limit is {MAX_QUOTE_CHARS}.")
    if len(summary) < MIN_SUMMARY_CHARS:
        found.error(
            "summary_too_short", f"The summary needs at least {MIN_SUMMARY_CHARS} characters of what was run and found."
        )
    if not str(draft.get("transferability_note") or "").strip():
        found.error(
            "no_transferability_note",
            "Say how the paper's substrate, scale or conditions differ from this project.",
        )

    page_hint = draft.get("page")
    try:
        page_hint = int(page_hint) if page_hint not in (None, "") else None
    except (TypeError, ValueError):
        found.warn("bad_page", f"page `{draft.get('page')}` is not a number; searching every page.")
        page_hint = None
    quote_norm = normalize_text(quote)
    page_text_norm = ""
    if texts is not None and quote and len(quote) <= MAX_QUOTE_CHARS:
        located = locate_quote(quote, texts, page_hint)
        if located is None:
            found.error(
                "quote_not_found",
                "The quote was not found in the paper's text (checked every page, both text layouts). "
                "Quote the paper word for word.",
            )
        else:
            result["page"] = located["page"]
            result["quote_match"] = {key: located[key] for key in ("mode", "score", "variant")}
            if page_hint is not None and located["page"] != page_hint:
                found.warn("page_corrected", f"The quote is on page {located['page']}, not page {page_hint}; corrected.")
            if located["mode"] == "fuzzy":
                found.warn("quote_approximate", f"The quote matches the paper only approximately ({located['score']}).")
            page_text_norm = " ".join(
                normalize_text(pages[located["page"] - 1]) for pages in texts.values() if located["page"] <= len(pages)
            )

    # Numbers in the summary must come from the paper.
    if summary and quote_norm:
        for number in dict.fromkeys(_PERCENT.findall(summary)):
            if not _has_number(quote_norm, number):
                found.error("number_not_in_quote", f"The summary says {number}%, which is not in the quote.")
        if page_text_norm:
            for number in dict.fromkeys(_OTHER_QUANTITY.findall(summary)):
                if not _has_number(page_text_norm, number):
                    found.warn("number_not_on_page", f"The summary gives {number}, which is not on the quoted page.")

    # Scope: exact design variable names.
    variables = context.get("variables", [])
    names = [item["name"] for item in variables]
    optimized = {item["name"] for item in variables if item.get("optimized")}
    scope = draft.get("variable_scope") or []
    scope = [item.strip() for item in scope.replace(";", ",").split(",")] if isinstance(scope, str) else list(scope)
    scope = [str(item).strip() for item in scope if str(item).strip()]
    for name in scope:
        if name not in names:
            close = difflib.get_close_matches(name, names, n=1)
            hint = f" Did you mean `{close[0]}`?" if close else ""
            found.error("unknown_variable", f"variable_scope names `{name}`, which is not a design variable.{hint}")
    result["variable_scope"] = scope
    if not scope:
        found.error("no_variable_scope", "List the design variables this finding is about (variable_scope).")

    # Status: drafts cannot claim the paper matches the project's own reaction.
    status = str(draft.get("proposed_mapping_status") or draft.get("mapping_status") or "same_reaction_family")
    status = status.strip().lower()
    if status in CHEMIST_ONLY_STATUSES:
        found.warn(
            "status_downgraded",
            f"`{status}` needs the chemist's judgement; set to same_reaction_family for review.",
        )
        status = "same_reaction_family"
    elif status not in DRAFTABLE_STATUSES:
        found.error("bad_status", f"mapping_status `{status}` must be one of {list(DRAFTABLE_STATUSES)}.")
    result["mapping_status"] = status
    if scope and status != "background" and all(name in names for name in scope) and not optimized.intersection(scope):
        found.warn(
            "not_retrievable",
            "None of these variables is optimized in this project, so the card would never be shown; "
            "use `background` or name an optimized variable.",
        )

    confidence = _confidence_text(draft.get("confidence", "medium"))
    if confidence is None:
        found.error("bad_confidence", "confidence must be a number from 0 to 1, or high / medium / low.")
    result["confidence"] = confidence or "medium"

    nodes = draft.get("target_nodes") or list(DEFAULT_TARGET_NODES)
    nodes = [item.strip() for item in nodes.split(",")] if isinstance(nodes, str) else list(nodes)
    for node in nodes:
        if node not in LAB_EVIDENCE_TARGET_NODES:
            found.error("bad_target_node", f"target node `{node}` is not one of {list(LAB_EVIDENCE_TARGET_NODES)}.")
    result["target_nodes"] = nodes

    # Which of the project's options the quote mentions, and whether it reads like a per-candidate result table.
    hits = _option_hits(quote_norm, variables)
    result["matched_options"] = hits
    if any(len(options) >= 2 for options in hits.values()) and _PERCENT.search(summary + " " + quote):
        found.warn(
            "lookup_like",
            "The quote gives results for several of this project's options; keep it as advisory context, "
            "not a table to read candidates off.",
        )

    if same_paper_card_ids:
        shown = ", ".join(same_paper_card_ids[:4]) + (" ..." if len(same_paper_card_ids) > 4 else "")
        found.warn(
            "paper_already_cited",
            f"{len(same_paper_card_ids)} existing card(s) already cite this paper ({shown}); "
            "accept this finding only if it adds something they don't say.",
        )

    card_id = result["card_id"]
    if card_id and card_id in taken_card_ids:
        result["status"] = "duplicate"
        found.warn("duplicate_card_id", f"card_id `{card_id}` already exists in the project.")
    elif quote_norm and any(
        len(quote_norm) >= MIN_DUPLICATE_CHARS and (quote_norm in old or old in quote_norm)
        for old in existing_excerpts
        if len(old) >= MIN_DUPLICATE_CHARS
    ):
        result["status"] = "duplicate"
        found.warn("duplicate_quote", "An existing card already quotes this passage.")
    else:
        result["status"] = "rejected" if found.errors else "ok"
    result["errors"] = found.errors
    result["warnings"] = found.warnings
    return result


def cards_citing(store: EvidenceStore, doi: str) -> list[str]:
    """Ids of existing cards whose doi field or source text carries this DOI."""
    wanted = doi.strip().lower().rstrip(".")
    if not wanted:
        return []
    return [
        card.card_id
        for card in store.cards
        if wanted == card.doi.strip().lower().rstrip(".") or wanted in card.source.lower()
    ]


def read_drafts(project_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """All drafts in evidence_work/drafts/*.jsonl, plus any lines that are not a JSON object."""
    drafts, problems = [], []
    for path in sorted((work_dir(project_dir) / "drafts").glob("*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("not a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                problems.append({"file": path.name, "line": str(number), "message": f"not a JSON object ({exc})"})
                continue
            drafts.append({**payload, "_draft_file": path.name, "_draft_line": number})
    return drafts, problems


def verify_drafts(project_dir: str | Path) -> dict[str, Any]:
    """Check every draft in evidence_work/drafts/ against the extracted paper text.

    Writes `verified.jsonl` (one line per draft, with status, errors and warnings) and
    returns a summary. A repeated finding_id within a source is an error on the later one.
    """
    project = LabProject(Path(project_dir))
    base = work_dir(project.project_dir)
    context_path = base / "context.json"
    if not context_path.exists():
        raise LiteratureError("No evidence_work/context.json; run evidence-prepare first.")
    context = json.loads(context_path.read_text(encoding="utf-8"))
    ready = {item["source_id"] for item in read_manifest(project.project_dir) if item.get("status") == "ready"}
    store = EvidenceStore.load(project.evidence_path)
    taken = {card.card_id for card in store.cards}
    existing_excerpts = [normalize_text(card.supporting_excerpt) for card in store.cards]
    dois = {item["source_id"]: item.get("doi", "") for item in read_manifest(project.project_dir)}
    drafts, unreadable = read_drafts(project.project_dir)
    text_cache: dict[str, dict[str, list[str]] | None] = {}
    seen: dict[tuple[str, str], str] = {}
    records = []
    for draft in drafts:
        source_id = str(draft.get("source_id") or "").strip()
        if source_id not in text_cache:
            text_cache[source_id] = load_page_texts(project.project_dir, source_id) if source_id in ready else None
        record = check_draft(
            draft,
            texts=text_cache[source_id],
            context=context,
            taken_card_ids=taken,
            existing_excerpts=existing_excerpts,
            same_paper_card_ids=cards_citing(store, dois.get(source_id, "")),
        )
        key = (record["source_id"], record["finding_id"])
        if record["finding_id"] and key in seen:
            record["errors"].append(
                {"code": "repeated_finding_id", "message": f"finding_id repeats {seen[key]}; ids must be unique per source."}
            )
            record["status"] = "rejected"
        seen.setdefault(key, f"{draft['_draft_file']} line {draft['_draft_line']}")
        records.append(record)
    _write_jsonl(base / "verified.jsonl", records)
    by_code: Counter[str] = Counter()
    for record in records:
        by_code.update(item["code"] for item in record["errors"] + record["warnings"])
    return {
        "verified_path": str(base / "verified.jsonl"),
        "checked": len(records),
        "ok": sum(item["status"] == "ok" for item in records),
        "rejected": sum(item["status"] == "rejected" for item in records),
        "duplicate": sum(item["status"] == "duplicate" for item in records),
        "with_warnings": sum(bool(item["warnings"]) for item in records if item["status"] == "ok"),
        "unreadable_lines": unreadable,
        "by_code": dict(by_code.most_common()),
        "records": records,
    }
