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

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from chem_agent_bo.lab.evidence import EvidenceStore
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


def extract_pages(pdf_path: str | Path) -> list[str]:
    """Text of each page, keeping the table layout (`pdftotext -layout`)."""
    path = Path(pdf_path)
    if not path.is_file():
        raise LiteratureError(f"No such file: {path}")
    exe = find_pdftotext()
    with tempfile.TemporaryDirectory() as tmp:
        # Some pdftotext builds mishandle non-ASCII paths (paper names carry °, ·, ...), so work on a copy.
        source = Path(tmp) / "paper.pdf"
        shutil.copyfile(path, source)
        completed = subprocess.run(
            [exe, "-layout", "-enc", "UTF-8", str(source), "-"],
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
            {"source_id": source_id, "file": path.name, "sha256": sha, "pages": pages}, ensure_ascii=False
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
        body = "\n".join(f"\n=== PAGE {number} ===\n{pages[number - 1].rstrip()}" for number in range(first, last + 1))
        (base / "packets" / name).write_text(header + body + "\n", encoding="utf-8")
        record["packets"].append(name)
    return record


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


def load_pages(project_dir: str | Path, source_id: str) -> list[str]:
    path = work_dir(project_dir) / "text" / f"{source_id}.json"
    if not path.exists():
        raise LiteratureError(f"No extracted text for source `{source_id}`; run evidence-prepare first.")
    return list(json.loads(path.read_text(encoding="utf-8"))["pages"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
