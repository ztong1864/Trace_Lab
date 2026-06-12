#!/usr/bin/env python
"""Build a local PDF manifest for lab evidence curation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reaction-scope", default="oxidative esterification")
    parser.add_argument("--glob", default="*.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = _resolve(args.source_dir)
    output = _resolve(args.output)
    pdf_paths = sorted(source_dir.glob(args.glob))
    records = [
        _manifest_record(path, reaction_scope=str(args.reaction_scope or ""))
        for path in pdf_paths
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(records)} PDF manifest records to {output}")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _manifest_record(path: Path, *, reaction_scope: str) -> dict[str, Any]:
    stem = path.stem
    source_id = f"labpdf_{_slug(stem)}_{_short_hash(str(path))}"
    info = _pdfinfo(path)
    return {
        "pdf_id": source_id,
        "source_id": source_id,
        "provider": "local_lab_literature",
        "source_type": "local_pdf",
        "status": "downloaded",
        "title": info.get("Title") or stem,
        "url": "",
        "doi": _doi_from_text(info.get("Subject", "") + " " + info.get("Title", "")),
        "local_path": str(path),
        "filename": path.name,
        "retrieved_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "metadata": {
            "reaction_scope": reaction_scope,
            "filename_tags": _filename_tags(stem),
            "temperature_hint": _temperature_hint(stem),
            "metal_hints": _metal_hints(stem),
            "nitroxyl_hints": _nitroxyl_hints(stem),
            "file_size": path.stat().st_size,
            "pdf_pages": _safe_int(info.get("Pages")),
        },
    }


def _pdfinfo(path: Path) -> dict[str, str]:
    try:
        completed = subprocess.run(
            ["pdfinfo", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return {}
    info: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def _filename_tags(stem: str) -> list[str]:
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", stem)
    return [token for token in normalized.split() if token]


def _temperature_hint(stem: str) -> str:
    lowered = stem.lower()
    if re.search(r"(^|[^a-z])rt($|[^a-z])", lowered):
        return "rt"
    match = re.search(r"(\d{2,3})\s*[°c]?", lowered)
    return match.group(1) + " C" if match else ""


def _metal_hints(stem: str) -> list[str]:
    metals = ("Fe", "Pd", "Cu", "Ru", "Ag", "Bi", "Co", "K")
    return [metal for metal in metals if re.search(rf"\b{re.escape(metal)}", stem)]


def _nitroxyl_hints(stem: str) -> list[str]:
    hints = []
    for token in ("4-OH-TEMPO", "TEMPO", "NHPI", "ABNO", "N-Oxyl"):
        if token.lower() in stem.lower():
            hints.append(token)
    return hints


def _doi_from_text(text: str) -> str:
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text)
    return match.group(0) if match else ""


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "pdf"


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _safe_int(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    sys.exit(main())
