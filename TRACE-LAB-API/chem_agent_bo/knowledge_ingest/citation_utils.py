"""Utilities for citation labels and source references."""

from __future__ import annotations


def build_citation_label(source_file: str, page_number: int | None, section: str = "") -> str:
    base = source_file.rsplit("/", 1)[-1]
    page = f"p.{page_number}" if page_number is not None else "p.?"
    sec = section.strip()
    if sec:
        return f"{base} {page} [{sec}]"
    return f"{base} {page}"
