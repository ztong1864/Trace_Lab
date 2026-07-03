"""Typed schema for parsed documents and chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedPage:
    page_number: int
    markdown: str = ""
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    doc_id: str
    source_file: str
    title: str = ""
    pages: list[ParsedPage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkRecord:
    id: str
    doc_id: str
    source_file: str
    title: str
    section: str
    page_number: int | None
    chunk_text: str
    dataset_scope: list[str] = field(default_factory=lambda: ["*"])
    variable_tags: list[str] = field(default_factory=list)
    trigger_tags: list[str] = field(default_factory=list)
    source_type: str = "document_chunk"
    citation_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
