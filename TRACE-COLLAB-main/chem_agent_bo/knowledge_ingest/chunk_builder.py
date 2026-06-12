"""Build chunk records from parsed markdown/text pages."""

from __future__ import annotations

import hashlib
import re

from chem_agent_bo.knowledge_ingest.citation_utils import build_citation_label
from chem_agent_bo.knowledge_ingest.document_schema import ChunkRecord, ParsedDocument


def _split_markdown(text: str, chunk_size: int = 900) -> list[str]:
    text = text.strip()
    if not text:
        return []
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(current) + len(block) + 2 <= chunk_size:
            current = f"{current}\n\n{block}".strip()
            continue
        if current:
            chunks.append(current)
        current = block
    if current:
        chunks.append(current)
    return chunks


def _as_tag_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        tags: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                tags.append(text)
        return tags
    text = str(value).strip()
    return [text] if text else []


def build_chunks_from_markdown(
    parsed_doc: ParsedDocument,
    *,
    dataset_scope: list[str] | None = None,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    scopes = dataset_scope or ["*"]
    for page in parsed_doc.pages:
        raw = page.markdown or page.text
        section = str(page.metadata.get("section", "")).strip()
        variable_tags = _as_tag_list(page.metadata.get("variable_tags"))
        trigger_tags = _as_tag_list(page.metadata.get("trigger_tags"))
        page_chunks = _split_markdown(raw)
        for idx, chunk_text in enumerate(page_chunks):
            digest = hashlib.md5(f"{parsed_doc.doc_id}:{page.page_number}:{idx}".encode()).hexdigest()[:12]
            chunk_id = f"{parsed_doc.doc_id}_{digest}"
            citation_label = build_citation_label(parsed_doc.source_file, page.page_number, section=section)
            chunks.append(
                ChunkRecord(
                    id=chunk_id,
                    doc_id=parsed_doc.doc_id,
                    source_file=parsed_doc.source_file,
                    title=parsed_doc.title,
                    section=section,
                    page_number=page.page_number,
                    chunk_text=chunk_text,
                    dataset_scope=list(scopes),
                    variable_tags=variable_tags,
                    trigger_tags=trigger_tags,
                    citation_label=citation_label,
                    metadata={
                        "page_confidence": page.metadata.get("confidence"),
                        "parse_metadata": page.metadata,
                    },
                )
            )
    return chunks
