"""Thin LlamaParse client wrapper for PDF parsing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from chem_agent_bo.knowledge_ingest.document_schema import ParsedDocument, ParsedPage

try:
    from llama_cloud import AsyncLlamaCloud
except Exception:  # noqa: BLE001
    AsyncLlamaCloud = None


class LlamaParseClient:
    """Client wrapper that returns ParsedDocument objects."""

    def __init__(self, api_key: str | None = None, tier: str = "agentic") -> None:
        if AsyncLlamaCloud is None:
            raise RuntimeError(
                "llama-cloud is not installed. Install with "
                "`pip install --target .vendor_py llama-cloud`."
            )
        if api_key:
            os.environ["LLAMA_CLOUD_API_KEY"] = api_key
        if not os.getenv("LLAMA_CLOUD_API_KEY"):
            raise RuntimeError(
                "LLAMA_CLOUD_API_KEY is missing. "
                "Set env var LLAMA_CLOUD_API_KEY before parsing."
            )
        self.client = AsyncLlamaCloud()
        self.tier = tier

    def _resolve_expand(self, expand: list[str] | None) -> list[str]:
        requested = list(expand or ["markdown", "text", "metadata", "items"])
        if self.tier == "fast":
            unsupported = {"markdown", "items", "markdown_content_metadata", "items_content_metadata"}
            return [x for x in requested if x not in unsupported]
        return requested

    async def parse_pdf(self, pdf_path: str, *, expand: list[str] | None = None) -> ParsedDocument:
        resolved_expand = self._resolve_expand(expand)
        path = Path(pdf_path)
        file_obj = await self.client.files.create(file=str(path), purpose="parse")
        result = await self.client.parsing.parse(
            file_id=file_obj.id,
            tier=self.tier,
            version="latest",
            expand=resolved_expand,
        )
        pages: list[ParsedPage] = []
        markdown_pages = getattr(getattr(result, "markdown", None), "pages", []) or []
        text_pages = getattr(getattr(result, "text", None), "pages", []) or []
        meta_pages = getattr(getattr(result, "metadata", None), "pages", []) or []
        items_pages = getattr(getattr(result, "items", None), "pages", []) or []
        page_count = max(len(markdown_pages), len(text_pages), len(meta_pages))
        for idx in range(page_count):
            md_page = markdown_pages[idx] if idx < len(markdown_pages) else None
            txt_page = text_pages[idx] if idx < len(text_pages) else None
            meta_page = meta_pages[idx] if idx < len(meta_pages) else None
            items_page = items_pages[idx] if idx < len(items_pages) else None
            page_number = int(getattr(meta_page, "page_number", idx + 1))
            metadata: dict[str, Any] = {}
            if meta_page is not None:
                metadata = {
                    "page_number": getattr(meta_page, "page_number", None),
                    "confidence": getattr(meta_page, "confidence", None),
                }
            if items_page is not None:
                metadata["items_count"] = len(getattr(items_page, "items", []) or [])
            pages.append(
                ParsedPage(
                    page_number=page_number,
                    markdown=str(getattr(md_page, "markdown", "") or ""),
                    text=str(getattr(txt_page, "text", "") or ""),
                    metadata=metadata,
                )
            )
        doc_id = path.stem.replace(" ", "_")
        return ParsedDocument(
            doc_id=doc_id,
            source_file=str(path),
            title=path.stem,
            pages=pages,
            metadata={"tier": self.tier, "expand": resolved_expand},
        )
