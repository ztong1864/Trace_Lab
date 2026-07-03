"""Schema definitions for layered knowledge objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _as_tag_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        tags: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                tags.append(text)
        return tags
    text = str(value).strip()
    return [text] if text else []


@dataclass
class KnowledgeUnit:
    """Backward-compatible unit for lightweight rule-style knowledge."""

    id: str
    content: str
    dataset_scope: list[str] = field(default_factory=lambda: ["*"])
    variable_tags: list[str] = field(default_factory=list)
    trigger_tags: list[str] = field(default_factory=list)
    source_type: str = "local_note"
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KnowledgeUnit":
        unit_id = str(raw.get("id") or "").strip()
        content = str(raw.get("content") or "").strip()
        if not unit_id:
            raise ValueError("KnowledgeUnit requires non-empty 'id'.")
        if not content:
            raise ValueError(f"KnowledgeUnit '{unit_id}' requires non-empty 'content'.")
        confidence_raw = raw.get("confidence", 0.5)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}
        return cls(
            id=unit_id,
            content=content,
            dataset_scope=_as_tag_list(raw.get("dataset_scope")) or ["*"],
            variable_tags=_as_tag_list(raw.get("variable_tags")),
            trigger_tags=_as_tag_list(raw.get("trigger_tags")),
            source_type=str(raw.get("source_type") or "local_note"),
            confidence=confidence,
            metadata=metadata,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "dataset_scope": list(self.dataset_scope),
            "variable_tags": list(self.variable_tags),
            "trigger_tags": list(self.trigger_tags),
            "source_type": self.source_type,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass
class DocumentChunk:
    """Parsed document chunk used for Qdrant/RAG retrieval."""

    id: str
    content: str
    title: str = ""
    section: str = ""
    page_number: int | None = None
    source_file: str = ""
    citation_label: str = ""
    dataset_scope: list[str] = field(default_factory=lambda: ["*"])
    variable_tags: list[str] = field(default_factory=list)
    trigger_tags: list[str] = field(default_factory=list)
    source_type: str = "document_chunk"
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "title": self.title,
            "section": self.section,
            "page_number": self.page_number,
            "source_file": self.source_file,
            "citation_label": self.citation_label,
            "dataset_scope": list(self.dataset_scope),
            "variable_tags": list(self.variable_tags),
            "trigger_tags": list(self.trigger_tags),
            "source_type": self.source_type,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass
class KnowledgeSnippet:
    """Unified snippet contract consumed by prompts."""

    id: str
    content: str
    knowledge_type: str
    source_type: str
    confidence: float = 0.0
    score: float = 0.0
    source_ref: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_reason: str = ""

    def to_prompt_item(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "knowledge_type": self.knowledge_type,
            "source_type": self.source_type,
            "confidence": self.confidence,
            "score": self.score,
            "source_ref": dict(self.source_ref),
            "metadata": dict(self.metadata),
            "retrieval_reason": self.retrieval_reason,
        }
