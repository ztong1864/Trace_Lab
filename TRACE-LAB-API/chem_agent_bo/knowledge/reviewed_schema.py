"""Schema for reviewed domain knowledge entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chem_agent_bo.knowledge.schema import KnowledgeSnippet


def _normalize_tags(value: Any, *, fallback: list[str] | None = None) -> list[str]:
    if value is None:
        return list(fallback or [])
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else list(fallback or [])
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else list(fallback or [])


@dataclass
class ReviewedKnowledgeEntry:
    """Canonical reviewed knowledge entry used in milestone 3."""

    knowledge_id: str
    title: str
    knowledge_type: str
    dataset_scope: list[str] = field(default_factory=lambda: ["*"])
    reaction_types: list[str] = field(default_factory=lambda: ["*"])
    target_nodes: list[str] = field(default_factory=lambda: ["*"])
    variable_tags: list[str] = field(default_factory=list)
    trigger_tags: list[str] = field(default_factory=list)
    assertion: str = ""
    operational_guidance: str = ""
    confidence: float = 0.5
    review_status: str = "draft"
    content_status: str = "canonical"
    content_origin: str = "unknown"
    reviewer: str = ""
    reviewed_at: str = ""
    revision_of: str = ""
    benchmark_safety: str = "unspecified"
    applicability_conditions: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    notes: str = ""
    source_file: str = ""

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        source_file: str,
        notes: str = "",
    ) -> "ReviewedKnowledgeEntry":
        knowledge_id = str(
            payload.get("knowledge_id")
            or payload.get("id")
            or ""
        ).strip()
        if not knowledge_id:
            raise ValueError("ReviewedKnowledgeEntry requires knowledge_id or id.")
        title = str(payload.get("title") or knowledge_id).strip()
        assertion = str(payload.get("assertion") or "").strip()
        operational_guidance = str(payload.get("operational_guidance") or "").strip()
        confidence_raw = payload.get("confidence", 0.5)
        try:
            confidence = max(0.0, min(1.0, float(confidence_raw)))
        except (TypeError, ValueError):
            confidence = 0.5
        return cls(
            knowledge_id=knowledge_id,
            title=title,
            knowledge_type=str(payload.get("knowledge_type") or "reviewed_domain_knowledge"),
            dataset_scope=_normalize_tags(payload.get("dataset_scope"), fallback=["*"]),
            reaction_types=_normalize_tags(payload.get("reaction_types"), fallback=["*"]),
            target_nodes=_normalize_tags(payload.get("target_nodes"), fallback=["*"]),
            variable_tags=_normalize_tags(payload.get("variable_tags")),
            trigger_tags=_normalize_tags(payload.get("trigger_tags")),
            assertion=assertion,
            operational_guidance=operational_guidance,
            confidence=confidence,
            review_status=str(payload.get("review_status") or "draft").strip().lower(),
            content_status=str(payload.get("content_status") or "canonical").strip().lower(),
            content_origin=str(payload.get("content_origin") or "unknown").strip().lower(),
            reviewer=str(payload.get("reviewer") or "").strip(),
            reviewed_at=str(payload.get("reviewed_at") or "").strip(),
            revision_of=str(payload.get("revision_of") or "").strip(),
            benchmark_safety=str(payload.get("benchmark_safety") or "unspecified").strip().lower(),
            applicability_conditions=_normalize_tags(payload.get("applicability_conditions")),
            evidence_refs=_normalize_tags(payload.get("evidence_refs")),
            notes=notes.strip(),
            source_file=str(source_file),
        )

    def matches_dataset(self, dataset: str) -> bool:
        dataset = str(dataset).strip().lower()
        scope = {item.lower() for item in self.dataset_scope}
        return "*" in scope or not dataset or dataset in scope

    def matches_reaction_type(self, reaction_type: str) -> bool:
        reaction_type = str(reaction_type).strip().lower()
        scope = {item.lower() for item in self.reaction_types}
        return "*" in scope or not reaction_type or reaction_type in scope

    def matches_target_node(self, node_name: str) -> bool:
        node_name = str(node_name).strip().lower()
        scope = {item.lower() for item in self.target_nodes}
        return "*" in scope or node_name in scope

    def render_content(self) -> str:
        parts = [f"Title: {self.title}"]
        if self.assertion:
            parts.append(f"Assertion: {self.assertion}")
        if self.operational_guidance:
            parts.append(f"Operational guidance: {self.operational_guidance}")
        if self.evidence_refs:
            parts.append("Evidence refs: " + "; ".join(self.evidence_refs))
        if self.notes:
            parts.append(f"Notes: {self.notes}")
        return "\n".join(parts)

    def to_snippet(
        self,
        *,
        score: float,
        retrieval_reason: str,
    ) -> KnowledgeSnippet:
        return KnowledgeSnippet(
            id=self.knowledge_id,
            content=self.render_content(),
            knowledge_type=self.knowledge_type,
            source_type="reviewed_entry",
            confidence=self.confidence,
            score=score,
            source_ref={
                "source_file": self.source_file,
                "knowledge_id": self.knowledge_id,
                "title": self.title,
            },
            metadata={
                "dataset_scope": list(self.dataset_scope),
                "reaction_types": list(self.reaction_types),
                "target_nodes": list(self.target_nodes),
                "variable_tags": list(self.variable_tags),
                "trigger_tags": list(self.trigger_tags),
                "review_status": self.review_status,
                "content_status": self.content_status,
                "content_origin": self.content_origin,
                "reviewer": self.reviewer,
                "reviewed_at": self.reviewed_at,
                "revision_of": self.revision_of,
                "benchmark_safety": self.benchmark_safety,
                "applicability_conditions": list(self.applicability_conditions),
                "evidence_refs": list(self.evidence_refs),
                "source_file": self.source_file,
                "title": self.title,
            },
            retrieval_reason=retrieval_reason,
        )

    @property
    def source_path(self) -> Path:
        return Path(self.source_file)
