"""Reviewed, benchmark-safe experience entries for milestone 4 diagnosis reads."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chem_agent_bo.knowledge.schema import KnowledgeSnippet

_RETIRED_CONTENT_STATUSES = {"deprecated", "retired"}


def _as_list(value: Any, *, fallback: list[str] | None = None) -> list[str]:
    if value is None:
        return list(fallback or [])
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else list(fallback or [])
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else list(fallback or [])


def _nested_value(payload: dict[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _int_from_state(payload: dict[str, Any], *paths: tuple[str, ...], default: int = 0) -> int:
    for path in paths:
        raw = _nested_value(payload, *path)
        try:
            if raw is None or str(raw).strip() == "":
                continue
            return int(raw)
        except (TypeError, ValueError):
            continue
    return int(default)


@dataclass
class ReviewedExperienceEntry:
    experience_id: str
    title: str
    dataset_scope: list[str] = field(default_factory=lambda: ["*"])
    target_nodes: list[str] = field(default_factory=lambda: ["stagnation_diagnosis"])
    trigger_tags: list[str] = field(default_factory=list)
    diagnosis_types: list[str] = field(default_factory=list)
    controller_modes: list[str] = field(default_factory=list)
    generalized_form: str = ""
    review_status: str = "draft"
    content_status: str = "reviewed_curated"
    benchmark_safety: str = "safe_abstraction"
    confidence: float = 0.5
    evidence_refs: list[str] = field(default_factory=list)
    min_no_improvement_rounds: int | None = None
    max_no_improvement_rounds: int | None = None
    min_remaining_budget: int | None = None
    max_remaining_budget: int | None = None
    require_recent_breakthrough: bool = False
    source_file: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, source_file: str) -> "ReviewedExperienceEntry":
        experience_id = str(payload.get("experience_id") or payload.get("id") or "").strip()
        if not experience_id:
            raise ValueError("ReviewedExperienceEntry requires experience_id")
        confidence_raw = payload.get("confidence", 0.5)
        try:
            confidence = max(0.0, min(1.0, float(confidence_raw)))
        except (TypeError, ValueError):
            confidence = 0.5
        def _int_or_none(raw: Any) -> int | None:
            try:
                if raw is None or str(raw).strip() == "":
                    return None
                return int(raw)
            except (TypeError, ValueError):
                return None
        return cls(
            experience_id=experience_id,
            title=str(payload.get("title") or experience_id).strip(),
            dataset_scope=_as_list(payload.get("dataset_scope"), fallback=["*"]),
            target_nodes=_as_list(payload.get("target_nodes"), fallback=["stagnation_diagnosis"]),
            trigger_tags=_as_list(payload.get("trigger_tags")),
            diagnosis_types=_as_list(payload.get("diagnosis_types")),
            controller_modes=_as_list(payload.get("controller_modes")),
            generalized_form=str(payload.get("generalized_form") or "").strip(),
            review_status=str(payload.get("review_status") or "draft").strip().lower(),
            content_status=str(payload.get("content_status") or "reviewed_curated").strip().lower(),
            benchmark_safety=str(payload.get("benchmark_safety") or "safe_abstraction").strip().lower(),
            confidence=confidence,
            evidence_refs=_as_list(payload.get("evidence_refs")),
            min_no_improvement_rounds=_int_or_none(payload.get("min_no_improvement_rounds")),
            max_no_improvement_rounds=_int_or_none(payload.get("max_no_improvement_rounds")),
            min_remaining_budget=_int_or_none(payload.get("min_remaining_budget")),
            max_remaining_budget=_int_or_none(payload.get("max_remaining_budget")),
            require_recent_breakthrough=bool(payload.get("require_recent_breakthrough", False)),
            source_file=source_file,
        )

    def matches_dataset(self, dataset: str) -> bool:
        scope = {item.lower() for item in self.dataset_scope}
        key = str(dataset).strip().lower()
        return "*" in scope or not key or key in scope

    def matches_target_node(self, node_name: str) -> bool:
        scope = {item.lower() for item in self.target_nodes}
        key = str(node_name).strip().lower()
        return "*" in scope or key in scope

    def to_snippet(self, *, score: float) -> KnowledgeSnippet:
        content = "\n".join(
            [
                f"Title: {self.title}",
                f"Generalized lesson: {self.generalized_form}",
                "Use this as reviewed experience, not as an answer-level hint.",
                ("Evidence refs: " + "; ".join(self.evidence_refs)) if self.evidence_refs else "",
            ]
        ).strip()
        return KnowledgeSnippet(
            id=self.experience_id,
            content=content,
            knowledge_type="reviewed_experience",
            source_type="reviewed_experience",
            confidence=self.confidence,
            score=score,
            source_ref={
                "source_file": self.source_file,
                "experience_id": self.experience_id,
                "title": self.title,
            },
            metadata={
                "dataset_scope": list(self.dataset_scope),
                "target_nodes": list(self.target_nodes),
                "trigger_tags": list(self.trigger_tags),
                "diagnosis_types": list(self.diagnosis_types),
                "controller_modes": list(self.controller_modes),
                "review_status": self.review_status,
                "content_status": self.content_status,
                "benchmark_safety": self.benchmark_safety,
                "evidence_refs": list(self.evidence_refs),
                "min_no_improvement_rounds": self.min_no_improvement_rounds,
                "max_no_improvement_rounds": self.max_no_improvement_rounds,
                "min_remaining_budget": self.min_remaining_budget,
                "max_remaining_budget": self.max_remaining_budget,
                "require_recent_breakthrough": self.require_recent_breakthrough,
            },
            retrieval_reason="reviewed_experience_match",
        )


class ReviewedExperienceStore:
    """Load reviewed generalizable experience for diagnosis-time reads."""

    def __init__(self, experience_dir: str | Path) -> None:
        self.experience_dir = Path(experience_dir)
        self._entries: list[ReviewedExperienceEntry] = []

    def reload(self) -> int:
        self._entries = []
        if not self.experience_dir.exists():
            return 0
        for path in sorted(self.experience_dir.rglob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                try:
                    entry = ReviewedExperienceEntry.from_dict(payload, source_file=str(path))
                except ValueError:
                    continue
                self._entries.append(entry)
        return len(self._entries)

    def search(
        self,
        *,
        node_name: str,
        dataset: str,
        trigger_tags: list[str] | None = None,
        node_state_view: dict[str, Any] | None = None,
        top_k: int = 2,
    ) -> list[KnowledgeSnippet]:
        trigger_set = {str(item).strip().lower() for item in (trigger_tags or []) if str(item).strip()}
        node_state_view = node_state_view or {}
        no_improvement_rounds = _int_from_state(
            node_state_view,
            ("no_improvement_rounds",),
            ("optimization_status", "no_improvement_rounds"),
            default=0,
        )
        remaining_budget = _int_from_state(
            node_state_view,
            ("remaining_budget",),
            ("run_identity", "remaining_budget"),
            default=0,
        )
        recent_trace = ((node_state_view.get("recent_decision_trace") or {}).get("items") or [])
        recent_breakthrough = any(
            bool(item.get("improved_best"))
            for item in recent_trace[-3:]
            if isinstance(item, dict)
        )
        scored: list[tuple[float, ReviewedExperienceEntry]] = []
        for entry in self._entries:
            if entry.review_status != "approved":
                continue
            if entry.content_status in _RETIRED_CONTENT_STATUSES:
                continue
            if entry.content_status not in {"reviewed_curated", "revised_curated"}:
                continue
            if entry.benchmark_safety not in {"safe_abstraction", "approved_generalizable"}:
                continue
            if not entry.matches_dataset(dataset):
                continue
            if not entry.matches_target_node(node_name):
                continue
            if (
                entry.min_no_improvement_rounds is not None
                and no_improvement_rounds < entry.min_no_improvement_rounds
            ):
                continue
            if (
                entry.max_no_improvement_rounds is not None
                and no_improvement_rounds > entry.max_no_improvement_rounds
            ):
                continue
            if (
                entry.min_remaining_budget is not None
                and remaining_budget < entry.min_remaining_budget
            ):
                continue
            if (
                entry.max_remaining_budget is not None
                and remaining_budget > entry.max_remaining_budget
            ):
                continue
            if entry.require_recent_breakthrough and not recent_breakthrough:
                continue
            score = 1.5 + 0.25 * float(entry.confidence)
            overlap = trigger_set.intersection({item.lower() for item in entry.trigger_tags})
            score += 0.5 * len(overlap)
            if entry.min_no_improvement_rounds is not None:
                score += 0.15
            if entry.max_no_improvement_rounds is not None:
                score += 0.10
            if entry.require_recent_breakthrough and recent_breakthrough:
                score += 0.2
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry.to_snippet(score=score) for score, entry in scored[: max(1, int(top_k))]]

    def titles(self) -> list[str]:
        return [entry.title for entry in self._entries if entry.review_status == "approved"]
