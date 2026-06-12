"""Reviewed knowledge store backed by markdown cards with frontmatter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from chem_agent_bo.knowledge.reviewed_schema import ReviewedKnowledgeEntry
from chem_agent_bo.knowledge.schema import KnowledgeSnippet


_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.DOTALL)
_RETIRED_CONTENT_STATUSES = {"deprecated", "retired"}


class ReviewedKnowledgeStore:
    """Load and score canonical reviewed knowledge entries."""

    def __init__(self, knowledge_dir: str | Path) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self._entries: list[ReviewedKnowledgeEntry] = []

    def reload(self) -> int:
        self._entries = []
        if not self.knowledge_dir.exists():
            return 0
        for path in sorted(self.knowledge_dir.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            match = _FRONTMATTER_RE.match(text)
            if match is None:
                continue
            frontmatter_text, body = match.groups()
            payload = yaml.safe_load(frontmatter_text) or {}
            if not isinstance(payload, dict):
                continue
            try:
                entry = ReviewedKnowledgeEntry.from_payload(
                    payload,
                    source_file=str(path),
                    notes=body.strip(),
                )
            except ValueError:
                continue
            self._entries.append(entry)
        return len(self._entries)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9_]+", text.lower())
            if len(token) >= 3
        }

    def search(
        self,
        *,
        node_name: str,
        query: str,
        dataset: str,
        reaction_type: str,
        variable_tags: list[str] | None = None,
        trigger_tags: list[str] | None = None,
        top_k: int = 3,
        allowed_content_statuses: list[str] | None = None,
        retrieval_reason: str = "reviewed_knowledge_match",
    ) -> list[KnowledgeSnippet]:
        query_tokens = self._tokenize(query)
        variable_set = {str(item).strip().lower() for item in (variable_tags or []) if str(item).strip()}
        trigger_set = {str(item).strip().lower() for item in (trigger_tags or []) if str(item).strip()}
        allowed_statuses = {
            str(item).strip().lower()
            for item in (allowed_content_statuses or [])
            if str(item).strip()
        }
        scored: list[tuple[float, ReviewedKnowledgeEntry]] = []
        for entry in self._entries:
            if entry.review_status != "approved":
                continue
            if entry.content_status in _RETIRED_CONTENT_STATUSES:
                continue
            if allowed_statuses and entry.content_status not in allowed_statuses:
                continue
            if not entry.matches_dataset(dataset):
                continue
            if not entry.matches_reaction_type(reaction_type):
                continue
            if not entry.matches_target_node(node_name):
                continue
            score = 0.0
            score += 1.6
            if dataset and dataset.lower() in {item.lower() for item in entry.dataset_scope}:
                score += 1.0
            if reaction_type and reaction_type.lower() in {item.lower() for item in entry.reaction_types}:
                score += 0.8
            score += 0.35 * len(variable_set.intersection({item.lower() for item in entry.variable_tags}))
            score += 0.50 * len(trigger_set.intersection({item.lower() for item in entry.trigger_tags}))
            entry_tokens = self._tokenize(
                " ".join(
                    [
                        entry.title,
                        entry.assertion,
                        entry.operational_guidance,
                        entry.notes,
                    ]
                )
            )
            score += 0.04 * len(query_tokens.intersection(entry_tokens))
            score += 0.20 * float(entry.confidence)
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        snippets: list[KnowledgeSnippet] = []
        for score, entry in scored[: max(1, int(top_k))]:
            snippets.append(
                entry.to_snippet(
                    score=float(score),
                    retrieval_reason=retrieval_reason,
                )
            )
        return snippets

    def search_partitioned(
        self,
        *,
        node_name: str,
        query: str,
        dataset: str,
        reaction_type: str,
        variable_tags: list[str] | None = None,
        trigger_tags: list[str] | None = None,
        top_k: int = 3,
        allowed_content_statuses: list[str] | None = None,
    ) -> tuple[list[KnowledgeSnippet], list[KnowledgeSnippet]]:
        allowed_statuses = [
            str(item).strip().lower()
            for item in (allowed_content_statuses or [])
            if str(item).strip()
        ]
        pinned_statuses = [item for item in allowed_statuses if item == "pinned_curated"]
        regular_statuses = [item for item in allowed_statuses if item != "pinned_curated"]
        pinned = self.search(
            node_name=node_name,
            query=query,
            dataset=dataset,
            reaction_type=reaction_type,
            variable_tags=variable_tags,
            trigger_tags=trigger_tags,
            top_k=top_k,
            allowed_content_statuses=pinned_statuses,
            retrieval_reason="reviewed_knowledge_pinned_match",
        ) if pinned_statuses else []
        regular = self.search(
            node_name=node_name,
            query=query,
            dataset=dataset,
            reaction_type=reaction_type,
            variable_tags=variable_tags,
            trigger_tags=trigger_tags,
            top_k=top_k,
            allowed_content_statuses=regular_statuses,
            retrieval_reason="reviewed_knowledge_match",
        ) if regular_statuses else []
        return pinned, regular

    def titles(self) -> list[str]:
        return [entry.title for entry in self._entries if entry.review_status == "approved"]
