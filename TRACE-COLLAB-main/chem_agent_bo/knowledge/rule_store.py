"""Rule-oriented semi-structured knowledge storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chem_agent_bo.knowledge.schema import KnowledgeSnippet, KnowledgeUnit


class RuleStore:
    """Load and filter curated rule knowledge from local JSONL files."""

    def __init__(self, rules_dir: str) -> None:
        self.rules_dir = Path(rules_dir)
        self._rules: list[KnowledgeUnit] = []

    def reload(self) -> int:
        self._rules = []
        if not self.rules_dir.exists():
            return 0
        for path in sorted(self.rules_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    raw = json.loads(text)
                    # `text` is allowed as alias for `content`.
                    if "content" not in raw and "text" in raw:
                        raw["content"] = raw["text"]
                    self._rules.append(KnowledgeUnit.from_dict(raw))
        return len(self._rules)

    def search(
        self,
        *,
        dataset: str,
        trigger_tags: list[str],
        variable_tags: list[str],
        top_k: int = 5,
    ) -> list[KnowledgeSnippet]:
        top_k = max(1, int(top_k))
        dataset = dataset.strip().lower()
        trigger_set = {tag.strip().lower() for tag in trigger_tags if tag.strip()}
        variable_set = {tag.strip().lower() for tag in variable_tags if tag.strip()}
        scored: list[tuple[float, KnowledgeUnit]] = []
        for rule in self._rules:
            scope = {x.strip().lower() for x in rule.dataset_scope if x.strip()}
            if "*" not in scope and dataset and dataset not in scope:
                continue
            score = 0.0
            matched_trigger = trigger_set.intersection({x.lower() for x in rule.trigger_tags})
            matched_variable = variable_set.intersection({x.lower() for x in rule.variable_tags})
            score += 1.5 * len(matched_trigger)
            score += 1.0 * len(matched_variable)
            score += 0.5 * float(rule.confidence)
            if score <= 0:
                # keep globally-scoped generic rules as backup
                score = 0.1 * float(rule.confidence)
            scored.append((score, rule))
        scored.sort(key=lambda item: item[0], reverse=True)
        snippets: list[KnowledgeSnippet] = []
        for score, rule in scored[:top_k]:
            snippets.append(
                KnowledgeSnippet(
                    id=rule.id,
                    content=rule.content,
                    knowledge_type=str(rule.metadata.get("knowledge_type", "rule")),
                    source_type=rule.source_type,
                    confidence=rule.confidence,
                    score=float(score),
                    source_ref={"rule_id": rule.id, "dataset_scope": rule.dataset_scope},
                    metadata=rule.metadata,
                    retrieval_reason="rule_filter_match",
                )
            )
        return snippets
