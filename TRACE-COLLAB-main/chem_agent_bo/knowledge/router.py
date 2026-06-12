"""Route retrieval across rule/doc/long-term memory layers."""

from __future__ import annotations

from typing import Any

from chem_agent_bo.knowledge.document_store import DocumentStore
from chem_agent_bo.knowledge.rule_store import RuleStore
from chem_agent_bo.knowledge.schema import KnowledgeSnippet
from chem_agent_bo.memory.long_term_store import LongTermMemoryStore


class KnowledgeRouter:
    """Router that merges layered knowledge sources into one snippet list."""

    def __init__(
        self,
        *,
        rule_store: RuleStore | None = None,
        document_store: DocumentStore | None = None,
        long_term_store: LongTermMemoryStore | None = None,
        default_top_k: int = 5,
    ) -> None:
        self.rule_store = rule_store
        self.document_store = document_store
        self.long_term_store = long_term_store
        self.default_top_k = max(1, int(default_top_k))

    def route_and_retrieve(
        self,
        *,
        rule_query: str,
        document_query: str,
        filters: dict[str, Any],
        route: str = "auto",
        enable_document_retrieval: bool = True,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        limit = self.default_top_k if top_k is None else max(1, int(top_k))
        dataset = str(filters.get("dataset", "")).strip()
        trigger_tags = [str(x) for x in filters.get("trigger_tags", [])]
        variable_tags = [str(x) for x in filters.get("variable_tags", [])]

        use_rules = route in {"auto", "rule", "hybrid"}
        use_docs = route in {"auto", "document", "hybrid"} and enable_document_retrieval
        use_memory = route in {"auto", "memory", "hybrid"}

        retrieved_rules: list[KnowledgeSnippet] = []
        retrieved_docs: list[KnowledgeSnippet] = []
        retrieved_memory: list[KnowledgeSnippet] = []
        if use_rules and self.rule_store is not None:
            retrieved_rules = self.rule_store.search(
                dataset=dataset,
                trigger_tags=trigger_tags,
                variable_tags=variable_tags,
                top_k=limit,
            )
        if use_docs and self.document_store is not None:
            retrieved_docs = self.document_store.search(
                document_query,
                filters=filters,
                top_k=limit,
            )
        if use_memory and self.long_term_store is not None:
            memory_rows = self.long_term_store.search(
                ("promoted_experience",),
                query=rule_query,
                filter_dict={"dataset": dataset} if dataset else None,
                limit=limit,
            )
            for idx, item in enumerate(memory_rows):
                retrieved_memory.append(
                    KnowledgeSnippet(
                        id=str(item.get("id", f"memory_{idx}")),
                        content=str(item.get("reflection_insight", item.get("content", ""))),
                        knowledge_type="memory",
                        source_type="long_term_memory",
                        confidence=0.6,
                        score=0.3,
                        source_ref={"namespace": "promoted_experience"},
                        metadata=dict(item),
                        retrieval_reason="memory_search",
                    )
                )
        snippets = [*retrieved_rules, *retrieved_docs, *retrieved_memory]

        snippets.sort(
            key=lambda s: (
                float(s.score),
                float(s.confidence),
                1.0 if s.knowledge_type == "rule" else 0.0,
            ),
            reverse=True,
        )
        injected = snippets[:limit]
        return {
            "retrieved": {
                "rule": retrieved_rules,
                "document": retrieved_docs,
                "memory": retrieved_memory,
            },
            "injected": injected,
        }
