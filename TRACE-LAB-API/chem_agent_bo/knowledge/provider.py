"""Unified retrieval provider for layered knowledge system."""

from __future__ import annotations

from typing import Any

from chem_agent_bo.knowledge.document_store import DocumentStore, LanceDocumentStore
from chem_agent_bo.knowledge.router import KnowledgeRouter
from chem_agent_bo.knowledge.schema import KnowledgeSnippet
from chem_agent_bo.knowledge.store import LanceDBStore


class KnowledgeProvider:
    """Fetch prompt-ready snippets from routed layered sources."""

    def __init__(
        self,
        *,
        router: KnowledgeRouter | None = None,
        document_store: DocumentStore | None = None,
        store: LanceDBStore | None = None,
        default_top_k: int = 5,
    ) -> None:
        if router is not None:
            self.router = router
        else:
            fallback_doc_store = document_store
            if fallback_doc_store is None and store is not None:
                fallback_doc_store = LanceDocumentStore(store)
            self.router = KnowledgeRouter(document_store=fallback_doc_store, default_top_k=default_top_k)
        self.default_top_k = max(1, int(default_top_k))

    def get_context_snippets(
        self,
        rule_query: str,
        document_query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        route: str = "auto",
        enable_document_retrieval: bool = True,
    ) -> dict[str, Any]:
        rule_query = (rule_query or "").strip()
        document_query = (document_query or "").strip()
        if not rule_query and not document_query:
            return {
                "injected_units": [],
                "retrieved_by_source": {"rule": 0, "document": 0, "memory": 0},
                "injected_by_source": {},
            }
        limit = self.default_top_k if top_k is None else max(1, int(top_k))
        result = self.router.route_and_retrieve(
            rule_query=rule_query,
            document_query=document_query,
            filters=filters or {},
            route=route,
            enable_document_retrieval=enable_document_retrieval,
            top_k=limit,
        )
        retrieved = result.get("retrieved", {})
        injected = result.get("injected", [])
        rows = [self._to_prompt_item(item) for item in injected]
        retrieved_units_by_source = {
            "rule": [self._to_prompt_item(item) for item in retrieved.get("rule", [])],
            "document": [self._to_prompt_item(item) for item in retrieved.get("document", [])],
            "memory": [self._to_prompt_item(item) for item in retrieved.get("memory", [])],
        }
        retrieved_by_source = {
            "rule": len(retrieved.get("rule", [])),
            "document": len(retrieved.get("document", [])),
            "memory": len(retrieved.get("memory", [])),
        }
        injected_by_source: dict[str, int] = {}
        for row in rows:
            source = str(row.get("source_type", "unknown"))
            injected_by_source[source] = injected_by_source.get(source, 0) + 1
        return {
            "injected_units": rows,
            "retrieved_units_by_source": retrieved_units_by_source,
            "retrieved_by_source": retrieved_by_source,
            "injected_by_source": injected_by_source,
        }

    @staticmethod
    def _to_prompt_item(item: KnowledgeSnippet) -> dict[str, Any]:
        return item.to_prompt_item()

    def get_experience_candidates(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        limit = self.default_top_k if top_k is None else max(1, int(top_k))
        results = self.router.route_and_retrieve(
            rule_query=query,
            document_query=query,
            filters=filters or {},
            route="memory",
            enable_document_retrieval=False,
            top_k=limit,
        )
        injected = results.get("injected", [])
        items: list[dict[str, Any]] = []
        for row in injected:
            items.append(row.to_prompt_item())
        return items
