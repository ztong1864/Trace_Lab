"""Document retrieval backends for layered knowledge system."""

from __future__ import annotations

import re
from typing import Any, Protocol

from chem_agent_bo.knowledge.hybrid_embeddings import QdrantHybridEmbedder
from chem_agent_bo.knowledge.schema import KnowledgeSnippet
from chem_agent_bo.knowledge.store import LanceDBStore

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
except Exception:  # noqa: BLE001
    QdrantClient = None
    qdrant_models = None


class DocumentStore(Protocol):
    """Abstract document retrieval backend."""

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[KnowledgeSnippet]:
        """Return document snippets for a query."""


class LanceDocumentStore:
    """Compatibility adapter: treat LanceDB as document backend."""

    def __init__(self, store: LanceDBStore) -> None:
        self.store = store

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[KnowledgeSnippet]:
        rows = self.store.similarity_search(query, filters=filters, top_k=top_k)
        snippets: list[KnowledgeSnippet] = []
        for row in rows:
            metadata = row.get("metadata", {})
            snippets.append(
                KnowledgeSnippet(
                    id=str(row.get("id", "")),
                    content=str(row.get("content", "")),
                    knowledge_type="document",
                    source_type=str(row.get("source_type", "document_chunk")),
                    confidence=float(row.get("confidence", 0.0)),
                    score=float(row.get("score", 0.0)),
                    source_ref={
                        "source_file": metadata.get("source_file", ""),
                        "page_number": metadata.get("page_number"),
                        "section": metadata.get("section", ""),
                        "title": metadata.get("title", ""),
                    },
                    metadata=metadata,
                    retrieval_reason="document_semantic",
                )
            )
        return snippets


class QdrantDocumentStore:
    """Qdrant backend with optional dense+sparse hybrid query path."""

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        collection_name: str = "chem_documents",
        api_key: str | None = None,
        hybrid_embedder: QdrantHybridEmbedder | None = None,
    ) -> None:
        if QdrantClient is None:
            raise RuntimeError(
                "qdrant-client is not installed. Install with "
                "`pip install --target .vendor_py qdrant-client`."
            )
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key)
        self.hybrid_embedder = hybrid_embedder or QdrantHybridEmbedder()
        self._validate_hybrid_collection()

    def _validate_hybrid_collection(self) -> None:
        if qdrant_models is None:
            return
        info = self.client.get_collection(self.collection_name)
        params = info.config.params
        dense_ok = isinstance(params.vectors, dict) and "dense" in params.vectors
        sparse_ok = isinstance(params.sparse_vectors, dict) and "sparse" in params.sparse_vectors
        if not (dense_ok and sparse_ok):
            raise RuntimeError(
                f"Collection '{self.collection_name}' is not configured as hybrid "
                "(expected named vectors: dense+sparse)."
            )

    @staticmethod
    def _payload_filter(
        filters: dict[str, Any] | None,
        *,
        include_semantic_tags: bool = True,
    ):  # noqa: ANN202
        if qdrant_models is None or not filters:
            return None
        must: list[Any] = []
        dataset = str(filters.get("dataset", "")).strip()
        if dataset:
            must.append(
                qdrant_models.FieldCondition(
                    key="dataset_scope",
                    match=qdrant_models.MatchAny(any=["*", dataset]),
                )
            )
        if include_semantic_tags:
            for field in ("variable_tags", "trigger_tags"):
                values = [str(v).strip() for v in filters.get(field, []) if str(v).strip()]
                if values:
                    must.append(
                        qdrant_models.FieldCondition(
                            key=field,
                            match=qdrant_models.MatchAny(any=values),
                        )
                    )
        if not must:
            return None
        return qdrant_models.Filter(must=must)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {x for x in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(x) >= 3}

    def _rerank_documents(
        self,
        *,
        hits: list[Any],
        query: str,
        filters: dict[str, Any] | None,
    ) -> list[Any]:
        query_tokens = self._tokenize(query)
        variable_tokens = self._tokenize(" ".join(filters.get("variable_tags", []))) if filters else set()
        trigger_tokens = self._tokenize(" ".join(filters.get("trigger_tags", []))) if filters else set()
        trigger_text_hints = {
            "stagnation": {"stagnation", "plateau", "diminishing"},
            "coverage_low": {"coverage", "underexplored", "screening"},
            "duplicate_high": {"diversity", "novelty", "duplicate"},
        }

        ranked: list[tuple[float, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            text_blob = " ".join(
                [
                    str(payload.get("title", "")),
                    str(payload.get("section", "")),
                    str(payload.get("chunk_text", payload.get("content", ""))),
                ]
            )
            text_tokens = self._tokenize(text_blob)
            overlap = len(query_tokens.intersection(text_tokens))
            variable_overlap = len(variable_tokens.intersection(text_tokens))
            trigger_overlap = 0
            for trig in trigger_tokens:
                trigger_overlap += len(trigger_text_hints.get(trig, {trig}).intersection(text_tokens))
            base_score = float(getattr(hit, "score", 0.0) or 0.0)
            final_score = (
                0.7 * base_score
                + 0.2 * float(variable_overlap)
                + 0.1 * float(trigger_overlap)
                + 0.03 * float(overlap)
            )
            ranked.append((final_score, hit))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in ranked]

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[KnowledgeSnippet]:
        top_k = max(1, int(top_k))
        candidate_limit = max(top_k * 6, 24)
        dense_query = self.hybrid_embedder.embed_dense_query(query)
        sparse_indices, sparse_values = self.hybrid_embedder.embed_sparse_query(query)
        # Document retrieval uses relaxed filtering (dataset only) to avoid over-pruning.
        # Fine-grained relevance is recovered by local reranking below.
        payload_filter = self._payload_filter(filters, include_semantic_tags=False)
        prefetch: list[Any] = [
            qdrant_models.Prefetch(
                query=dense_query,
                using="dense",
                limit=candidate_limit,
            )
        ]
        if sparse_indices and sparse_values:
            sparse_query = qdrant_models.SparseVector(indices=sparse_indices, values=sparse_values)
            prefetch.append(
                qdrant_models.Prefetch(
                    query=sparse_query,
                    using="sparse",
                    limit=candidate_limit,
                )
            )
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=qdrant_models.FusionQuery(fusion=qdrant_models.Fusion.RRF),
            query_filter=payload_filter,
            limit=candidate_limit,
            with_payload=True,
        )
        hits = self._rerank_documents(
            hits=list(response.points),
            query=query,
            filters=filters or {},
        )[:top_k]
        snippets: list[KnowledgeSnippet] = []
        for hit in hits:
            payload = hit.payload or {}
            snippets.append(
                KnowledgeSnippet(
                    id=str(payload.get("id", hit.id)),
                    content=str(payload.get("chunk_text", payload.get("content", ""))),
                    knowledge_type="document",
                    source_type=str(payload.get("source_type", "document_chunk")),
                    confidence=float(payload.get("confidence", 0.0)),
                    score=float(getattr(hit, "score", 0.0) or 0.0),
                    source_ref={
                        "source_file": payload.get("source_file", ""),
                        "page_number": payload.get("page_number"),
                        "section": payload.get("section", ""),
                        "title": payload.get("title", ""),
                        "citation_label": payload.get("citation_label", ""),
                    },
                    metadata=dict(payload),
                    retrieval_reason="document_hybrid",
                )
            )
        return snippets
