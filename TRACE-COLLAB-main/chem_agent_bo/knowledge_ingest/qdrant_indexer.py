"""Qdrant indexing utilities for document chunks."""

from __future__ import annotations

import uuid
from typing import Any

from chem_agent_bo.knowledge_ingest.document_schema import ChunkRecord

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
except Exception:  # noqa: BLE001
    QdrantClient = None
    qdrant_models = None


class QdrantIndexer:
    """Minimal chunk upsert helper."""

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        collection_name: str = "chem_documents",
        api_key: str | None = None,
    ) -> None:
        if QdrantClient is None:
            raise RuntimeError(
                "qdrant-client is not installed. Install with "
                "`pip install --target .vendor_py qdrant-client`."
            )
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self, vector_size: int = 384) -> None:
        if qdrant_models is None:
            return
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            info = self.client.get_collection(self.collection_name)
            params = info.config.params
            dense_ok = isinstance(params.vectors, dict) and "dense" in params.vectors
            sparse_ok = isinstance(params.sparse_vectors, dict) and "sparse" in params.sparse_vectors
            if not (dense_ok and sparse_ok):
                raise RuntimeError(
                    f"Collection '{self.collection_name}' already exists but is not hybrid-ready "
                    "(missing named dense/sparse vectors). Create a new collection name for hybrid mode."
                )
        else:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": qdrant_models.VectorParams(
                        size=vector_size,
                        distance=qdrant_models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": qdrant_models.SparseVectorParams(
                        modifier=qdrant_models.Modifier.IDF
                    )
                },
            )
        # Build payload indexes early for stable filtered retrieval performance.
        for field_name in ("dataset_scope", "variable_tags", "trigger_tags"):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    def upsert_chunks(
        self,
        chunks: list[ChunkRecord],
        dense_vectors: list[list[float]],
        sparse_vectors: list[tuple[list[int], list[float]]],
    ) -> int:
        if qdrant_models is None:
            return 0
        if not chunks:
            return 0
        if not (len(chunks) == len(dense_vectors) == len(sparse_vectors)):
            raise ValueError("chunks/dense_vectors/sparse_vectors length mismatch.")
        points: list[Any] = []
        for chunk, dense_vector, sparse_vector in zip(chunks, dense_vectors, sparse_vectors):
            # Some Qdrant deployments only accept numeric or UUID point IDs.
            # Use deterministic UUID5 to keep id stable across re-ingests.
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id))
            sparse_indices, sparse_values = sparse_vector
            sparse_payload = qdrant_models.SparseVector(indices=sparse_indices, values=sparse_values)
            payload = {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "source_file": chunk.source_file,
                "title": chunk.title,
                "section": chunk.section,
                "page_number": chunk.page_number,
                "chunk_text": chunk.chunk_text,
                "dataset_scope": chunk.dataset_scope,
                "variable_tags": chunk.variable_tags,
                "trigger_tags": chunk.trigger_tags,
                "source_type": chunk.source_type,
                "citation_label": chunk.citation_label,
                **chunk.metadata,
            }
            points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vector,
                        "sparse": sparse_payload,
                    },
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)
