"""Hybrid dense+sparse embedding helpers for Qdrant retrieval."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from langchain_core.embeddings import Embeddings

from chem_agent_bo.knowledge.store import HashEmbeddings

try:
    from fastembed import SparseTextEmbedding, TextEmbedding
except Exception:  # noqa: BLE001
    SparseTextEmbedding = None
    TextEmbedding = None


class StableHashEmbeddings(HashEmbeddings):
    """Deterministic fallback embeddings across processes."""

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = self._tokenize(text)
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dimensions
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 0.0:
            return vec
        return [v / norm for v in vec]


class StableSparseEncoder:
    """Lightweight deterministic sparse encoder fallback."""

    def __init__(self, max_terms: int = 256) -> None:
        self.max_terms = max(16, int(max_terms))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    @staticmethod
    def _token_index(token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        counts: dict[int, int] = {}
        for token in self._tokenize(text):
            idx = self._token_index(token)
            counts[idx] = counts.get(idx, 0) + 1
        if not counts:
            return [], []
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[: self.max_terms]
        indices = [idx for idx, _ in ranked]
        values = [math.log1p(float(cnt)) for _, cnt in ranked]
        return indices, values


class QdrantHybridEmbedder:
    """Unified embedding adapter used by both ingestion and query."""

    def __init__(
        self,
        *,
        dense_model: str = "BAAI/bge-small-en-v1.5",
        sparse_model: str = "Qdrant/bm25",
    ) -> None:
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self._dense: Embeddings | None = None
        self._sparse = None
        self._sparse_fallback = StableSparseEncoder()
        self._dense_dim = 384
        self._init_dense()
        self._init_sparse()

    @property
    def dense_dimensions(self) -> int:
        return int(self._dense_dim)

    def _init_dense(self) -> None:
        if TextEmbedding is None:
            self._dense = StableHashEmbeddings(dimensions=384)
            self._dense_dim = 384
            return
        self._dense = TextEmbedding(model_name=self.dense_model, lazy_load=True)
        for item in TextEmbedding.list_supported_models():
            if not isinstance(item, dict):
                continue
            if item.get("model") == self.dense_model and item.get("dim"):
                self._dense_dim = int(item["dim"])
                return
        probe = next(self._dense.embed(["dimension_probe"]))
        self._dense_dim = len(probe.tolist())

    def _init_sparse(self) -> None:
        if SparseTextEmbedding is None:
            self._sparse = None
            return
        self._sparse = SparseTextEmbedding(model_name=self.sparse_model, lazy_load=True)

    def embed_dense_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if TextEmbedding is not None and isinstance(self._dense, TextEmbedding):
            vectors = list(self._dense.embed(texts))
            return [vec.tolist() for vec in vectors]
        assert self._dense is not None
        return self._dense.embed_documents(texts)

    def embed_dense_query(self, text: str) -> list[float]:
        if TextEmbedding is not None and isinstance(self._dense, TextEmbedding):
            return next(self._dense.embed([text])).tolist()
        assert self._dense is not None
        return self._dense.embed_query(text)

    def embed_sparse_documents(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        if not texts:
            return []
        if self._sparse is None:
            return [self._sparse_fallback.encode(text) for text in texts]
        vectors = list(self._sparse.embed(texts))
        return [
            (vector.indices.tolist(), [float(v) for v in vector.values.tolist()]) for vector in vectors
        ]

    def embed_sparse_query(self, text: str) -> tuple[list[int], list[float]]:
        if self._sparse is None:
            return self._sparse_fallback.encode(text)
        vector = next(self._sparse.embed([text]))
        return vector.indices.tolist(), [float(v) for v in vector.values.tolist()]

    def describe(self) -> dict[str, Any]:
        return {
            "dense_model": self.dense_model,
            "dense_dimensions": self.dense_dimensions,
            "dense_backend": (
                "fastembed"
                if TextEmbedding is not None and isinstance(self._dense, TextEmbedding)
                else "stable_hash"
            ),
            "sparse_model": self.sparse_model,
            "sparse_backend": "fastembed" if self._sparse is not None else "stable_sparse",
        }
