"""LanceDB-backed storage for knowledge units."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from chem_agent_bo.knowledge.schema import KnowledgeUnit

try:
    import lancedb
except Exception:  # noqa: BLE001
    lancedb = None


class HashEmbeddings(Embeddings):
    """Deterministic local fallback embeddings when API key is unavailable."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = max(32, int(dimensions))

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = self._tokenize(text)
        if not tokens:
            return vec
        for token in tokens:
            idx = hash(token) % self.dimensions
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 0.0:
            return vec
        return [v / norm for v in vec]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]


def _escape_like_token(text: str) -> str:
    return text.replace("'", "''")


def _pack_tag_field(tags: list[str]) -> str:
    if not tags:
        return "|"
    cleaned = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
    return "|" + "|".join(cleaned) + "|"


def _pack_payload(unit: KnowledgeUnit, vector: list[float]) -> dict[str, Any]:
    return {
        "id": unit.id,
        "content": unit.content,
        "dataset_scope": _pack_tag_field(unit.dataset_scope),
        "variable_tags": _pack_tag_field(unit.variable_tags),
        "trigger_tags": _pack_tag_field(unit.trigger_tags),
        "source_type": unit.source_type,
        "confidence": float(unit.confidence),
        "metadata_json": json.dumps(unit.metadata, ensure_ascii=False, default=str),
        "vector": vector,
    }


class LanceDBStore:
    """Small wrapper around LanceDB local table operations."""

    def __init__(
        self,
        db_dir: str,
        table_name: str = "knowledge_units",
        embeddings: Embeddings | None = None,
    ) -> None:
        if lancedb is None:
            raise RuntimeError(
                "lancedb is not installed. Install with "
                "`pip install --target .vendor_py lancedb`."
            )
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.table_name = table_name
        self.embeddings = embeddings or self._default_embeddings()
        self.db = lancedb.connect(str(self.db_dir))
        self.table = None

    @staticmethod
    def _default_embeddings() -> Embeddings:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            from langchain_openai import OpenAIEmbeddings

            kwargs: dict[str, Any] = {"model": "text-embedding-3-small", "api_key": api_key}
            api_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
            if api_base:
                kwargs["base_url"] = api_base
            return OpenAIEmbeddings(**kwargs)
        return HashEmbeddings()

    def _open_or_create_table(self) -> None:
        if self.table is not None:
            return
        table_names = set(self.db.table_names())
        if self.table_name in table_names:
            self.table = self.db.open_table(self.table_name)
            return
        placeholder = _pack_payload(
            KnowledgeUnit(id="__bootstrap__", content="bootstrap row", metadata={"bootstrap": True}),
            vector=self.embeddings.embed_query("bootstrap row"),
        )
        self.table = self.db.create_table(self.table_name, data=[placeholder], mode="create")
        self.table.delete("id = '__bootstrap__'")

    def upsert_units(self, units: list[KnowledgeUnit]) -> int:
        if not units:
            return 0
        self._open_or_create_table()
        assert self.table is not None
        texts = [unit.content for unit in units]
        vectors = self.embeddings.embed_documents(texts)
        payloads = [_pack_payload(unit, vector) for unit, vector in zip(units, vectors)]
        ids = [_escape_like_token(unit.id) for unit in units]
        if ids:
            quoted = ",".join(f"'{item}'" for item in ids)
            self.table.delete(f"id IN ({quoted})")
        self.table.add(payloads)
        return len(payloads)

    def _build_filter(self, filters: dict[str, Any] | None) -> str | None:
        if not filters:
            return None
        clauses: list[str] = []
        dataset = str(filters.get("dataset", "")).strip().lower()
        if dataset:
            clauses.append(
                f"(dataset_scope LIKE '%|*|%' OR dataset_scope LIKE '%|{_escape_like_token(dataset)}|%')"
            )
        variable_tags = [str(x).strip().lower() for x in filters.get("variable_tags", []) if str(x).strip()]
        if variable_tags:
            variable_clauses = [
                f"variable_tags LIKE '%|{_escape_like_token(tag)}|%'" for tag in variable_tags
            ]
            clauses.append("(" + " OR ".join(variable_clauses) + ")")
        trigger_tags = [str(x).strip().lower() for x in filters.get("trigger_tags", []) if str(x).strip()]
        if trigger_tags:
            trigger_clauses = [
                f"trigger_tags LIKE '%|{_escape_like_token(tag)}|%'" for tag in trigger_tags
            ]
            clauses.append("(" + " OR ".join(trigger_clauses) + ")")
        if not clauses:
            return None
        return " AND ".join(clauses)

    def similarity_search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        self._open_or_create_table()
        assert self.table is not None
        top_k = max(1, int(top_k))
        vector = self.embeddings.embed_query(query)
        filter_expr = self._build_filter(filters)
        search = self.table.search(vector)
        if filter_expr:
            search = search.where(filter_expr, prefilter=True)
        rows = search.limit(top_k).to_list()
        results: list[dict[str, Any]] = []
        for row in rows:
            metadata_json = row.get("metadata_json") or "{}"
            try:
                metadata = json.loads(metadata_json)
            except Exception:  # noqa: BLE001
                metadata = {"raw_metadata": metadata_json}
            item = {
                "id": row.get("id"),
                "content": row.get("content", ""),
                "source_type": row.get("source_type", "unknown"),
                "confidence": float(row.get("confidence", 0.0)),
                "score": float(row.get("_distance", 0.0)),
                "metadata": metadata,
            }
            results.append(item)
        return results
