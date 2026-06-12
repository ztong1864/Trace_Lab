"""Long-term memory adapter based on LangGraph stores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from langgraph.store.memory import InMemoryStore
except Exception:  # noqa: BLE001
    InMemoryStore = None


class LongTermMemoryStore:
    """Unified API for long-term memory with local JSON fallback."""

    def __init__(self, namespace_prefix: tuple[str, ...], fallback_json_path: str) -> None:
        self.namespace_prefix = namespace_prefix
        self.fallback_json_path = Path(fallback_json_path)
        self._store = InMemoryStore() if InMemoryStore is not None else None
        self._fallback_data: dict[str, dict[str, Any]] = {}
        self._load_fallback()

    def _load_fallback(self) -> None:
        if not self.fallback_json_path.exists():
            self._fallback_data = {}
            return
        try:
            with self.fallback_json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                self._fallback_data = {
                    str(key): value
                    for key, value in data.items()
                    if isinstance(value, dict)
                }
        except Exception:  # noqa: BLE001
            self._fallback_data = {}

    def _persist_fallback(self) -> None:
        self.fallback_json_path.parent.mkdir(parents=True, exist_ok=True)
        with self.fallback_json_path.open("w", encoding="utf-8") as handle:
            json.dump(self._fallback_data, handle, indent=2, ensure_ascii=False, default=str)

    def put(self, namespace_suffix: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        namespace = self.namespace_prefix + namespace_suffix
        if self._store is not None:
            self._store.put(namespace, key, value)
            return
        joined = "/".join(namespace + (key,))
        self._fallback_data[joined] = value
        self._persist_fallback()

    def search(
        self,
        namespace_suffix: tuple[str, ...],
        *,
        query: str,
        filter_dict: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        namespace = self.namespace_prefix + namespace_suffix
        if self._store is not None:
            items = self._store.search(
                namespace,
                query=query,
                filter=filter_dict or {},
                limit=max(1, int(limit)),
            )
            return [item.value for item in items if hasattr(item, "value")]
        results: list[dict[str, Any]] = []
        ns_prefix = "/".join(namespace) + "/"
        for key, value in self._fallback_data.items():
            if not key.startswith(ns_prefix):
                continue
            if filter_dict:
                matched = True
                for f_key, f_val in filter_dict.items():
                    if value.get(f_key) != f_val:
                        matched = False
                        break
                if not matched:
                    continue
            text = json.dumps(value, ensure_ascii=False)
            if query.strip() and query.lower() not in text.lower():
                continue
            results.append(value)
        return results[: max(1, int(limit))]
