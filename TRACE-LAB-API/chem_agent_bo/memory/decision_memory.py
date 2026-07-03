"""Trace sink for optimization logs (not long-term memory)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DecisionMemory:
    """In-memory trace list with JSON persistence."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def record_decision(self, record: dict[str, Any]) -> None:
        self._records.append(record)

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._records)

    def save_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self._records, handle, indent=2, ensure_ascii=False, default=str)

