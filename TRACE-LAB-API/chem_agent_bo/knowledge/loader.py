"""Load layered local knowledge resources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chem_agent_bo.knowledge.schema import KnowledgeUnit
from chem_agent_bo.knowledge.store import LanceDBStore


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _load_yaml(path: Path) -> list[dict[str, Any]]:
    try:
        import yaml
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Reading yaml knowledge files requires pyyaml. "
            "Install with `pip install --target .vendor_py pyyaml`."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"Unexpected yaml data in {path}. Expect list/dict.")
    return [item for item in data if isinstance(item, dict)]


def load_units_from_local_dir(local_dir: str) -> list[KnowledgeUnit]:
    path = Path(local_dir)
    if not path.exists():
        return []
    raw_items: list[dict[str, Any]] = []
    for file_path in sorted(path.glob("*.jsonl")):
        raw_items.extend(_load_jsonl(file_path))
    for file_path in sorted(path.glob("*.yaml")):
        raw_items.extend(_load_yaml(file_path))
    for file_path in sorted(path.glob("*.yml")):
        raw_items.extend(_load_yaml(file_path))
    units: list[KnowledgeUnit] = []
    seen_ids: set[str] = set()
    for item in raw_items:
        unit = KnowledgeUnit.from_dict(item)
        if unit.id in seen_ids:
            continue
        seen_ids.add(unit.id)
        units.append(unit)
    return units


def load_local_knowledge_into_store(local_dir: str, store: LanceDBStore) -> int:
    units = load_units_from_local_dir(local_dir)
    if not units:
        return 0
    return store.upsert_units(units)


def load_rule_units_from_dir(rules_dir: str) -> list[KnowledgeUnit]:
    """Load curated rule/failure/intervention knowledge from JSONL."""
    path = Path(rules_dir)
    if not path.exists():
        return []
    units: list[KnowledgeUnit] = []
    seen: set[str] = set()
    for file_path in sorted(path.glob("*.jsonl")):
        rows = _load_jsonl(file_path)
        for row in rows:
            if "content" not in row and "text" in row:
                row["content"] = row["text"]
            unit = KnowledgeUnit.from_dict(row)
            if unit.id in seen:
                continue
            seen.add(unit.id)
            units.append(unit)
    return units
