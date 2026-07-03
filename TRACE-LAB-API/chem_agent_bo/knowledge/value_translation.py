"""Persistent cache helpers for value translation / annotation context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # noqa: BLE001
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = PROJECT_ROOT / "chem_agent_bo" / "knowledge" / "value_translation_cache.yaml"


def load_translation_cache(cache_path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = cache_path or CACHE_PATH
    if not path.exists() or yaml is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    entries = data.get("entries", {}) if isinstance(data, dict) else {}
    return entries if isinstance(entries, dict) else {}


def save_translation_cache(
    entries: dict[str, dict[str, Any]],
    cache_path: Path | None = None,
) -> None:
    path = cache_path or CACHE_PATH
    if yaml is None:
        return
    payload = {
        "version": 1,
        "entries": dict(sorted(entries.items(), key=lambda item: item[0])),
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            payload,
            handle,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )


def merge_translation_entry(
    existing: dict[str, Any] | None,
    *,
    original_value: str,
    column: str,
    translated_description: str,
    brief_properties: list[str] | None,
    likely_role: str,
    confidence: str,
    source: str = "llm_translation",
) -> dict[str, Any]:
    prev = dict(existing or {})
    columns_seen = {str(item) for item in prev.get("columns_seen", []) if str(item).strip()}
    if column.strip():
        columns_seen.add(column.strip())
    properties = [str(item).strip() for item in (brief_properties or []) if str(item).strip()]
    return {
        "original_value": original_value,
        "translated_description": translated_description.strip(),
        "brief_properties": properties,
        "likely_role": likely_role.strip(),
        "confidence": confidence.strip() or "medium",
        "columns_seen": sorted(columns_seen),
        "source": source,
    }


def render_annotation_context(
    entries: dict[str, dict[str, Any]],
    requested_values: dict[str, list[str]],
    *,
    max_entries: int = 40,
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for column, values in requested_values.items():
        for value in values:
            key = (str(column), str(value))
            if key in seen:
                continue
            seen.add(key)
            cached = entries.get(str(value))
            if not cached:
                continue
            rendered.append(
                {
                    "column": str(column),
                    "original_value": str(value),
                    "translated_description": str(cached.get("translated_description", "")),
                    "brief_properties": list(cached.get("brief_properties", [])),
                    "likely_role": str(cached.get("likely_role", "")),
                    "confidence": str(cached.get("confidence", "medium")),
                    "columns_seen": list(cached.get("columns_seen", [])),
                    "source": str(cached.get("source", "llm_translation")),
                }
            )
            if len(rendered) >= max_entries:
                return rendered
    return rendered
