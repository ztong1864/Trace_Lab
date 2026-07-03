"""Lightweight procedural skill registry for prompt-time injection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProceduralSkillCard:
    skill_id: str
    name: str
    description: str
    version: str
    target_nodes: tuple[str, ...]
    content: str
    path: str


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, text.strip()
    try:
        closing_idx = next(
            idx for idx, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration:
        return {}, text.strip()
    raw_meta = "\n".join(lines[1:closing_idx]).strip()
    body = "\n".join(lines[closing_idx + 1 :]).strip()
    parsed = yaml.safe_load(raw_meta) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    return parsed, body


def _coerce_target_nodes(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        cleaned = value.strip()
        return (cleaned,) if cleaned else ()
    if isinstance(value, list):
        nodes = [str(item).strip() for item in value if str(item).strip()]
        return tuple(nodes)
    return ()


def _load_card(path: Path) -> ProceduralSkillCard:
    text = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)
    skill_id = str(meta.get("id") or meta.get("name") or path.stem).strip()
    name = str(meta.get("name") or skill_id).strip()
    description = str(meta.get("description") or "").strip()
    version = str(meta.get("version") or "1.0").strip()
    target_nodes = _coerce_target_nodes(meta.get("target_nodes"))
    return ProceduralSkillCard(
        skill_id=skill_id or path.stem,
        name=name or path.stem,
        description=description,
        version=version or "1.0",
        target_nodes=target_nodes,
        content=body.strip(),
        path=str(path),
    )


@lru_cache(maxsize=8)
def _load_cards_from_dir(cards_dir: str) -> tuple[ProceduralSkillCard, ...]:
    root = Path(cards_dir)
    if not root.exists():
        return ()
    return tuple(_load_card(path) for path in sorted(root.glob("*.md")))


class ProceduralSkillRegistry:
    """Selects and renders prompt-time procedural skill cards."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        cards_dir: str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.cards_dir = (
            Path(cards_dir).resolve()
            if cards_dir
            else (Path(__file__).resolve().parent / "cards")
        )

    def render_for_node(self, node_name: str) -> tuple[str, dict[str, Any]]:
        node = str(node_name).strip()
        meta: dict[str, Any] = {
            "node_name": node,
            "skills_enabled": self.enabled,
            "cards_dir": str(self.cards_dir),
            "loaded_skill_ids": [],
            "loaded_skills": [],
            "skill_count": 0,
            "skill_char_count": 0,
        }
        if not self.enabled or not node:
            return "", meta
        matched = [
            card
            for card in _load_cards_from_dir(str(self.cards_dir))
            if node in card.target_nodes
        ]
        if not matched:
            return "", meta
        rendered = "\n\n".join(card.content for card in matched if card.content.strip()).strip()
        meta["loaded_skill_ids"] = [card.skill_id for card in matched]
        meta["loaded_skills"] = [
            {
                "id": card.skill_id,
                "name": card.name,
                "version": card.version,
                "description": card.description,
                "path": card.path,
            }
            for card in matched
        ]
        meta["skill_count"] = len(matched)
        meta["skill_char_count"] = len(rendered)
        return rendered, meta
