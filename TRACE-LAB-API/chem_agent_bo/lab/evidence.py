"""Scoped evidence cards for real-lab TRACE projects."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ALLOWED_MAPPING_STATUSES = {
    "direct",
    "same_start_end",
    "same_reaction_family",
    "variable_level",
    "background",
    "out_of_scope",
}
BLOCKED_ALLOWED_USES = {"blocked", "do_not_use", "oracle"}
BLOCKED_LEAKAGE_RISKS = {
    "oracle",
    "benchmark_oracle",
    "candidate_lookup",
    "candidate_level_lookup",
    "do_not_use",
    "blocked",
}
MAPPING_PRIORITY = {
    "direct": 50,
    "same_start_end": 42,
    "same_reaction_family": 34,
    "variable_level": 25,
    "background": 10,
    "out_of_scope": -100,
}


@dataclass
class EvidenceCard:
    card_id: str
    source: str
    summary: str
    reaction_scope: str = ""
    variable_scope: list[str] = field(default_factory=list)
    target_nodes: list[str] = field(default_factory=list)
    mapping_status: str = "background"
    confidence: str = "medium"
    allowed_use: str = "advisory"
    source_type: str = "literature"
    source_path: str = ""
    doi: str = ""
    supporting_excerpt: str = ""
    transferability_note: str = ""
    leakage_risk: str = "clean_literature_prior"
    notes: str = ""

    def normalized(self) -> "EvidenceCard":
        status = str(self.mapping_status or "background").strip().lower()
        if status not in ALLOWED_MAPPING_STATUSES:
            status = "background"
        return EvidenceCard(
            card_id=str(self.card_id),
            source=str(self.source),
            summary=str(self.summary),
            reaction_scope=str(self.reaction_scope),
            variable_scope=_as_list(self.variable_scope),
            target_nodes=_as_list(self.target_nodes),
            mapping_status=status,
            confidence=str(self.confidence or "medium"),
            allowed_use=str(self.allowed_use or "advisory"),
            source_type=str(self.source_type or "literature"),
            source_path=str(self.source_path or ""),
            doi=str(self.doi or ""),
            supporting_excerpt=str(self.supporting_excerpt or ""),
            transferability_note=str(self.transferability_note or ""),
            leakage_risk=str(self.leakage_risk or "clean_literature_prior"),
            notes=str(self.notes or ""),
        )


class EvidenceStore:
    """Small file-backed store for scoped, non-oracle evidence cards."""

    def __init__(self, cards: list[EvidenceCard] | None = None) -> None:
        self.cards = [card.normalized() for card in cards or []]

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceStore":
        path_obj = Path(path)
        if not path_obj.exists():
            return cls([])
        if path_obj.suffix.lower() == ".jsonl":
            cards = []
            with path_obj.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    cards.append(_card_from_dict(json.loads(line)))
            return cls(cards)
        if path_obj.suffix.lower() == ".csv":
            with path_obj.open("r", encoding="utf-8-sig", newline="") as handle:
                return cls([_card_from_dict(row) for row in csv.DictReader(handle)])
        raise ValueError(f"Unsupported evidence card file: {path_obj}")

    def write_jsonl(self, path: str | Path) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w", encoding="utf-8") as handle:
            for card in self.cards:
                handle.write(json.dumps(asdict(card), ensure_ascii=False) + "\n")

    def applicable(
        self,
        *,
        variables: list[str],
        reaction_scope: str = "",
        target_nodes: list[str] | None = None,
        max_items: int = 5,
    ) -> list[EvidenceCard]:
        variable_set = {_norm(item) for item in variables}
        target_set = {_norm(item) for item in target_nodes or []}
        scored: list[tuple[float, int, EvidenceCard]] = []
        for card in self.cards:
            if card.mapping_status == "out_of_scope":
                continue
            if card.allowed_use.strip().lower() in BLOCKED_ALLOWED_USES:
                continue
            if _norm(card.leakage_risk) in BLOCKED_LEAKAGE_RISKS:
                continue
            card_variables = {_norm(item) for item in card.variable_scope}
            variable_overlap = variable_set.intersection(card_variables)
            if card_variables:
                if not variable_overlap and card.mapping_status != "background":
                    continue
            card_targets = {_norm(item) for item in card.target_nodes}
            target_overlap = target_set.intersection(card_targets)
            if target_set and card_targets and not target_overlap:
                continue
            if reaction_scope and card.reaction_scope:
                reaction_match = _reaction_match(reaction_scope, card.reaction_scope)
                if not reaction_match:
                    if card.mapping_status in {"direct", "same_start_end"}:
                        continue
            else:
                reaction_match = False
            score = _card_score(
                card,
                variable_overlap=len(variable_overlap),
                target_overlap=len(target_overlap),
                reaction_match=reaction_match,
            )
            scored.append((score, len(scored), card))
        limit = max(0, int(max_items))
        return [
            card
            for _score, _idx, card in sorted(scored, key=lambda item: (-item[0], item[1]))[
                :limit
            ]
        ]


def _card_from_dict(payload: dict[str, Any]) -> EvidenceCard:
    return EvidenceCard(
        card_id=str(payload.get("card_id") or payload.get("id") or ""),
        source=str(payload.get("source") or ""),
        summary=str(payload.get("summary") or payload.get("content") or ""),
        reaction_scope=str(payload.get("reaction_scope") or ""),
        variable_scope=_as_list(payload.get("variable_scope", [])),
        target_nodes=_as_list(payload.get("target_nodes", [])),
        mapping_status=str(payload.get("mapping_status") or "background"),
        confidence=str(payload.get("confidence") or "medium"),
        allowed_use=str(payload.get("allowed_use") or "advisory"),
        source_type=str(payload.get("source_type") or "literature"),
        source_path=str(payload.get("source_path") or ""),
        doi=str(payload.get("doi") or ""),
        supporting_excerpt=str(payload.get("supporting_excerpt") or ""),
        transferability_note=str(payload.get("transferability_note") or ""),
        leakage_risk=str(payload.get("leakage_risk") or "clean_literature_prior"),
        notes=str(payload.get("notes") or ""),
    )


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    return [str(item).strip() for item in list(value or []) if str(item).strip()]


def _norm(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _reaction_match(requested: str, card_scope: str) -> bool:
    request = str(requested or "").strip().lower()
    scope = str(card_scope or "").strip().lower()
    return bool(request and scope and (request in scope or scope in request))


def _confidence_score(value: object) -> float:
    text = str(value or "").strip().lower()
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        pass
    if text in {"high", "strong"}:
        return 0.9
    if text in {"medium", "moderate"}:
        return 0.6
    if text in {"low", "weak"}:
        return 0.3
    return 0.5


def _card_score(
    card: EvidenceCard,
    *,
    variable_overlap: int,
    target_overlap: int,
    reaction_match: bool,
) -> float:
    score = float(MAPPING_PRIORITY.get(card.mapping_status, 0))
    score += 10.0 * _confidence_score(card.confidence)
    score += 4.0 * variable_overlap
    score += 3.0 * target_overlap
    if reaction_match:
        score += 5.0
    if card.mapping_status == "background" and not variable_overlap:
        score -= 2.0
    return score
