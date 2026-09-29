"""Scoped evidence cards for real-lab TRACE projects."""

from __future__ import annotations

import csv
import io
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
# The controller nodes that ask for evidence. Every ask retrieves once, for all of them together.
LAB_EVIDENCE_TARGET_NODES = (
    "design_init_experiments",
    "stagnation_diagnosis",
    "hypothesis_action",
    "semantic_assessment",
    "verification_pass",
    "reflection_action",
    "lab_batch_composition",
)
SCREEN_REASONS = {
    "out_of_scope": "mapping_status is out_of_scope.",
    "allowed_use_blocked": "allowed_use blocks the controller from seeing it.",
    "leakage_risk_blocked": "leakage_risk marks it as oracle-like.",
    "no_variable_overlap": "variable_scope shares no variable with the design space.",
    "target_node_mismatch": "target_nodes lists no controller node that asks for evidence.",
    "reaction_scope_mismatch": (
        "mapping_status is direct/same_start_end, but the card's reaction_scope is not contained in "
        "(or containing) the project's reaction_scope."
    ),
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
        scored: list[tuple[float, int, EvidenceCard]] = []
        for card in self.cards:
            score, _reason = self._screen(
                card, variables=variables, reaction_scope=reaction_scope, target_nodes=target_nodes
            )
            if score is not None:
                scored.append((score, len(scored), card))
        limit = max(0, int(max_items))
        return [
            card
            for _score, _idx, card in sorted(scored, key=lambda item: (-item[0], item[1]))[
                :limit
            ]
        ]

    def explain(
        self,
        *,
        variables: list[str],
        reaction_scope: str = "",
        target_nodes: list[str] | None = None,
        max_items: int = 5,
    ) -> list[dict[str, Any]]:
        """One entry per card: whether `applicable` would return it, at which rank, or why never.

        Uses the same screening as `applicable`, so the two cannot disagree. `reason` is
        empty for a card that passes the screen (it may still miss the `max_items` cut).
        """
        screened = [
            (card, *self._screen(card, variables=variables, reaction_scope=reaction_scope, target_nodes=target_nodes))
            for card in self.cards
        ]
        passing = [(index, score) for index, (_card, score, _reason) in enumerate(screened) if score is not None]
        # Same ordering as `applicable`: score descending, then file order.
        ranked = [index for index, _score in sorted(passing, key=lambda item: (-item[1], item[0]))]
        rank_of = {index: rank for rank, index in enumerate(ranked, start=1)}
        limit = max(0, int(max_items))
        entries = []
        for index, (card, score, reason) in enumerate(screened):
            rank = rank_of.get(index)
            entries.append(
                {
                    "card_id": card.card_id,
                    "mapping_status": card.mapping_status,
                    "confidence": card.confidence,
                    "score": None if score is None else round(score, 2),
                    "rank": rank,
                    "retrieved": rank is not None and rank <= limit,
                    "reason": reason,
                    "reason_text": SCREEN_REASONS.get(reason, ""),
                }
            )
        return entries

    @staticmethod
    def _screen(
        card: EvidenceCard,
        *,
        variables: list[str],
        reaction_scope: str = "",
        target_nodes: list[str] | None = None,
    ) -> tuple[float | None, str]:
        """Score a card for retrieval, or return (None, reason) when it can never be retrieved."""
        variable_set = {_norm(item) for item in variables}
        target_set = {_norm(item) for item in target_nodes or []}
        if card.mapping_status == "out_of_scope":
            return None, "out_of_scope"
        if card.allowed_use.strip().lower() in BLOCKED_ALLOWED_USES:
            return None, "allowed_use_blocked"
        if _norm(card.leakage_risk) in BLOCKED_LEAKAGE_RISKS:
            return None, "leakage_risk_blocked"
        card_variables = {_norm(item) for item in card.variable_scope}
        variable_overlap = variable_set.intersection(card_variables)
        if card_variables:
            if not variable_overlap and card.mapping_status != "background":
                return None, "no_variable_overlap"
        card_targets = {_norm(item) for item in card.target_nodes}
        target_overlap = target_set.intersection(card_targets)
        if target_set and card_targets and not target_overlap:
            return None, "target_node_mismatch"
        if reaction_scope and card.reaction_scope:
            reaction_match = _reaction_match(reaction_scope, card.reaction_scope)
            if not reaction_match:
                if card.mapping_status in {"direct", "same_start_end"}:
                    return None, "reaction_scope_mismatch"
        else:
            reaction_match = False
        score = _card_score(
            card,
            variable_overlap=len(variable_overlap),
            target_overlap=len(target_overlap),
            reaction_match=reaction_match,
        )
        return score, ""


def retrieval_preview(
    store: EvidenceStore,
    *,
    variables: list[str],
    reaction_scope: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """What the controller would be shown for a project, using the retrieval an ask really makes.

    An ask retrieves once, for all `LAB_EVIDENCE_TARGET_NODES` together and the design
    variables, and keeps the `top_k` best cards; those same cards serve every controller
    node in that ask. Cards split into `retrieved`, `not_shown` (eligible, but below the
    cut) and `never_retrieved` (screened out, with the reason).
    """
    entries = store.explain(
        variables=variables,
        reaction_scope=reaction_scope,
        target_nodes=list(LAB_EVIDENCE_TARGET_NODES),
        max_items=top_k,
    )
    retrieved = sorted((item for item in entries if item["retrieved"]), key=lambda item: item["rank"])
    not_shown = sorted(
        (item for item in entries if item["rank"] is not None and not item["retrieved"]),
        key=lambda item: item["rank"],
    )
    never = [item for item in entries if item["rank"] is None]
    return {
        "top_k": int(top_k),
        "card_count": len(entries),
        "retrieved": retrieved,
        "not_shown": not_shown,
        "never_retrieved": never,
    }


class EvidenceImportError(ValueError):
    """Evidence cards failed validation on import; `problems` lists every issue found."""

    def __init__(self, message: str, problems: list[str]) -> None:
        super().__init__(message)
        self.problems = problems


def parse_evidence_upload(
    *,
    cards: list[dict[str, Any]] | None = None,
    jsonl: str | None = None,
    csv_text: str | None = None,
) -> list[EvidenceCard]:
    """Strictly validate uploaded evidence cards (exactly one source).

    Unlike loading a project's own file, nothing is silently repaired here: a missing
    card_id or summary, a repeated card_id, or a mapping_status outside
    ALLOWED_MAPPING_STATUSES (which loading would quietly turn into `background`) is an
    error, and all problems are reported together.
    """
    given = [source for source in (cards, jsonl, csv_text) if source]
    if len(given) != 1:
        raise EvidenceImportError("Give the evidence cards either as cards, as JSONL or as CSV (exactly one).", [])
    raw: list[tuple[str, dict[str, Any]]] = []
    problems: list[str] = []
    if cards:
        raw = [(f"card {index}", dict(item)) for index, item in enumerate(cards, start=1)]
    elif jsonl:
        for number, line in enumerate(str(jsonl).splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw.append((f"line {number}", dict(json.loads(line))))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                problems.append(f"line {number}: not a JSON object ({exc}).")
    else:
        reader = csv.DictReader(io.StringIO(str(csv_text).lstrip("﻿")))
        raw = [(f"line {index + 2}", dict(row)) for index, row in enumerate(reader)]
    seen: dict[str, str] = {}
    parsed: list[EvidenceCard] = []
    for where, payload in raw:
        card_id = str(payload.get("card_id") or payload.get("id") or "").strip()
        if not card_id:
            problems.append(f"{where}: missing card_id.")
            continue
        if card_id in seen:
            problems.append(f"{where}: card_id `{card_id}` repeats {seen[card_id]}.")
        seen.setdefault(card_id, where)
        if not str(payload.get("summary") or payload.get("content") or "").strip():
            problems.append(f"{where} ({card_id}): empty summary.")
        status = str(payload.get("mapping_status") or "background").strip().lower()
        if status not in ALLOWED_MAPPING_STATUSES:
            problems.append(
                f"{where} ({card_id}): mapping_status `{status}` must be one of {sorted(ALLOWED_MAPPING_STATUSES)}."
            )
        parsed.append(_card_from_dict(payload))
    if problems:
        raise EvidenceImportError(f"{len(problems)} problem(s) in the evidence cards; nothing was imported.", problems)
    return [card.normalized() for card in parsed]


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
