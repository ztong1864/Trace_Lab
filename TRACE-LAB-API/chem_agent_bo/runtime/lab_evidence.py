"""Scoped evidence adapter for real-lab controller calls."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from chem_agent_bo.lab.evidence import EvidenceCard, EvidenceStore


class LabEvidenceProvider:
    """Expose lab evidence cards as non-oracle controller knowledge units."""

    def __init__(self, store: EvidenceStore) -> None:
        self.store = store

    def retrieve(
        self,
        *,
        variables: list[str],
        reaction_scope: str,
        target_nodes: list[str] | None = None,
        max_items: int = 5,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cards = self.store.applicable(
            variables=variables,
            reaction_scope=reaction_scope,
            target_nodes=target_nodes or [],
            max_items=max_items,
        )
        units = [self._card_to_unit(card, rank=rank) for rank, card in enumerate(cards, start=1)]
        meta = {
            "provider": "lab_evidence_cards",
            "reaction_scope": reaction_scope,
            "variables": list(variables),
            "target_nodes": list(target_nodes or []),
            "retrieved_count": len(units),
            "retrieved_card_ids": [unit["id"] for unit in units],
            "retrieved_mapping_statuses": [
                unit["metadata"].get("mapping_status") for unit in units
            ],
            "oracle_use_allowed": False,
        }
        return units, meta

    @staticmethod
    def _card_to_unit(card: EvidenceCard, *, rank: int) -> dict[str, Any]:
        payload = asdict(card)
        content_parts = [
            card.summary,
            f"Mapping status: {card.mapping_status}.",
            f"Allowed use: {card.allowed_use}.",
        ]
        if card.transferability_note:
            content_parts.append(f"Transferability: {card.transferability_note}")
        if card.supporting_excerpt:
            content_parts.append(f"Evidence excerpt: {card.supporting_excerpt}")
        return {
            "id": card.card_id,
            "content": " ".join(part.strip() for part in content_parts if str(part).strip()),
            "source_type": "lab_evidence_card",
            "confidence": card.confidence,
            "score": None,
            "rank": rank,
            "metadata": {
                **payload,
                "oracle_use_allowed": False,
            },
        }
