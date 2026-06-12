"""Experience promotion policy for fair knowledge accumulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExperienceCandidate:
    """Candidate memory distilled from runtime traces."""

    id: str
    dataset: str
    trigger_reasons: list[str]
    reflection_insight: str
    reasoning: str
    confidence: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "trigger_reasons": list(self.trigger_reasons),
            "reflection_insight": self.reflection_insight,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


def build_experience_candidate(
    *,
    iteration: int,
    dataset: str,
    trigger_reasons: list[str],
    reflection_action: dict[str, Any] | None,
    intervention_plan: dict[str, Any] | None,
) -> ExperienceCandidate | None:
    reflection = reflection_action or {}
    insight = str(reflection.get("insight", "")).strip()
    if not insight:
        return None
    return ExperienceCandidate(
        id=f"{dataset}_iter_{iteration}",
        dataset=dataset,
        trigger_reasons=list(trigger_reasons),
        reflection_insight=insight,
        reasoning=str((intervention_plan or {}).get("reasoning", "")),
        confidence=str(reflection.get("confidence", "low")),
        metadata={
            "next_step_hypothesis": reflection.get("next_step_hypothesis", ""),
            "suggested_focus": reflection.get("suggested_focus", []),
            "avoid_pattern": reflection.get("avoid_pattern", []),
        },
    )


def should_promote_experience(candidate: ExperienceCandidate) -> bool:
    """Conservative promotion gate to avoid polluting static knowledge."""
    if candidate.confidence.lower() not in {"high", "medium"}:
        return False
    if len(candidate.trigger_reasons) == 0:
        return False
    if len(candidate.reflection_insight) < 60:
        return False
    return True
