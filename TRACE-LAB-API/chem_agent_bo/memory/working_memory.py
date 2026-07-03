"""Lightweight run-level working memory."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class WorkingMemory:
    """In-memory adapter for short-term decision memory."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {
            "research_state": {
                "current_phase": "full_space_search",
                "promising_regions": [],
                "weak_regions": [],
                "trusted_patterns": [],
                "open_hypotheses": [],
                "suggested_next_focus": [],
                "confidence": "low",
                "summary": "",
            },
            "last_hypothesis": None,
            "last_diagnosis": None,
            "last_coverage": None,
            "last_semantic_assessment": None,
            "last_reflection": None,
            "notes": [],
        }

    def summarize(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def replace_state(self, new_state: dict[str, Any]) -> None:
        self._state = deepcopy(new_state)

    def get_research_state(self) -> dict[str, Any]:
        return dict(self._state["research_state"])

    def update_research_state(self, updates: dict[str, Any]) -> None:
        state = dict(self._state["research_state"])
        for key in (
            "current_phase",
            "promising_regions",
            "weak_regions",
            "trusted_patterns",
            "open_hypotheses",
            "suggested_next_focus",
            "confidence",
            "summary",
        ):
            if key in updates and updates[key] is not None:
                state[key] = updates[key]
        self._state["research_state"] = state

    def update_from_reflection(self, reflection_action: dict[str, Any]) -> None:
        self._state["last_reflection"] = reflection_action
        self.update_research_state(
            {
                "suggested_next_focus": reflection_action.get("suggested_focus", []),
                "summary": reflection_action.get("insight", ""),
                "confidence": reflection_action.get("confidence", "low"),
            }
        )
        note = {
            "insight": reflection_action.get("insight", ""),
            "next_step_hypothesis": reflection_action.get("next_step_hypothesis", ""),
            "confidence": reflection_action.get("confidence", "low"),
        }
        self._state["notes"] = (self._state["notes"] + [note])[-10:]

    def update_from_v030_actions(
        self,
        *,
        hypothesis_action: dict[str, Any] | None = None,
        diagnosis: dict[str, Any] | None = None,
        coverage_insight: dict[str, Any] | None = None,
        semantic_assessment: dict[str, Any] | None = None,
        reflection_action: dict[str, Any] | None = None,
    ) -> None:
        if hypothesis_action is not None:
            self._state["last_hypothesis"] = hypothesis_action
            hypotheses = hypothesis_action.get("hypotheses", [])
            self.update_research_state(
                {
                    "open_hypotheses": hypotheses,
                    "suggested_next_focus": hypothesis_action.get(
                        "suggested_focus_variables",
                        [],
                    ),
                }
            )

        if diagnosis is not None:
            self._state["last_diagnosis"] = diagnosis
            current_phase = "full_space_search"
            if diagnosis.get("is_stagnating", False):
                current_phase = "stagnation_recovery"
            self.update_research_state({"current_phase": current_phase})

        if coverage_insight is not None:
            self._state["last_coverage"] = coverage_insight
            self.update_research_state(
                {
                    "weak_regions": coverage_insight.get("underexplored_dimensions", []),
                }
            )

        if semantic_assessment is not None:
            self._state["last_semantic_assessment"] = semantic_assessment

        if reflection_action is not None:
            self.update_from_reflection(reflection_action)
