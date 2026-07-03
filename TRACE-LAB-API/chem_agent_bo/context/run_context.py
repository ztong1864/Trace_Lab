"""Compatibility builders for LLM decision context."""

from __future__ import annotations

from typing import Any

from chem_agent_bo.config.schema import PromptConfig
from chem_agent_bo.state import OnlineDecisionState


def build_decision_context(
    history: list[dict[str, Any]],
    working_memory_summary: dict[str, Any],
    best_observation: dict[str, Any] | None,
    iteration: int,
    search_space=None,  # noqa: ANN001
    search_space_meta: dict[str, Any] | None = None,
    goal: str = "maximize",
    knowledge_units: list[dict[str, Any]] | None = None,
    knowledge_query: str | None = None,
    knowledge_meta: dict[str, Any] | None = None,
    prompt_config: PromptConfig | None = None,
) -> dict[str, Any]:
    """Build the legacy decision_context view from online decision state."""
    state = OnlineDecisionState(prompt_config=prompt_config or PromptConfig())
    state.load_working_memory_summary(working_memory_summary)
    state.refresh_from_history(
        bootstrap_history=None,
        history=history,
        best_observation=best_observation,
        current_best=None,
        iteration=iteration,
        observations=len(history),
        total_budget=None,
        search_space=search_space,
        search_space_meta=search_space_meta,
        goal=goal,
        knowledge_units=knowledge_units,
        knowledge_query=knowledge_query,
        knowledge_meta=knowledge_meta,
    )
    return state.build_decision_context(search_space_meta=search_space_meta)
