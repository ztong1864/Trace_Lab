"""Planner interface shared by BO backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePlanner(ABC):
    """Common planner interface for baseline and agentic runs."""

    @abstractmethod
    def suggest(
        self,
        observations,  # noqa: ANN001
        subspace,  # noqa: ANN001
        known_constraints=None,  # noqa: ANN001
        known_constraints_signature: str | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def suggest_shortlist(
        self,
        observations,  # noqa: ANN001
        subspace,  # noqa: ANN001
        shortlist_size: int,
        known_constraints=None,  # noqa: ANN001
        known_constraints_signature: str | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def planner_diagnostics(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
