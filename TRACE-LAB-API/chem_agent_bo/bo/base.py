"""Planner interface shared by BO backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class ExclusionConstraint:
    """Known constraint that bans specific combinations of option values.

    It is callable like any known constraint (returns False for a banned
    candidate), and it also exposes the banned keys together with the variable
    names they are ordered by, so vectorized planners (chunked_gp) can apply the
    exclusion to a whole chunk at once and match key positions by name.
    """

    def __init__(
        self,
        *,
        variable_names: list[str],
        excluded_keys: set[tuple[str, ...]],
        normalize_value: Callable[[Any], str],
        check: Callable[[Any], bool],
    ) -> None:
        self.variable_names = list(variable_names)
        self.excluded_keys = set(excluded_keys)
        self.normalize_value = normalize_value
        self._check = check

    def __call__(self, values: Any) -> bool:
        return self._check(values)


def normalize_acquisition_type(value: Any, *, supported: tuple[str, ...], planner_name: str) -> str:
    """Normalize an acquisition name and check it against one planner's supported set."""
    name = str(value or "ei").strip().lower()
    if name not in supported:
        raise ValueError(
            f"The {planner_name} planner does not support acquisition function `{name}`; "
            f"expected one of {list(supported)}."
        )
    return name


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
