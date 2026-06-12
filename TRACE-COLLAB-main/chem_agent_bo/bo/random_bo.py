"""Random planner for protocol and baseline comparisons."""

from __future__ import annotations

import random
from typing import Any

from olympus.objects import ParameterVector

from chem_agent_bo.bo.base import BasePlanner


class RandomPlanner(BasePlanner):
    def __init__(self, *, seed: int = 7, goal: str = "maximize") -> None:
        self._seed = int(seed)
        self._goal = goal
        self._rng = random.Random(self._seed)

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    @staticmethod
    def _sample_key(values: Any, param_space) -> tuple[str, ...]:  # noqa: ANN001
        if hasattr(values, "to_dict"):
            try:
                data = values.to_dict()
                return tuple(str(data[param.name]) for param in param_space)
            except Exception:  # noqa: BLE001
                pass
        if isinstance(values, dict):
            return tuple(str(values[param.name]) for param in param_space)
        return tuple(str(getattr(values, param.name)) for param in param_space)

    def _observed_keys(self, observations, subspace) -> set[tuple[str, ...]]:  # noqa: ANN001
        params = observations.get_params(as_array=True)
        keys: set[tuple[str, ...]] = set()
        for row in params:
            keys.add(tuple(str(row[idx]) for idx, _ in enumerate(subspace)))
        return keys

    def _random_candidate(self, subspace, known_constraints, banned_keys):  # noqa: ANN001
        for _ in range(2048):
            data: dict[str, Any] = {}
            for param in subspace:
                if param.type == "categorical":
                    data[param.name] = self._rng.choice(list(param.options))
                elif param.type == "discrete":
                    options = list(getattr(param, "options", []) or [])
                    data[param.name] = self._rng.choice(options) if options else param.low
                elif param.type == "continuous":
                    data[param.name] = self._rng.uniform(float(param.low), float(param.high))
                else:
                    data[param.name] = None
            if known_constraints and not all(constraint(data) for constraint in known_constraints):
                continue
            key = self._sample_key(data, subspace)
            if key in banned_keys:
                continue
            return ParameterVector().from_dict(data, param_space=subspace)
        raise RuntimeError("RandomPlanner failed to sample a feasible candidate.")

    def suggest(
        self,
        observations,  # noqa: ANN001
        subspace,  # noqa: ANN001
        known_constraints=None,  # noqa: ANN001
        known_constraints_signature: str | None = None,
    ):
        del known_constraints_signature
        observed = self._observed_keys(observations, subspace)
        return [self._random_candidate(subspace, known_constraints, observed)]

    def suggest_shortlist(
        self,
        observations,  # noqa: ANN001
        subspace,  # noqa: ANN001
        shortlist_size: int,
        known_constraints=None,  # noqa: ANN001
        known_constraints_signature: str | None = None,
    ):
        del known_constraints_signature
        observed = self._observed_keys(observations, subspace)
        shortlist = []
        for _ in range(max(1, int(shortlist_size))):
            candidate = self._random_candidate(subspace, known_constraints, observed)
            observed.add(self._sample_key(candidate, subspace))
            shortlist.append(candidate)
        return shortlist

    def planner_diagnostics(self) -> dict[str, Any]:
        return {
            "planner_name": "random",
            "planner_family": "random",
            "supports_shortlist": True,
            "search_space_type": "mixed",
            "candidate_pool_mode": "random_sampling",
            "surrogate_name": "none",
            "acquisition_name": "none",
            "encoding_name": "none",
            "goal": self._goal,
            "planner_refreshed": False,
            "planner_refresh_reason": "not_applicable",
            "planner_refresh_count": 0,
            "constraint_signature": "__random__",
        }
