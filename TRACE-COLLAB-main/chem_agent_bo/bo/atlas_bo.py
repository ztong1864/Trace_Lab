"""Atlas GPPlanner wrapper for candidate suggestion."""

from __future__ import annotations

import json
from typing import Any, Callable

from atlas.planners.gp.planner import GPPlanner

from chem_agent_bo.bo.base import BasePlanner


class AtlasBOTool(BasePlanner):
    """A minimal wrapper around Atlas GPPlanner."""

    def __init__(
        self,
        num_init_design: int = 5,
        seed: int = 7,
        goal: str = "maximize",
        known_constraints: list[Callable[[Any], bool]] | None = None,
        use_descriptors: bool = False,
    ) -> None:
        self._num_init_design = num_init_design
        self._seed = seed
        self._goal = goal
        self._known_constraints = list(known_constraints or [])
        self._use_descriptors = bool(use_descriptors)
        self._constraint_signature = self._signature_for_constraints(self._known_constraints)
        self._planner_refresh_count = 0
        self._last_planner_refresh_reason = "init"
        self._last_planner_refreshed = True
        self.planner = self._build_planner()

    def _build_planner(self, batch_size: int | None = None) -> GPPlanner:
        return GPPlanner(
            goal=self._goal,
            init_design_strategy="random",
            acquisition_type="ei",
            acquisition_optimizer_kind="gradient",
            num_init_design=self._num_init_design,
            batch_size=batch_size or 1,
            random_seed=self._seed,
            known_constraints=self._known_constraints,
            use_descriptors=self._use_descriptors,
        )

    def reset(self) -> None:
        self.planner = self._build_planner()
        self._planner_refresh_count += 1
        self._last_planner_refreshed = True
        self._last_planner_refresh_reason = "reset"

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
        if all(hasattr(values, param.name) for param in param_space):
            return tuple(str(getattr(values, param.name)) for param in param_space)
        if hasattr(values, "tolist"):
            arr = values.tolist()
            if isinstance(arr, list):
                return tuple(str(item) for item in arr)
        if isinstance(values, (list, tuple)):
            return tuple(str(item) for item in values)
        raise TypeError(f"Unsupported sample type for shortlist key extraction: {type(values)}")

    @staticmethod
    def _signature_for_constraints(
        known_constraints: list[Callable[[Any], bool]] | None,
    ) -> str:
        if not known_constraints:
            return "__none__"
        # Fall back to object identity when no semantic signature is supplied.
        return json.dumps([id(item) for item in known_constraints])

    def set_known_constraints(
        self,
        known_constraints: list[Callable[[Any], bool]] | None,
        *,
        signature: str | None = None,
        refresh_reason: str = "constraints_changed",
    ) -> None:
        next_signature = signature or self._signature_for_constraints(known_constraints)
        if next_signature == self._constraint_signature:
            self._last_planner_refreshed = False
            self._last_planner_refresh_reason = "constraints_unchanged"
            return
        self._known_constraints = list(known_constraints or [])
        self._constraint_signature = next_signature
        self.planner = self._build_planner()
        self._planner_refresh_count += 1
        self._last_planner_refreshed = True
        self._last_planner_refresh_reason = refresh_reason

    def suggest(
        self,
        observations,  # noqa: ANN001
        subspace,  # noqa: ANN001
        known_constraints: list[Callable[[Any], bool]] | None = None,
        known_constraints_signature: str | None = None,
    ):  # noqa: ANN201
        self._last_planner_refreshed = False
        self._last_planner_refresh_reason = "reused_existing_planner"
        if known_constraints is not None:
            self.set_known_constraints(
                known_constraints,
                signature=known_constraints_signature,
            )
        self.planner.set_param_space(subspace)
        return self.planner.recommend(observations)

    def suggest_shortlist(
        self,
        observations,  # noqa: ANN001
        subspace,  # noqa: ANN001
        shortlist_size: int,
        known_constraints: list[Callable[[Any], bool]] | None = None,
        known_constraints_signature: str | None = None,
    ):  # noqa: ANN201
        size = max(1, int(shortlist_size))
        if size == 1:
            return self.suggest(
                observations=observations,
                subspace=subspace,
                known_constraints=known_constraints,
                known_constraints_signature=known_constraints_signature,
            )
        self._last_planner_refreshed = False
        self._last_planner_refresh_reason = "reused_existing_planner"
        if known_constraints is not None:
            self.set_known_constraints(
                known_constraints,
                signature=known_constraints_signature,
            )
        shortlist: list[Any] = []
        excluded_keys: set[tuple[str, ...]] = set()
        base_constraints = list(self._known_constraints)
        for _ in range(size):
            iter_constraints = list(base_constraints)
            if excluded_keys:
                disallowed = set(excluded_keys)

                def _exclude_selected(values: Any, *, _disallowed=disallowed) -> bool:
                    return self._sample_key(values, subspace) not in _disallowed

                iter_constraints.append(_exclude_selected)
            shortlist_planner = GPPlanner(
                goal=self._goal,
                init_design_strategy="random",
                acquisition_type="ei",
                acquisition_optimizer_kind="gradient",
                num_init_design=self._num_init_design,
                batch_size=1,
                random_seed=self._seed,
                known_constraints=iter_constraints,
                use_descriptors=self._use_descriptors,
            )
            shortlist_planner.set_param_space(subspace)
            try:
                samples = shortlist_planner.recommend(observations)
            except IndexError:
                break
            if not samples:
                break
            sample = samples[0]
            sample_key = self._sample_key(sample, subspace)
            if sample_key in excluded_keys:
                break
            excluded_keys.add(sample_key)
            shortlist.append(sample)
        return shortlist

    def planner_diagnostics(self) -> dict[str, Any]:
        return {
            "planner_name": "atlas",
            "planner_family": "bo",
            "supports_shortlist": True,
            "use_descriptors": self._use_descriptors,
            "search_space_type": "mixed",
            "candidate_pool_mode": "planner_internal",
            "surrogate_name": "AtlasGPPlanner",
            "acquisition_name": "ei",
            "encoding_name": "planner_internal",
            "constraint_signature": self._constraint_signature,
            "planner_refresh_count": self._planner_refresh_count,
            "planner_refreshed": self._last_planner_refreshed,
            "planner_refresh_reason": self._last_planner_refresh_reason,
        }
