"""Finite-pool discrete BO planner with explicit acquisition enumeration."""

from __future__ import annotations

from itertools import product
import json
import random
from typing import Any

import numpy as np
from olympus.objects import ParameterVector
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from chem_agent_bo.bo.base import BasePlanner


class DiscreteBOPlanner(BasePlanner):
    """Finite-pool BO planner that scores all legal unseen candidates with EI."""

    def __init__(
        self,
        *,
        seed: int = 7,
        goal: str = "maximize",
        num_init_design: int = 5,
        known_constraints=None,  # noqa: ANN001
    ) -> None:
        self._seed = int(seed)
        self._goal = str(goal).strip().lower()
        self._num_init_design = int(num_init_design)
        self._known_constraints = list(known_constraints or [])
        self._constraint_signature = self._signature_for_constraints(self._known_constraints)
        self._rng = random.Random(self._seed)
        self._candidate_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        self._planner_refresh_count = 0
        self._last_planner_refreshed = False
        self._last_planner_refresh_reason = "init"
        self._last_fit_ok = False
        self._last_fallback_reason: str | None = None
        self._last_candidate_count = 0
        self._last_scored_count = 0

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._candidate_cache.clear()
        self._planner_refresh_count += 1
        self._last_planner_refreshed = True
        self._last_planner_refresh_reason = "reset"
        self._last_fit_ok = False
        self._last_fallback_reason = None
        self._last_candidate_count = 0
        self._last_scored_count = 0

    @staticmethod
    def _signature_for_constraints(known_constraints) -> str:  # noqa: ANN001
        if not known_constraints:
            return "__none__"
        return json.dumps([id(item) for item in known_constraints])

    def set_known_constraints(
        self,
        known_constraints,  # noqa: ANN001
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
        self._planner_refresh_count += 1
        self._last_planner_refreshed = True
        self._last_planner_refresh_reason = refresh_reason

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
        raise TypeError(f"Unsupported sample type for key extraction: {type(values)}")

    @staticmethod
    def _to_parameter_vector(candidate: dict[str, Any], param_space):  # noqa: ANN001
        return ParameterVector().from_dict(candidate, param_space=param_space)

    def _space_signature(self, subspace) -> tuple[Any, ...]:  # noqa: ANN001
        signature: list[Any] = []
        for param in subspace:
            options = tuple(str(item) for item in list(getattr(param, "options", []) or []))
            signature.append((param.name, getattr(param, "type", None), options))
        return tuple(signature)

    def _enumerate_candidates(self, subspace) -> list[dict[str, Any]]:  # noqa: ANN001
        signature = self._space_signature(subspace)
        cached = self._candidate_cache.get(signature)
        if cached is not None:
            return list(cached)
        param_options: list[list[Any]] = []
        param_names: list[str] = []
        for param in subspace:
            options = list(getattr(param, "options", []) or [])
            if not options:
                raise ValueError(
                    "DiscreteBOPlanner currently supports finite-pool/categorical spaces with explicit options only."
                )
            param_options.append(options)
            param_names.append(param.name)
        candidates = [
            {name: value for name, value in zip(param_names, combo, strict=False)}
            for combo in product(*param_options)
        ]
        self._candidate_cache[signature] = list(candidates)
        return candidates

    def _legal_unseen_candidates(
        self,
        observations,  # noqa: ANN001
        subspace,  # noqa: ANN001
        known_constraints,  # noqa: ANN001
    ) -> list[dict[str, Any]]:
        observed = {
            tuple(str(item) for item in row)
            for row in observations.get_params(as_array=True)
        }
        legal: list[dict[str, Any]] = []
        for candidate in self._enumerate_candidates(subspace):
            if known_constraints and not all(constraint(candidate) for constraint in known_constraints):
                continue
            if self._sample_key(candidate, subspace) in observed:
                continue
            legal.append(candidate)
        return legal

    def _observed_training_data(self, observations, subspace) -> tuple[list[dict[str, Any]], np.ndarray]:  # noqa: ANN001
        params = observations.get_params(as_array=True)
        values = observations.get_values(as_array=True)
        if len(values) == 0:
            return [], np.asarray([], dtype=float)
        records = []
        for row in params:
            records.append({param.name: row[idx] for idx, param in enumerate(subspace)})
        flat_values = np.asarray(values, dtype=float).reshape(-1)
        return records, flat_values

    @staticmethod
    def _feature_layout(subspace) -> list[tuple[str, str | None, list[str] | None]]:  # noqa: ANN001
        layout: list[tuple[str, str | None, list[str] | None]] = []
        for param in subspace:
            options = list(getattr(param, "options", []) or [])
            if options:
                layout.append((param.name, "categorical", [str(option) for option in options]))
            elif hasattr(param, "low") and hasattr(param, "high"):
                layout.append((param.name, "continuous", None))
            else:
                raise ValueError(
                    "DiscreteBOPlanner only supports categorical/discrete options and simple continuous values."
                )
        return layout

    @staticmethod
    def _encode_candidates(
        candidates: list[dict[str, Any]],
        layout: list[tuple[str, str | None, list[str] | None]],
    ) -> np.ndarray:
        rows: list[list[float]] = []
        for candidate in candidates:
            row: list[float] = []
            for name, kind, options in layout:
                if kind == "categorical" and options is not None:
                    value = str(candidate.get(name))
                    row.extend(1.0 if value == option else 0.0 for option in options)
                else:
                    row.append(float(candidate.get(name)))
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def _score_candidates(
        self,
        train_candidates: list[dict[str, Any]],
        train_values: np.ndarray,
        legal_candidates: list[dict[str, Any]],
        subspace,  # noqa: ANN001
    ) -> list[tuple[float, dict[str, Any]]]:
        if len(legal_candidates) == 0:
            self._last_fit_ok = False
            self._last_fallback_reason = "no_legal_candidates"
            return []
        if len(train_candidates) < 2 or len(np.unique(train_values)) < 2:
            self._last_fit_ok = False
            self._last_fallback_reason = "insufficient_observations"
            shuffled = list(legal_candidates)
            self._rng.shuffle(shuffled)
            return [(float("nan"), item) for item in shuffled]

        layout = self._feature_layout(subspace)
        train_x = self._encode_candidates(train_candidates, layout)
        cand_x = self._encode_candidates(legal_candidates, layout)
        dim = max(1, train_x.shape[1])
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=self._seed,
            n_restarts_optimizer=1,
        )
        try:
            gp.fit(train_x, train_values)
            mu, std = gp.predict(cand_x, return_std=True)
            std = np.maximum(std, 1e-12)
            best = float(np.min(train_values) if self._goal == "minimize" else np.max(train_values))
            improvement = best - mu if self._goal == "minimize" else mu - best
            z = improvement / std
            scores = improvement * norm.cdf(z) + std * norm.pdf(z)
            self._last_fit_ok = True
            self._last_fallback_reason = None
            ranked = sorted(
                zip(scores.tolist(), legal_candidates, strict=False),
                key=lambda item: float(item[0]),
                reverse=True,
            )
            return [(float(score), candidate) for score, candidate in ranked]
        except Exception:  # noqa: BLE001
            self._last_fit_ok = False
            self._last_fallback_reason = "gp_fit_failed"
            shuffled = list(legal_candidates)
            self._rng.shuffle(shuffled)
            return [(float("nan"), item) for item in shuffled]

    def suggest(
        self,
        observations,  # noqa: ANN001
        subspace,  # noqa: ANN001
        known_constraints=None,  # noqa: ANN001
        known_constraints_signature: str | None = None,
    ):
        shortlist = self.suggest_shortlist(
            observations=observations,
            subspace=subspace,
            shortlist_size=1,
            known_constraints=known_constraints,
            known_constraints_signature=known_constraints_signature,
        )
        return shortlist[:1]

    def suggest_shortlist(
        self,
        observations,  # noqa: ANN001
        subspace,  # noqa: ANN001
        shortlist_size: int,
        known_constraints=None,  # noqa: ANN001
        known_constraints_signature: str | None = None,
    ):
        self._last_planner_refreshed = False
        self._last_planner_refresh_reason = "reused_existing_planner"
        if known_constraints is not None:
            self.set_known_constraints(
                known_constraints,
                signature=known_constraints_signature,
            )
        legal_candidates = self._legal_unseen_candidates(
            observations=observations,
            subspace=subspace,
            known_constraints=self._known_constraints,
        )
        self._last_candidate_count = len(legal_candidates)
        train_candidates, train_values = self._observed_training_data(observations, subspace)
        ranked = self._score_candidates(train_candidates, train_values, legal_candidates, subspace)
        limit = max(1, int(shortlist_size))
        shortlist = [
            self._to_parameter_vector(candidate, subspace)
            for _, candidate in ranked[:limit]
        ]
        self._last_scored_count = len(ranked)
        return shortlist

    def planner_diagnostics(self) -> dict[str, Any]:
        return {
            "planner_name": "discrete",
            "planner_family": "bo",
            "supports_shortlist": True,
            "search_space_type": "finite_pool_categorical",
            "candidate_pool_mode": "explicit_enumeration",
            "surrogate_name": "GaussianProcessRegressor",
            "acquisition_name": "ei",
            "encoding_name": "global_one_hot",
            "constraint_signature": self._constraint_signature,
            "planner_refresh_count": self._planner_refresh_count,
            "planner_refreshed": self._last_planner_refreshed,
            "planner_refresh_reason": self._last_planner_refresh_reason,
            "fit_ok": self._last_fit_ok,
            "fallback_reason": self._last_fallback_reason,
            "candidate_count": self._last_candidate_count,
            "scored_candidate_count": self._last_scored_count,
            "goal": self._goal,
        }
