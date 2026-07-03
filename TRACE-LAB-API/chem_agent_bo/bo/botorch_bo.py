"""BoTorch-style finite-pool BO planners aligned with ChemBOMAS basic BO."""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from olympus.objects import ParameterVector

from chem_agent_bo.bo.base import BasePlanner

try:
    from botorch.acquisition import qLogExpectedImprovement

    HAS_QLOGEI = True
except Exception:  # noqa: BLE001
    qLogExpectedImprovement = None
    HAS_QLOGEI = False


class BoTorchFinitePoolPlanner(BasePlanner):
    """Finite-pool GP planner using BoTorch acquisitions over explicit candidates."""

    def __init__(
        self,
        *,
        seed: int = 7,
        goal: str = "maximize",
        num_init_design: int = 5,
        known_constraints=None,  # noqa: ANN001
        planner_name: str = "botorch_qei",
        acquisition_mode: str = "qei",
    ) -> None:
        self._seed = int(seed)
        self._goal = str(goal).strip().lower()
        self._num_init_design = int(num_init_design)
        self._known_constraints = list(known_constraints or [])
        self._constraint_signature = self._signature_for_constraints(self._known_constraints)
        self._rng = random.Random(self._seed)
        self._encoding_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._planner_refresh_count = 0
        self._last_planner_refreshed = False
        self._last_planner_refresh_reason = "init"
        self._last_fit_ok = False
        self._last_fallback_reason: str | None = None
        self._last_candidate_count = 0
        self._last_scored_count = 0
        self._planner_name = str(planner_name)
        self._acquisition_mode = str(acquisition_mode).strip().lower()
        self._runtime_env_name = os.path.basename(os.path.dirname(os.path.dirname(sys.executable))) or "unknown"
        self._runtime_python = sys.executable
        self._acquisition_cls = self._resolve_acquisition_class()
        self._acquisition_name = (
            "qLogExpectedImprovement" if self._acquisition_mode == "qlogei" else "qExpectedImprovement"
        )

    def _resolve_acquisition_class(self):
        if self._acquisition_mode == "qei":
            return qExpectedImprovement
        if self._acquisition_mode == "qlogei":
            if not HAS_QLOGEI or qLogExpectedImprovement is None:
                raise RuntimeError(
                    "Planner `botorch_qlogei` requires botorch.acquisition.qLogExpectedImprovement, "
                    f"but it is unavailable in env `{self._runtime_env_name}` "
                    f"({self._runtime_python})."
                )
            return qLogExpectedImprovement
        raise ValueError(f"Unsupported BoTorch acquisition_mode `{self._acquisition_mode}`.")

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._encoding_cache.clear()
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

    def _encoding_bundle(self, subspace) -> dict[str, Any]:  # noqa: ANN001
        signature = self._space_signature(subspace)
        cached = self._encoding_cache.get(signature)
        if cached is not None:
            return cached

        names: list[str] = []
        categories: dict[str, list[str]] = {}
        candidates: list[dict[str, Any]] = [{}]
        feature_dim = 0
        for param in subspace:
            options = list(getattr(param, "options", []) or [])
            if not options:
                raise ValueError(
                    "BoTorchFinitePoolPlanner currently supports finite-pool categorical spaces only."
                )
            str_options = [str(option) for option in options]
            names.append(param.name)
            categories[param.name] = str_options
            next_candidates: list[dict[str, Any]] = []
            for base in candidates:
                for option in str_options:
                    next_candidates.append({**base, param.name: option})
            candidates = next_candidates
            feature_dim += len(str_options)

        bundle = {
            "signature": signature,
            "param_names": names,
            "categories": categories,
            "candidates": candidates,
            "feature_dim": feature_dim,
        }
        self._encoding_cache[signature] = bundle
        return bundle

    def _encode_candidates(self, candidates: list[dict[str, Any]], bundle: dict[str, Any]) -> torch.Tensor:
        rows: list[list[float]] = []
        for candidate in candidates:
            row: list[float] = []
            for name in bundle["param_names"]:
                value = str(candidate.get(name))
                options = bundle["categories"][name]
                row.extend(1.0 if value == option else 0.0 for option in options)
            rows.append(row)
        if not rows:
            feature_dim = int(bundle["feature_dim"])
            return torch.empty((0, feature_dim), dtype=torch.double)
        return torch.tensor(rows, dtype=torch.double)

    def _legal_unseen_candidates(
        self,
        observations,  # noqa: ANN001
        subspace,  # noqa: ANN001
        known_constraints,  # noqa: ANN001
        bundle: dict[str, Any],
    ) -> list[dict[str, Any]]:
        observed = {tuple(str(item) for item in row) for row in observations.get_params(as_array=True)}
        legal: list[dict[str, Any]] = []
        for candidate in bundle["candidates"]:
            if known_constraints and not all(constraint(candidate) for constraint in known_constraints):
                continue
            if self._sample_key(candidate, subspace) in observed:
                continue
            legal.append(candidate)
        return legal

    def _observed_training_data(
        self,
        observations,  # noqa: ANN001
        subspace,  # noqa: ANN001
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        params = observations.get_params(as_array=True)
        values = observations.get_values(as_array=True)
        if len(values) == 0:
            return [], np.asarray([], dtype=float)
        records = []
        for row in params:
            records.append({param.name: str(row[idx]) for idx, param in enumerate(subspace)})
        return records, np.asarray(values, dtype=float).reshape(-1)

    def _fit_and_score(
        self,
        train_candidates: list[dict[str, Any]],
        train_values: np.ndarray,
        legal_candidates: list[dict[str, Any]],
        bundle: dict[str, Any],
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

        train_x = self._encode_candidates(train_candidates, bundle)
        candidate_x = self._encode_candidates(legal_candidates, bundle)
        transformed_y = np.asarray(train_values, dtype=float)
        if self._goal == "minimize":
            transformed_y = -transformed_y
        train_y = torch.tensor(transformed_y, dtype=torch.double).view(-1, 1)
        try:
            torch.manual_seed(self._seed)
            model = SingleTaskGP(
                train_x,
                train_y,
                input_transform=Normalize(d=train_x.shape[-1]),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            acq = self._acquisition_cls(model=model, best_f=train_y.max())
            choices = candidate_x.unsqueeze(-2)
            with torch.no_grad():
                scores = acq(choices).detach().cpu().reshape(-1)
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                raise ValueError(f"{self._acquisition_name} produced non-finite scores.")
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
            self._last_fallback_reason = f"{self._planner_name}_fit_or_acq_failed"
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

        bundle = self._encoding_bundle(subspace)
        legal_candidates = self._legal_unseen_candidates(
            observations=observations,
            subspace=subspace,
            known_constraints=self._known_constraints,
            bundle=bundle,
        )
        self._last_candidate_count = len(legal_candidates)
        train_candidates, train_values = self._observed_training_data(observations, subspace)
        ranked = self._fit_and_score(train_candidates, train_values, legal_candidates, bundle)
        self._last_scored_count = len(ranked)
        limit = max(1, int(shortlist_size))
        return [
            self._to_parameter_vector(candidate, subspace)
            for _, candidate in ranked[:limit]
        ]

    def planner_diagnostics(self) -> dict[str, Any]:
        return {
            "planner_name": self._planner_name,
            "planner_family": "bo",
            "supports_shortlist": True,
            "search_space_type": "categorical_finite_pool",
            "candidate_pool_mode": "explicit_enumeration",
            "surrogate_name": "SingleTaskGP",
            "acquisition_name": self._acquisition_name,
            "encoding_name": "global_one_hot",
            "constraint_signature": self._constraint_signature,
            "planner_refresh_count": self._planner_refresh_count,
            "planner_refreshed": self._last_planner_refreshed,
            "planner_refresh_reason": self._last_planner_refresh_reason,
            "fit_ok": self._last_fit_ok,
            "fallback_reason": self._last_fallback_reason,
            "candidate_count": self._last_candidate_count,
            "scored_candidate_count": self._last_scored_count,
            "runtime_env": self._runtime_env_name,
            "runtime_python": self._runtime_python,
        }
