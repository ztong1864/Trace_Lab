"""Reaction environment wrapper for emulator/lookup/finite-pool backends."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from olympus import Dataset
from olympus.emulators import Emulator
from olympus.objects import ParameterVector

from chem_agent_bo.datasets import (
    get_finite_pool_dataset_names,
    load_finite_pool_table,
)

LOOKUP_DATASETS = {
    "buchwald_a",
    "buchwald_b",
    "buchwald_c",
    "buchwald_d",
    "buchwald_e",
}


class ReactionEnv:
    """Thin wrapper for Olympus and finite-pool execution."""

    def __init__(self, dataset: str = "suzuki") -> None:
        self.dataset = dataset
        self._finite_pool_table = None
        finite_pool_names = set(get_finite_pool_dataset_names())
        if dataset in finite_pool_names:
            self._backend = "finite_pool"
        elif dataset in LOOKUP_DATASETS:
            self._backend = "lookup"
        else:
            self._backend = "emulator"
        if self._backend == "lookup":
            self._lookup_dataset = Dataset(kind=dataset)
            self._emulator = None
            self._objective_count = len(self._lookup_dataset.target_names)
            self._goal = self._infer_lookup_goal()
        elif self._backend == "finite_pool":
            self._lookup_dataset = None
            self._emulator = None
            self._finite_pool_table = load_finite_pool_table(dataset)
            self._objective_count = 1
            self._goal = self._finite_pool_table.goal
        else:
            self._emulator = Emulator(dataset=dataset, model="BayesNeuralNet")
            self._lookup_dataset = None
            self._objective_count = len(self._emulator.value_space)
            self._goal = self._infer_emulator_goal()
        if self._objective_count != 1:
            raise ValueError(
                f"Dataset `{dataset}` has {self._objective_count} objectives. "
                "Current Agentic BO pipeline supports single-objective datasets only."
            )

    def _infer_emulator_goal(self) -> str:
        try:
            goal = str(self._emulator.get_goal()).strip().lower()
        except Exception:  # noqa: BLE001
            goal = "maximize"
        if goal in {"maximize", "max"}:
            return "maximize"
        if goal in {"minimize", "min"}:
            return "minimize"
        return "maximize"

    def _infer_lookup_goal(self) -> str:
        objective_name = self.objective_name.lower()
        if "impurity" in objective_name or "error" in objective_name:
            return "minimize"
        return "maximize"

    @property
    def param_space(self):  # noqa: ANN201
        if self._backend == "finite_pool":
            return self._finite_pool_table.param_space
        if self._backend == "lookup":
            return self._lookup_dataset.param_space
        return self._emulator.param_space

    @property
    def value_space(self):  # noqa: ANN201
        if self._backend == "finite_pool":
            return self._finite_pool_table.value_space
        if self._backend == "lookup":
            return self._lookup_dataset.value_space
        return self._emulator.value_space

    @property
    def goal(self) -> str:
        return self._goal

    @property
    def objective_name(self) -> str:
        if self._backend == "finite_pool":
            return self._finite_pool_table.target_column
        if self._backend == "lookup":
            return self._lookup_dataset.target_names[0]
        return self.value_space[0].name

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_finite_pool(self) -> bool:
        return self._backend == "finite_pool"

    @property
    def candidate_count(self) -> int | None:
        if self._backend != "finite_pool":
            return None
        return self._finite_pool_table.candidate_count

    def candidate_keys(self) -> set[tuple[str, ...]] | None:
        if self._backend != "finite_pool":
            return None
        return set(self._finite_pool_table.record_keys())

    def allowed_keys_signature(self, allowed_keys: set[tuple[str, ...]] | None) -> str | None:
        if self._backend != "finite_pool" or allowed_keys is None:
            return None
        return self._finite_pool_table.allowed_keys_signature(allowed_keys)

    def build_membership_constraint(self, allowed_keys: set[tuple[str, ...]] | None):  # noqa: ANN201
        if self._backend != "finite_pool" or allowed_keys is None:
            return None
        return self._finite_pool_table.membership_constraint(allowed_keys=allowed_keys)

    def dataset_meta(self) -> dict[str, Any]:
        if self._backend == "finite_pool":
            return self._finite_pool_table.dataset_meta()
        if self._backend == "lookup":
            return {
                "backend": "olympus_lookup",
                "dataset_name": self.dataset,
                "candidate_count": None,
            }
        return {
            "backend": "emulator",
            "dataset_name": self.dataset,
            "candidate_count": None,
        }

    def is_valid_candidate(self, sample: Any) -> bool:
        if self._backend != "finite_pool":
            return True
        return self._finite_pool_table.is_valid_candidate(sample)

    def filter_candidates(
        self,
        *,
        focus_variables: list[str] | None = None,
        best_observation: dict[str, Any] | None = None,
        filter_mode: str = "exact_match",
    ) -> tuple[set[tuple[str, ...]] | None, bool]:
        if self._backend != "finite_pool":
            return None, False
        keys, fallback = self._finite_pool_table.filter_keys_by_focus(
            focus_variables=focus_variables,
            best_observation=best_observation,
            filter_mode=filter_mode,
        )
        return keys, fallback

    def build_known_constraints(
        self,
        *,
        focus_variables: list[str] | None = None,
        best_observation: dict[str, Any] | None = None,
        filter_mode: str = "exact_match",
    ) -> tuple[list[Callable[[Any], bool]], dict[str, Any]]:
        if self._backend != "finite_pool":
            return [], {
                "mode": "none",
                "pool_size": None,
                "total_candidate_count": None,
                "focus_variables": [],
                "filter_mode": filter_mode,
                "pool_size_before": None,
                "pool_size_after": None,
            }
        allowed_keys, subpool_empty_fallback = self._finite_pool_table.filter_keys_by_focus(
            focus_variables=focus_variables,
            best_observation=best_observation,
            filter_mode=filter_mode,
        )
        constraint = self._finite_pool_table.membership_constraint(allowed_keys=allowed_keys)
        mode = "focused_filter" if focus_variables else "full_pool"
        summary = {
            "mode": mode,
            "filter_mode": filter_mode,
            "pool_size": len(allowed_keys),
            "pool_size_before": self._finite_pool_table.candidate_count,
            "pool_size_after": len(allowed_keys),
            "total_candidate_count": self._finite_pool_table.candidate_count,
            "focus_variables": list(focus_variables or []),
            "subpool_empty_fallback": subpool_empty_fallback,
            "constraint_signature": self._finite_pool_table.allowed_keys_signature(allowed_keys),
        }
        return [constraint], summary

    def build_llm_search_constraints(
        self,
        constraint_specs: list[dict[str, Any]],
        min_pool_fraction: float = 0.05,
    ) -> tuple[list[Callable[[Any], bool]], dict[str, Any]]:
        """Convert LLM constraint specs to known_constraints callables.

        Each spec is a LLMConstraintSpec dict with fields:
        variable, constraint_type ("include_values"|"exclude_values"), values.

        Returns ([constraint_callable], summary_dict).
        summary_dict["fallback_triggered"]=True means constraints were ignored
        (pool too small) and the full candidate set is used.
        """
        if self._backend != "finite_pool" or not constraint_specs:
            return [], {"pool_size": None, "fallback_triggered": False, "constraint_summary": ""}

        allowed_keys, fallback_triggered, constraint_summary = self.filter_candidate_keys_by_constraint_specs(
            constraint_specs=constraint_specs,
            min_pool_fraction=min_pool_fraction,
        )
        if fallback_triggered:
            return [], {
                "pool_size": self._finite_pool_table.candidate_count,
                "fallback_triggered": True,
                "constraint_summary": constraint_summary,
                "constraint_signature": self._finite_pool_table.allowed_keys_signature(
                    self._finite_pool_table._key_set  # noqa: SLF001
                ),
            }
        constraint = self._finite_pool_table.membership_constraint(allowed_keys=allowed_keys)
        return [constraint], {
            "pool_size": len(allowed_keys),
            "fallback_triggered": False,
            "constraint_summary": constraint_summary,
            "constraint_signature": self._finite_pool_table.allowed_keys_signature(allowed_keys),
        }

    def filter_candidate_keys_by_constraint_specs(
        self,
        *,
        constraint_specs: list[dict[str, Any]],
        min_pool_fraction: float = 0.05,
    ) -> tuple[set[tuple[str, ...]] | None, bool, str]:
        if self._backend != "finite_pool" or not constraint_specs:
            return self.candidate_keys(), False, ""
        include_map: dict[str, list[str]] = {}
        exclude_map: dict[str, list[str]] = {}
        summaries: list[str] = []
        for spec in constraint_specs:
            var = spec.get("variable", "")
            ctype = spec.get("constraint_type", "include_values")
            values = spec.get("values", [])
            rationale = spec.get("rationale", "")
            if not var or not values:
                continue
            if ctype == "include_values":
                include_map.setdefault(var, [])
                include_map[var] = list(set(include_map[var]) | set(values))
            else:
                exclude_map.setdefault(var, [])
                exclude_map[var] = list(set(exclude_map[var]) | set(values))
            summaries.append(f"{ctype}({var}={values}): {rationale}")
        allowed_keys, fallback_triggered = self._finite_pool_table.filter_keys_by_llm_constraint(
            include_map=include_map or None,
            exclude_map=exclude_map or None,
            min_pool_fraction=min_pool_fraction,
        )
        return allowed_keys, fallback_triggered, "; ".join(summaries)

    def search_space_meta(self) -> dict[str, Any]:
        """Return search space metadata for LLM prompts (finite-pool only)."""
        if self._backend != "finite_pool":
            return {}
        pool = self._finite_pool_table
        valid_values: dict[str, list[str]] = {}
        for col in pool.feature_columns:
            valid_values[col] = list(dict.fromkeys(pool._df[col].astype(str).tolist()))  # noqa: SLF001
        return {
            "feature_columns": list(pool.feature_columns),
            "scaffold_dims": list(pool.spec.scaffold_dims),
            "key_dimensions": list(pool.spec.key_dimensions or pool.spec.scaffold_dims),
            "valid_values_per_col": valid_values,
        }

    def run(self, sample: Any) -> float:
        if self._backend == "finite_pool":
            return self._finite_pool_table.evaluate(sample)
        if isinstance(sample, dict):
            sample = ParameterVector().from_dict(sample, param_space=self.param_space)
        if self._backend == "lookup":
            measurement = self._lookup_dataset.run(sample, return_paramvector=False)
        else:
            measurement, _, _ = self._emulator.run(sample, return_paramvector=False)
        if isinstance(measurement, (list, tuple, np.ndarray)):
            measurement = np.asarray(measurement).reshape(-1)[0]
        return float(measurement)
