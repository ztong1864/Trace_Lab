"""Execution adapters for TRACE runtimes."""

from __future__ import annotations

from typing import Any

from olympus.objects import ParameterVector


class BenchmarkExecutionAdapter:
    """Synchronous benchmark adapter around ReactionEnv and Campaign.

    This is the only production path in the benchmark runner that should call
    `ReactionEnv.run()`. The controller runtime remains outcome-agnostic.
    """

    def __init__(self, *, env, campaign) -> None:  # noqa: ANN001
        self.env = env
        self.campaign = campaign

    @property
    def is_finite_pool(self) -> bool:
        return bool(getattr(self.env, "is_finite_pool", False))

    def validate_candidate(self, candidate: dict[str, Any]) -> None:
        if self.is_finite_pool and not self.env.is_valid_candidate(candidate):
            raise ValueError(f"Invalid finite-pool candidate proposed by BO: {candidate}")

    def evaluate(self, candidate: dict[str, Any]) -> float:
        return float(self.env.run(candidate))

    def observe(self, candidate: dict[str, Any], result: float) -> None:
        full_vector = ParameterVector().from_dict(candidate, param_space=self.env.param_space)
        self.campaign.add_observation(full_vector, float(result))

    def evaluate_and_observe(self, candidate: dict[str, Any]) -> float:
        self.validate_candidate(candidate)
        result = self.evaluate(candidate)
        self.observe(candidate, result)
        return result

    def candidate_values(self, shortlist_candidates: list[dict[str, Any]]) -> dict[int, float]:
        if not self.is_finite_pool:
            return {}
        values: dict[int, float] = {}
        for idx, item in enumerate(shortlist_candidates):
            candidate = item.get("candidate")
            if not isinstance(candidate, dict):
                continue
            values[int(item.get("candidate_index", idx))] = self.evaluate(candidate)
        return values
