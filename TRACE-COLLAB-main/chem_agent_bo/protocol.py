"""Experiment protocol helpers and shared initialization."""

from __future__ import annotations

import random
from typing import Any


PROTOCOL_VERSION = "v1.0"


def protocol_budget_metadata(total_budget: int, init_budget: int) -> dict[str, int]:
    total = max(0, int(total_budget))
    init = max(0, min(int(init_budget), total))
    return {
        "total_budget": total,
        "init_budget": init,
        "bo_budget": max(0, total - init),
    }


def sample_key(candidate: dict[str, Any], param_names: list[str]) -> tuple[str, ...]:
    return tuple(str(candidate.get(name)) for name in param_names)


def shared_initial_candidates(
    *,
    env,  # noqa: ANN001
    seed: int,
    init_budget: int,
) -> list[dict[str, Any]]:
    """Build deterministic shared init candidates for finite-pool runs.

    For non-finite-pool environments, return an empty list and let the planner
    handle initialization internally.
    """
    if not getattr(env, "is_finite_pool", False):
        return []
    budget = max(0, int(init_budget))
    if budget == 0:
        return []
    pool = getattr(env, "_finite_pool_table", None)
    if pool is None:
        return []
    param_names = [param.name for param in env.param_space]
    keys = list(pool.record_keys())
    rng = random.Random(int(seed))
    rng.shuffle(keys)
    selected = keys[: min(len(keys), budget)]
    return [
        {name: value for name, value in zip(param_names, key, strict=False)}
        for key in selected
    ]
