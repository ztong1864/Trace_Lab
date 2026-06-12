"""Shared experiment helpers for baseline and agentic runners."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any
import sys

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore[no-redef]

import numpy as np

from chem_agent_bo.protocol import sample_key


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_budget_args(
    *,
    budget: int,
    total_budget: int | None,
    num_init_design: int,
    init_budget: int | None,
) -> tuple[int, int]:
    total = int(total_budget if total_budget is not None else budget)
    init = int(init_budget if init_budget is not None else num_init_design)
    init = max(0, min(init, total))
    return total, init


def build_run_output_dir(
    root_dir: str | Path,
    *,
    dataset: str,
    method_family: str,
    planner_name: str,
    seed: int,
    total_budget: int,
    init_budget: int,
) -> Path:
    root = Path(root_dir)
    run_dir = root / (
        f"{dataset}__{method_family}__{planner_name}"
        f"__seed{seed}__tb{total_budget}__ib{init_budget}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_run_stem(
    *,
    dataset: str,
    seed: int,
    total_budget: int,
) -> str:
    return f"{dataset}_seed{seed}_budget{total_budget}"


def initial_candidate_keys(
    *,
    initial_candidates: list[dict[str, Any]],
    param_space,  # noqa: ANN001
) -> list[list[str]]:
    names = [param.name for param in param_space]
    return [list(sample_key(candidate, names)) for candidate in initial_candidates]


def runtime_metadata() -> dict[str, Any]:
    executable = sys.executable
    env_name = os.path.basename(os.path.dirname(os.path.dirname(executable))) or "unknown"
    try:
        botorch_version = importlib_metadata.version("botorch")
    except Exception:  # noqa: BLE001
        botorch_version = None
    return {
        "runtime_env_name": env_name,
        "runtime_python": executable,
        "runtime_botorch_version": botorch_version,
    }
