"""Planner registry shared by baseline and agentic runners."""

from __future__ import annotations

from typing import Any

from chem_agent_bo.bo.base import BasePlanner
from chem_agent_bo.bo.botorch_bo import BoTorchFinitePoolPlanner
from chem_agent_bo.bo.chunked_gp import ChunkedGPPlanner
from chem_agent_bo.bo.discrete_bo import DiscreteBOPlanner
from chem_agent_bo.bo.random_bo import RandomPlanner


PLANNER_NAMES = (
    "atlas",
    "chunked_gp",
    "random",
    "discrete",
    "botorch",
    "botorch_qei",
    "botorch_qlogei",
)


def planner_choices() -> tuple[str, ...]:
    return PLANNER_NAMES


def build_planner(
    planner_name: str,
    *,
    env,  # noqa: ANN001
    init_budget: int,
    seed: int,
    known_constraints: list[Any] | None = None,
    use_descriptors: bool = False,
    acquisition_type: str = "ei",
) -> BasePlanner:
    name = str(planner_name).strip().lower()
    if name == "random":
        return RandomPlanner(seed=seed, goal=env.goal)
    if name == "atlas":
        from chem_agent_bo.bo.atlas_bo import AtlasBOTool

        return AtlasBOTool(
            num_init_design=init_budget,
            seed=seed,
            goal=env.goal,
            known_constraints=known_constraints,
            use_descriptors=use_descriptors,
            acquisition_type=acquisition_type,
        )
    if name == "chunked_gp":
        return ChunkedGPPlanner(
            seed=seed,
            goal=env.goal,
            known_constraints=known_constraints,
            use_descriptors=use_descriptors,
            acquisition_type=acquisition_type,
        )
    if name == "discrete":
        if not getattr(env, "is_finite_pool", False):
            raise ValueError("Discrete-BO currently supports finite-pool datasets only.")
        return DiscreteBOPlanner(
            seed=seed,
            goal=env.goal,
            num_init_design=init_budget,
            known_constraints=known_constraints,
        )
    if name in {"botorch", "botorch_qei", "botorch_qlogei"}:
        if not getattr(env, "is_finite_pool", False):
            raise ValueError("BoTorch-BO currently supports finite-pool datasets only.")
        return BoTorchFinitePoolPlanner(
            seed=seed,
            goal=env.goal,
            num_init_design=init_budget,
            known_constraints=known_constraints,
            planner_name="botorch_qei" if name == "botorch" else name,
            acquisition_mode="qei" if name in {"botorch", "botorch_qei"} else "qlogei",
        )
    raise ValueError(f"Unsupported planner_name `{planner_name}`. Available: {', '.join(PLANNER_NAMES)}")
