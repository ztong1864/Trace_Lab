"""Planner registry shared by baseline and agentic runners."""

from __future__ import annotations

from typing import Any

from chem_agent_bo.bo.base import BasePlanner
from chem_agent_bo.bo.botorch_bo import BoTorchFinitePoolPlanner
from chem_agent_bo.bo.chunked_gp import CHUNKED_GP_ACQUISITION_TYPES, ChunkedGPPlanner
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

# Settings a project can give a planner through `planner_options` in project.yaml,
# keyed by planner name. Each value must be a positive integer.
PLANNER_OPTION_NAMES: dict[str, tuple[str, ...]] = {
    "chunked_gp": ("min_changed_variables", "max_scan_size", "finalist_count"),
}


def planner_choices() -> tuple[str, ...]:
    return PLANNER_NAMES


def supported_acquisitions(planner_name: str) -> tuple[str, ...] | None:
    """Acquisition functions a planner accepts, or None if it doesn't use one."""
    name = str(planner_name).strip().lower()
    if name == "atlas":
        from chem_agent_bo.bo.atlas_bo import ATLAS_ACQUISITION_TYPES

        return ATLAS_ACQUISITION_TYPES
    if name == "chunked_gp":
        return CHUNKED_GP_ACQUISITION_TYPES
    return None


def validate_planner_options(planner_options: Any) -> dict[str, dict[str, int]]:
    """Check a project's `planner_options` ({planner: {option: value}}); return a clean copy."""
    if planner_options in (None, ""):
        return {}
    if not isinstance(planner_options, dict):
        raise ValueError("planner_options must map a planner name to its options.")
    cleaned: dict[str, dict[str, int]] = {}
    for planner, options in planner_options.items():
        name = str(planner).strip().lower()
        allowed = PLANNER_OPTION_NAMES.get(name)
        if allowed is None:
            raise ValueError(
                f"Planner `{name}` has no configurable options; "
                f"planners with options: {sorted(PLANNER_OPTION_NAMES)}."
            )
        if not isinstance(options, dict):
            raise ValueError(f"planner_options.{name} must be a mapping of option to value.")
        cleaned[name] = {}
        for key, value in options.items():
            if key not in allowed:
                raise ValueError(f"Unknown option `{key}` for planner `{name}`; allowed: {list(allowed)}.")
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"planner_options.{name}.{key} must be a positive integer; got {value!r}.")
            cleaned[name][key] = value
    return cleaned


def build_planner(
    planner_name: str,
    *,
    env,  # noqa: ANN001
    init_budget: int,
    seed: int,
    known_constraints: list[Any] | None = None,
    use_descriptors: bool = False,
    acquisition_type: str = "ei",
    planner_options: dict[str, int] | None = None,
) -> BasePlanner:
    """Build a planner. `planner_options` are that planner's own settings (see
    PLANNER_OPTION_NAMES); planners without configurable options reject any."""
    name = str(planner_name).strip().lower()
    options = dict(planner_options or {})
    if options:
        validate_planner_options({name: options})
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
            **options,
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
