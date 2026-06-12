"""Subspace builder and candidate completion utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterVector


def _default_for_param(param) -> Any:  # noqa: ANN001, ANN401
    if param.type == "categorical":
        return param.options[0]
    if param.type == "discrete":
        if len(param.options) > 0:
            return param.options[0]
        return param.low
    if param.type == "continuous":
        return float((param.low + param.high) / 2.0)
    return None


def extract_default_values(param_space) -> dict[str, Any]:  # noqa: ANN001
    return {param.name: _default_for_param(param) for param in param_space}


def build_subspace(original_space, active_variables: list[str]):  # noqa: ANN001, ANN201
    subspace = ParameterSpace()
    active_set = set(active_variables)
    for param in original_space:
        if param.name in active_set:
            subspace.add(param)
    return subspace


def complete_candidate(
    partial_candidate: dict[str, Any],
    completion_overrides: dict[str, Any],
    fallback_defaults: dict[str, Any],
    param_space=None,  # noqa: ANN001
) -> dict[str, Any]:
    # Priority: BO partial > LLM completion override > fallback defaults.
    merged = dict(fallback_defaults)
    merged.update(completion_overrides)
    merged.update(partial_candidate)
    if param_space is None:
        return merged
    return {param.name: merged[param.name] for param in param_space}


def build_subspace_campaign(
    full_campaign: Campaign,
    original_space,
    active_variables: list[str],
) -> Campaign:  # noqa: ANN001
    subspace = build_subspace(original_space, active_variables)
    sub_campaign = Campaign()
    sub_campaign.set_param_space(subspace)
    sub_campaign.set_value_space(full_campaign.value_space)

    all_params = full_campaign.observations.get_params(as_array=True)
    all_values = full_campaign.observations.get_values(as_array=True)
    if len(all_values) == 0:
        return sub_campaign

    all_names = [param.name for param in original_space]
    active_idx = [all_names.index(name) for name in active_variables]

    for row, value in zip(all_params, all_values):
        sub_dict = {
            name: row[idx] for name, idx in zip(active_variables, active_idx)
        }
        param_vec = ParameterVector().from_dict(sub_dict, param_space=subspace)
        if isinstance(value, np.ndarray):
            cast_value = float(value.reshape(-1)[0])
        else:
            cast_value = float(value)
        sub_campaign.add_observation(param_vec, cast_value)
    return sub_campaign

