"""Reusable runtime helpers for v0.6 action-package execution."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetContrastAdapter:
    dataset: str
    reaction_type: str
    scaffold_dims: tuple[str, ...]
    primary_dims: tuple[str, ...]
    secondary_dims: tuple[str, ...]
    condition_dims: tuple[str, ...]
    dimension_groups: dict[str, str]


def candidate_key(candidate: dict[str, Any], feature_columns: list[str]) -> tuple[str, ...]:
    return tuple(str(candidate.get(name)) for name in feature_columns)


def scaffold_key(candidate: dict[str, Any], scaffold_dims: list[str]) -> tuple[str, ...]:
    return tuple(str(candidate.get(name)) for name in scaffold_dims)


def build_dataset_contrast_adapter(
    *,
    dataset: str,
    reaction_type: str,
    feature_columns: list[str],
    scaffold_dims: list[str],
) -> DatasetContrastAdapter:
    group_map: dict[str, str] = {name: "condition" for name in feature_columns}
    primary_dims = tuple(scaffold_dims[:1])
    secondary_dims = tuple(scaffold_dims[1:2])
    if dataset == "arylation":
        group_map.update(
            {
                "Aryl_halide_SMILES": "substrate",
                "Additive_SMILES": "additive",
                "Base_SMILES": "base",
                "Ligand_SMILES": "ligand",
            }
        )
        primary_dims = ("Aryl_halide_SMILES",)
        secondary_dims = ("Additive_SMILES",)
    elif dataset.startswith("buchwald"):
        group_map.update(
            {
                "Reactant2": "substrate",
                "Ligand": "ligand",
                "Base": "base",
                "Additive": "additive",
            }
        )
        primary_dims = ("Reactant2",)
        secondary_dims = ("Ligand",)
    elif dataset == "suzuki_hte_full":
        group_map.update(
            {
                "electrophile": "electrophile",
                "nucleophile": "nucleophile",
                "catalyst": "catalyst",
                "ligand": "ligand",
                "base": "base",
                "solvent": "solvent",
            }
        )
        primary_dims = ("electrophile",)
        secondary_dims = ("nucleophile",)
    else:
        for name in scaffold_dims:
            group_map[name] = "scaffold"
    for name in scaffold_dims:
        group_map.setdefault(name, "scaffold")
    condition_dims = tuple(name for name in feature_columns if name not in scaffold_dims)
    return DatasetContrastAdapter(
        dataset=dataset,
        reaction_type=reaction_type,
        scaffold_dims=tuple(scaffold_dims),
        primary_dims=tuple(name for name in primary_dims if name in feature_columns),
        secondary_dims=tuple(name for name in secondary_dims if name in feature_columns),
        condition_dims=condition_dims,
        dimension_groups={name: group_map.get(name, "condition") for name in feature_columns},
    )


def history_value_stats(
    *,
    history: list[dict[str, Any]],
    feature_columns: list[str],
    goal: str,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    grouped: dict[str, dict[str, list[float]]] = {name: {} for name in feature_columns}
    for row in history:
        candidate = row.get("candidate")
        result = _float_or_none(row.get("result"))
        if not isinstance(candidate, dict) or result is None:
            continue
        for name in feature_columns:
            value = str(candidate.get(name))
            grouped.setdefault(name, {}).setdefault(value, []).append(result)
    stats: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for name, values_by_key in grouped.items():
        stats[name] = {}
        for key, values in values_by_key.items():
            if not values:
                continue
            stats[name][key] = {
                "count": len(values),
                "best": min(values) if goal == "minimize" else max(values),
                "mean": sum(values) / len(values),
            }
    return stats


def nearest_analogues(
    *,
    candidate: dict[str, Any],
    history: list[dict[str, Any]],
    feature_columns: list[str],
    scaffold_dims: list[str],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    scored: list[tuple[int, int, dict[str, Any]]] = []
    cand_key = candidate_key(candidate, feature_columns)
    cand_scaffold = scaffold_key(candidate, scaffold_dims)
    for row in history:
        observed = row.get("candidate")
        if not isinstance(observed, dict):
            continue
        obs_key = candidate_key(observed, feature_columns)
        mismatches = sum(1 for left, right in zip(cand_key, obs_key, strict=False) if left != right)
        obs_scaffold = scaffold_key(observed, scaffold_dims)
        scaffold_mismatches = sum(
            1 for left, right in zip(cand_scaffold, obs_scaffold, strict=False) if left != right
        )
        scored.append(
            (
                mismatches,
                scaffold_mismatches,
                {"candidate": observed, "result": row.get("result")},
            )
        )
    scored = sorted(scored, key=lambda item: (item[0], item[1]))
    analogues: list[dict[str, Any]] = []
    for mismatches, scaffold_mismatches, row in scored[: max(0, int(top_k))]:
        observed = row["candidate"]
        changed = {
            name: {"candidate": candidate.get(name), "analogue": observed.get(name)}
            for name in feature_columns
            if str(candidate.get(name)) != str(observed.get(name))
        }
        shared = [name for name in feature_columns if str(candidate.get(name)) == str(observed.get(name))]
        analogues.append(
            {
                "mismatch_count": mismatches,
                "scaffold_mismatch_count": scaffold_mismatches,
                "observed_candidate": observed,
                "observed_result": row.get("result"),
                "changed_variables": changed,
                "shared_variables": shared,
            }
        )
    return analogues


def build_candidate_contrastive_evidence(
    *,
    shortlist_candidates: list[dict[str, Any]],
    history: list[dict[str, Any]],
    feature_columns: list[str],
    goal: str,
    adapter: DatasetContrastAdapter,
) -> dict[str, Any]:
    if not shortlist_candidates:
        return {}
    bo_item = _main_bo_item(shortlist_candidates)
    bo_candidate = dict(bo_item.get("candidate") or {})
    stats = history_value_stats(history=history, feature_columns=feature_columns, goal=goal)
    by_candidate: dict[str, Any] = {}
    for item in shortlist_candidates:
        idx = int(item.get("candidate_index", 0) or 0)
        by_candidate[str(idx)] = _contrastive_evidence(
            candidate_item=item,
            bo_item=bo_item,
            history=history,
            feature_columns=feature_columns,
            goal=goal,
            adapter=adapter,
            history_value_stats_map=stats,
        )
    return {
        "bo_top1_candidate_index": int(bo_item.get("candidate_index", 0) or 0),
        "bo_top1_candidate": bo_candidate,
        "feature_columns": list(feature_columns),
        "scaffold_dims": list(adapter.scaffold_dims),
        "by_candidate_index": by_candidate,
    }


def dominant_mask_keys(
    *,
    allowed_keys: set[tuple[str, ...]],
    feature_columns: list[str],
    scaffold_dims: list[str],
    dominant_scaffold: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> tuple[set[tuple[str, ...]], dict[str, Any]]:
    dominant_key = tuple(str(dominant_scaffold.get(name)) for name in scaffold_dims) if isinstance(dominant_scaffold, dict) else ()
    if not dominant_key or not any(value for value in dominant_key):
        dominant_key = _dominant_scaffold_from_history(
            history=history,
            scaffold_dims=scaffold_dims,
        )
    if not dominant_key:
        return set(allowed_keys), {
            "mask_action": "mask_dominant_resuggest",
            "mask_applied": False,
            "fallback_reason": "no_dominant_scaffold_identified",
            "pool_size_before": len(allowed_keys),
            "pool_size_after": len(allowed_keys),
            "excluded_candidate_count": 0,
            "mask_basis": {},
        }
    idx_map = {name: i for i, name in enumerate(feature_columns)}
    filtered = {
        key
        for key in allowed_keys
        if tuple(key[idx_map[name]] for name in scaffold_dims) != dominant_key
    }
    if not filtered:
        return set(allowed_keys), {
            "mask_action": "mask_dominant_resuggest",
            "mask_applied": False,
            "fallback_reason": "mask_would_empty_pool",
            "pool_size_before": len(allowed_keys),
            "pool_size_after": len(allowed_keys),
            "excluded_candidate_count": 0,
            "mask_basis": {name: value for name, value in zip(scaffold_dims, dominant_key, strict=False)},
        }
    return filtered, {
        "mask_action": "mask_dominant_resuggest",
        "mask_applied": len(filtered) < len(allowed_keys),
        "fallback_reason": None,
        "pool_size_before": len(allowed_keys),
        "pool_size_after": len(filtered),
        "excluded_candidate_count": len(allowed_keys) - len(filtered),
        "mask_basis": {name: value for name, value in zip(scaffold_dims, dominant_key, strict=False)},
    }


def scaffold_corridor_mask_keys(
    *,
    allowed_keys: set[tuple[str, ...]],
    feature_columns: list[str],
    scaffold_dims: list[str],
    dominant_values_by_dim: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> tuple[set[tuple[str, ...]], dict[str, Any]]:
    if not scaffold_dims:
        return set(allowed_keys), {
            "mask_action": "mask_scaffold_corridor_resuggest",
            "mask_applied": False,
            "fallback_reason": "no_scaffold_dimensions",
            "pool_size_before": len(allowed_keys),
            "pool_size_after": len(allowed_keys),
            "excluded_candidate_count": 0,
            "mask_basis": {},
        }
    dominant_map = dict(dominant_values_by_dim or {})
    ranked_dims: list[tuple[str, str, float]] = []
    for name in scaffold_dims:
        payload = dominant_map.get(name)
        if isinstance(payload, dict):
            value = str(payload.get("value", "") or "")
            ratio = float(payload.get("ratio", 0.0) or 0.0)
        else:
            value = ""
            ratio = 0.0
        if value:
            ranked_dims.append((name, value, ratio))
    if not ranked_dims:
        dominant_scaffold = {
            name: value
            for name, value in zip(
                scaffold_dims,
                _dominant_scaffold_from_history(
                    history=history,
                    scaffold_dims=scaffold_dims,
                ),
                strict=False,
            )
            if value
        }
        for name in scaffold_dims:
            value = str(dominant_scaffold.get(name, "") or "")
            if value:
                ranked_dims.append((name, value, 0.0))
    if not ranked_dims:
        return set(allowed_keys), {
            "mask_action": "mask_scaffold_corridor_resuggest",
            "mask_applied": False,
            "fallback_reason": "no_dominant_scaffold_corridor_identified",
            "pool_size_before": len(allowed_keys),
            "pool_size_after": len(allowed_keys),
            "excluded_candidate_count": 0,
            "mask_basis": {},
        }
    ranked_dims.sort(key=lambda item: item[2], reverse=True)
    primary_dim, primary_value, primary_ratio = ranked_dims[0]
    idx_map = {name: i for i, name in enumerate(feature_columns)}
    if primary_dim not in idx_map:
        return set(allowed_keys), {
            "mask_action": "mask_scaffold_corridor_resuggest",
            "mask_applied": False,
            "fallback_reason": "primary_corridor_dimension_not_in_feature_columns",
            "pool_size_before": len(allowed_keys),
            "pool_size_after": len(allowed_keys),
            "excluded_candidate_count": 0,
            "mask_basis": {},
        }
    filtered = {
        key for key in allowed_keys if key[idx_map[primary_dim]] != primary_value
    }
    if not filtered:
        return set(allowed_keys), {
            "mask_action": "mask_scaffold_corridor_resuggest",
            "mask_applied": False,
            "fallback_reason": "corridor_mask_would_empty_pool",
            "pool_size_before": len(allowed_keys),
            "pool_size_after": len(allowed_keys),
            "excluded_candidate_count": 0,
            "mask_basis": {
                "corridor_dims": [primary_dim],
                "dominant_values": {primary_dim: primary_value},
                "dominant_ratios": {primary_dim: round(primary_ratio, 6)},
                "corridor_mode": "primary_scaffold_value_slice",
            },
        }
    return filtered, {
        "mask_action": "mask_scaffold_corridor_resuggest",
        "mask_applied": len(filtered) < len(allowed_keys),
        "fallback_reason": None,
        "pool_size_before": len(allowed_keys),
        "pool_size_after": len(filtered),
        "excluded_candidate_count": len(allowed_keys) - len(filtered),
        "mask_basis": {
            "corridor_dims": [primary_dim],
            "dominant_values": {primary_dim: primary_value},
            "dominant_ratios": {primary_dim: round(primary_ratio, 6)},
            "corridor_mode": "primary_scaffold_value_slice",
        },
    }


def low_repeat_mask_keys(
    *,
    allowed_keys: set[tuple[str, ...]],
    feature_columns: list[str],
    anchor_dims: list[str],
    anchor_values: dict[str, Any],
    dominant_scaffold: dict[str, Any] | None,
    scaffold_dims: list[str],
    history: list[dict[str, Any]],
) -> tuple[set[tuple[str, ...]], dict[str, Any]]:
    active_anchor_dims = [name for name in anchor_dims if name in feature_columns and name in anchor_values]
    idx_map = {name: i for i, name in enumerate(feature_columns)}
    filtered = set(allowed_keys)
    mask_basis: dict[str, Any] = {}
    fallback_reason = None
    if active_anchor_dims:
        filtered = {
            key
            for key in allowed_keys
            if any(key[idx_map[name]] != str(anchor_values.get(name)) for name in active_anchor_dims)
        }
        mask_basis = {name: anchor_values.get(name) for name in active_anchor_dims}
    if not active_anchor_dims or not filtered:
        dominant_filtered, dominant_summary = dominant_mask_keys(
            allowed_keys=allowed_keys,
            feature_columns=feature_columns,
            scaffold_dims=scaffold_dims,
            dominant_scaffold=dominant_scaffold,
            history=history,
        )
        if dominant_summary.get("mask_applied"):
            return dominant_filtered, {
                "mask_action": "mask_low_repeat_resuggest",
                "mask_applied": True,
                "fallback_reason": None,
                "pool_size_before": len(allowed_keys),
                "pool_size_after": len(dominant_filtered),
                "excluded_candidate_count": len(allowed_keys) - len(dominant_filtered),
                "mask_basis": {
                    "fallback_to_dominant_scaffold": True,
                    **(dominant_summary.get("mask_basis") or {}),
                },
            }
        fallback_reason = "no_anchor_repeat_region_identified"
        filtered = set(allowed_keys)
    return filtered, {
        "mask_action": "mask_low_repeat_resuggest",
        "mask_applied": len(filtered) < len(allowed_keys),
        "fallback_reason": fallback_reason,
        "pool_size_before": len(allowed_keys),
        "pool_size_after": len(filtered),
        "excluded_candidate_count": len(allowed_keys) - len(filtered),
        "mask_basis": mask_basis,
    }


def shortlist_value_audit(
    *,
    shortlist_candidates: list[dict[str, Any]],
    candidate_values: dict[int, float],
    bo_top1_index: int,
    llm_selected_index: int | None,
    final_selected_index: int | None,
    goal: str,
) -> dict[str, Any]:
    if not shortlist_candidates or bo_top1_index not in candidate_values:
        return {}
    by_index = {int(item.get("candidate_index", idx)): item for idx, item in enumerate(shortlist_candidates)}
    ordered = [(idx, value) for idx, value in candidate_values.items() if idx in by_index]
    if not ordered:
        return {}
    best_idx, best_value = (
        min(ordered, key=lambda item: item[1]) if goal == "minimize" else max(ordered, key=lambda item: item[1])
    )
    non_top = [(idx, value) for idx, value in ordered if idx != bo_top1_index]
    best_non_top_idx = None
    best_non_top_value = None
    if non_top:
        best_non_top_idx, best_non_top_value = (
            min(non_top, key=lambda item: item[1]) if goal == "minimize" else max(non_top, key=lambda item: item[1])
        )
    bo_value = candidate_values.get(bo_top1_index)
    llm_value = candidate_values.get(llm_selected_index) if llm_selected_index is not None else None
    final_value = candidate_values.get(final_selected_index) if final_selected_index is not None else None
    return {
        "shortlist_true_values": {str(idx): value for idx, value in candidate_values.items()},
        "llm_selected_index": llm_selected_index,
        "llm_selected_true_value": llm_value,
        "final_selected_index": final_selected_index,
        "selected_true_value": final_value,
        "bo_top1_true_value": bo_value,
        "best_shortlist_index": best_idx,
        "best_shortlist_true_value": best_value,
        "best_non_top1_index": best_non_top_idx,
        "best_non_top1_true_value": best_non_top_value,
        "shortlist_oracle_headroom": _delta_vs_bo(best_value, bo_value, goal=goal),
        "best_non_top1_headroom": _delta_vs_bo(best_non_top_value, bo_value, goal=goal),
        "llm_headroom": _delta_vs_bo(llm_value, bo_value, goal=goal),
        "final_headroom": _delta_vs_bo(final_value, bo_value, goal=goal),
    }


def _main_bo_item(shortlist_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    tagged = [item for item in shortlist_candidates if bool(item.get("is_main_bo_top1"))]
    if tagged:
        return dict(tagged[0])
    main_pool_sorted = sorted(
        (
            item
            for item in shortlist_candidates
            if item.get("main_pool_rank") is not None
        ),
        key=lambda item: int(item.get("main_pool_rank", 999) or 999),
    )
    if main_pool_sorted:
        return dict(main_pool_sorted[0])
    return dict(shortlist_candidates[0])


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta_vs_bo(value: float | None, bo_value: float | None, *, goal: str) -> float | None:
    if value is None or bo_value is None:
        return None
    return bo_value - value if goal == "minimize" else value - bo_value


def _history_match_stats(
    *,
    history: list[dict[str, Any]],
    candidate: dict[str, Any],
    dims: list[str],
    goal: str,
) -> dict[str, float | int | None]:
    observed: list[float] = []
    for row in history:
        current = row.get("candidate")
        value = _float_or_none(row.get("result"))
        if not isinstance(current, dict) or value is None:
            continue
        if all(str(current.get(name)) == str(candidate.get(name)) for name in dims):
            observed.append(value)
    if not observed:
        return {"match_count": 0, "best": None, "mean": None}
    return {
        "match_count": len(observed),
        "best": min(observed) if goal == "minimize" else max(observed),
        "mean": sum(observed) / len(observed),
    }


def _analogue_summary(analogues: list[dict[str, Any]], *, goal: str) -> dict[str, float | int | None]:
    observed = [(_float_or_none(item.get("observed_result")), int(item.get("mismatch_count", 0) or 0)) for item in analogues]
    observed = [(value, mismatch) for value, mismatch in observed if value is not None]
    if not observed:
        return {"count": 0, "best_result": None, "mean_result": None, "closest_mismatch_count": None}
    values = [value for value, _mismatch in observed]
    return {
        "count": len(observed),
        "best_result": min(values) if goal == "minimize" else max(values),
        "mean_result": sum(values) / len(values),
        "closest_mismatch_count": min(mismatch for _value, mismatch in observed),
    }


def _contrastive_evidence(
    *,
    candidate_item: dict[str, Any],
    bo_item: dict[str, Any],
    history: list[dict[str, Any]],
    feature_columns: list[str],
    goal: str,
    adapter: DatasetContrastAdapter,
    history_value_stats_map: dict[str, dict[str, dict[str, float | int | None]]],
) -> dict[str, Any]:
    candidate = dict(candidate_item.get("candidate") or {})
    bo_candidate = dict(bo_item.get("candidate") or {})
    changed_dims = [name for name in feature_columns if str(candidate.get(name)) != str(bo_candidate.get(name))]
    shared_dims = [name for name in feature_columns if name not in changed_dims]
    analogue_candidates = nearest_analogues(
        candidate=candidate,
        history=history,
        feature_columns=feature_columns,
        scaffold_dims=list(adapter.scaffold_dims),
        top_k=3,
    )
    analogue_bo = nearest_analogues(
        candidate=bo_candidate,
        history=history,
        feature_columns=feature_columns,
        scaffold_dims=list(adapter.scaffold_dims),
        top_k=3,
    )
    candidate_analogue = _analogue_summary(analogue_candidates, goal=goal)
    bo_analogue = _analogue_summary(analogue_bo, goal=goal)
    same_scaffold_candidate = _history_match_stats(
        history=history,
        candidate=candidate,
        dims=list(adapter.scaffold_dims),
        goal=goal,
    )
    same_scaffold_bo = _history_match_stats(
        history=history,
        candidate=bo_candidate,
        dims=list(adapter.scaffold_dims),
        goal=goal,
    )
    changed_dimension_rows: list[dict[str, Any]] = []
    candidate_better_dim_count = 0
    bo_better_dim_count = 0
    for name in changed_dims:
        dim_stats = history_value_stats_map.get(name, {})
        candidate_stats = dim_stats.get(str(candidate.get(name)), {"count": 0, "best": None, "mean": None})
        bo_stats = dim_stats.get(str(bo_candidate.get(name)), {"count": 0, "best": None, "mean": None})
        best_delta = _delta_vs_bo(
            _float_or_none(candidate_stats.get("best")),
            _float_or_none(bo_stats.get("best")),
            goal=goal,
        )
        if best_delta is not None:
            if best_delta > 0:
                candidate_better_dim_count += 1
            elif best_delta < 0:
                bo_better_dim_count += 1
        changed_dimension_rows.append(
            {
                "dimension": name,
                "group": adapter.dimension_groups.get(name, "condition"),
                "candidate_value": candidate.get(name),
                "bo_top1_value": bo_candidate.get(name),
                "candidate_history_count": int(candidate_stats.get("count", 0) or 0),
                "candidate_history_best": _float_or_none(candidate_stats.get("best")),
                "candidate_history_mean": _float_or_none(candidate_stats.get("mean")),
                "bo_top1_history_count": int(bo_stats.get("count", 0) or 0),
                "bo_top1_history_best": _float_or_none(bo_stats.get("best")),
                "bo_top1_history_mean": _float_or_none(bo_stats.get("mean")),
                "candidate_minus_bo_best": best_delta,
            }
        )
    anchor_dims = shared_dims or list(adapter.scaffold_dims)
    same_anchor_candidate = _history_match_stats(
        history=history,
        candidate=candidate,
        dims=anchor_dims,
        goal=goal,
    )
    same_anchor_bo = _history_match_stats(
        history=history,
        candidate=bo_candidate,
        dims=anchor_dims,
        goal=goal,
    )
    return {
        "same_as_bo_top1": int(candidate_item.get("candidate_index", -1)) == int(bo_item.get("candidate_index", -2)),
        "changed_dimension_count": len(changed_dims),
        "shared_dimension_count": len(shared_dims),
        "changed_dimensions": changed_dimension_rows,
        "changed_scaffold_dims": [name for name in changed_dims if name in set(adapter.scaffold_dims)],
        "changed_condition_dims": [name for name in changed_dims if name not in set(adapter.scaffold_dims)],
        "candidate_better_dimension_count": candidate_better_dim_count,
        "bo_top1_better_dimension_count": bo_better_dim_count,
        "same_scaffold_support": {
            "candidate": same_scaffold_candidate,
            "bo_top1": same_scaffold_bo,
            "candidate_minus_bo_best": _delta_vs_bo(
                _float_or_none(same_scaffold_candidate.get("best")),
                _float_or_none(same_scaffold_bo.get("best")),
                goal=goal,
            ),
        },
        "same_anchor_support": {
            "anchor_dims": anchor_dims,
            "candidate": same_anchor_candidate,
            "bo_top1": same_anchor_bo,
            "candidate_minus_bo_best": _delta_vs_bo(
                _float_or_none(same_anchor_candidate.get("best")),
                _float_or_none(same_anchor_bo.get("best")),
                goal=goal,
            ),
        },
        "analogue_support": {
            "candidate": candidate_analogue,
            "bo_top1": bo_analogue,
            "candidate_minus_bo_best": _delta_vs_bo(
                _float_or_none(candidate_analogue.get("best_result")),
                _float_or_none(bo_analogue.get("best_result")),
                goal=goal,
            ),
        },
    }


def _dominant_scaffold_from_history(
    *,
    history: list[dict[str, Any]],
    scaffold_dims: list[str],
) -> tuple[str, ...]:
    if not history or not scaffold_dims:
        return tuple()
    counts: Counter[tuple[str, ...]] = Counter()
    for row in history:
        candidate = row.get("candidate")
        if isinstance(candidate, dict):
            counts[scaffold_key(candidate, scaffold_dims)] += 1
    return counts.most_common(1)[0][0] if counts else tuple()
