"""Run-scoped online decision state for current optimization context."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from chem_agent_bo.config.schema import PromptConfig


def _compact_text(value: Any, max_chars: int = 200) -> str:
    text = str(value) if value is not None else ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _best_progress(history: list[dict[str, Any]], goal: str = "maximize") -> list[float]:
    best = float("inf") if goal == "minimize" else float("-inf")
    curve: list[float] = []
    for row in history:
        missing_default = float("inf") if goal == "minimize" else float("-inf")
        result = float(row.get("result", missing_default))
        if goal == "minimize":
            best = min(best, result)
        else:
            best = max(best, result)
        curve.append(best)
    return curve


def _recent_duplicate_ratio(history: list[dict[str, Any]], window: int = 6) -> float:
    recent = history[-window:]
    if len(recent) <= 1:
        return 0.0
    seen: set[str] = set()
    dup = 0
    for row in recent:
        key = str(row.get("candidate", {}))
        if key in seen:
            dup += 1
        seen.add(key)
    return dup / len(recent)


def _recent_feasibility_counts(
    history: list[dict[str, Any]],
    window: int = 6,
) -> dict[str, int]:
    counts: dict[str, int] = {"accept": 0, "revise": 0, "reject": 0}
    for row in history[-window:]:
        action = str(row.get("feasibility_action", "accept"))
        if action in counts:
            counts[action] += 1
    return counts


def _stage_streak(history: list[dict[str, Any]]) -> int:
    if not history:
        return 0
    last_stage = str(history[-1].get("stage", ""))
    streak = 0
    for row in reversed(history):
        if str(row.get("stage", "")) != last_stage:
            break
        streak += 1
    return streak


def _coverage_snapshot(
    history: list[dict[str, Any]],
    search_space=None,  # noqa: ANN001
    search_space_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if search_space is None or len(history) == 0:
        return {
            "overall_ratio": 0.0,
            "dimension_ratios": {},
            "dimension_metric_types": {},
            "underexplored": [],
            "key_dimensions": [],
            "key_dimension_ratios": {},
            "min_key_dim_ratio": 0.0,
            "mean_key_dim_ratio": 0.0,
            "weighted_key_dim_ratio": 0.0,
        }

    candidates = [row.get("candidate", {}) for row in history if isinstance(row.get("candidate"), dict)]
    if len(candidates) == 0:
        return {
            "overall_ratio": 0.0,
            "dimension_ratios": {},
            "dimension_metric_types": {},
            "underexplored": [],
            "key_dimensions": [],
            "key_dimension_ratios": {},
            "min_key_dim_ratio": 0.0,
            "mean_key_dim_ratio": 0.0,
            "weighted_key_dim_ratio": 0.0,
        }

    dim_ratios: dict[str, float] = {}
    dim_metric_types: dict[str, str] = {}
    for param in search_space:
        name = param.name
        values = [cand.get(name) for cand in candidates if name in cand]
        if len(values) == 0:
            dim_ratios[name] = 0.0
            dim_metric_types[name] = "none"
            continue
        if hasattr(param, "low") and hasattr(param, "high"):
            low = float(getattr(param, "low"))
            high = float(getattr(param, "high"))
            if high > low:
                numeric_values = []
                for value in values:
                    try:
                        numeric_values.append(float(value))
                    except (TypeError, ValueError):
                        pass
                if numeric_values:
                    span = max(numeric_values) - min(numeric_values)
                    dim_ratios[name] = max(0.0, min(1.0, span / (high - low)))
                    dim_metric_types[name] = "continuous_span_proxy"
                    continue
        options = getattr(param, "options", None)
        unique = len({str(value) for value in values})
        if options is not None and len(options) > 0:
            dim_ratios[name] = min(1.0, unique / max(1, len(options)))
            dim_metric_types[name] = "categorical_option_coverage"
        else:
            dim_ratios[name] = min(1.0, unique / max(1, len(values)))
            dim_metric_types[name] = "categorical_seen_ratio_fallback"

    overall = sum(dim_ratios.values()) / max(1, len(dim_ratios))
    underexplored = [name for name, ratio in dim_ratios.items() if ratio < 0.35]
    key_dims = _resolve_key_dimensions(search_space, search_space_meta)
    key_dim_ratios = {
        name: dim_ratios[name]
        for name in key_dims
        if name in dim_ratios
    }
    if key_dim_ratios:
        min_key_dim_ratio = min(key_dim_ratios.values())
        mean_key_dim_ratio = sum(key_dim_ratios.values()) / len(key_dim_ratios)
        weighted_key_dim_ratio = 0.7 * min_key_dim_ratio + 0.3 * mean_key_dim_ratio
    else:
        min_key_dim_ratio = overall
        mean_key_dim_ratio = overall
        weighted_key_dim_ratio = overall
    return {
        "overall_ratio": round(overall, 4),
        "dimension_ratios": dim_ratios,
        "dimension_metric_types": dim_metric_types,
        "underexplored": underexplored,
        "key_dimensions": key_dims,
        "key_dimension_ratios": key_dim_ratios,
        "min_key_dim_ratio": round(float(min_key_dim_ratio), 4),
        "mean_key_dim_ratio": round(float(mean_key_dim_ratio), 4),
        "weighted_key_dim_ratio": round(float(weighted_key_dim_ratio), 4),
    }


def _finite_pool_progress(
    history: list[dict[str, Any]],
    search_space_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    if not search_space_meta:
        return {
            "visited_candidate_count": None,
            "visited_candidate_ratio": None,
            "remaining_candidate_count": None,
            "current_subpool_ratio": None,
        }
    total = search_space_meta.get("candidate_count")
    if total is None:
        return {
            "visited_candidate_count": None,
            "visited_candidate_ratio": None,
            "remaining_candidate_count": None,
            "current_subpool_ratio": None,
        }
    visited = len(
        {
            str(row.get("candidate", {}))
            for row in history
            if isinstance(row.get("candidate"), dict)
        }
    )
    visited_ratio = visited / total if total else 0.0
    remaining = max(0, int(total) - int(visited))
    current_subpool_ratio = None
    if history:
        latest_pool_size = history[-1].get("candidate_pool_size")
        if isinstance(latest_pool_size, (int, float)) and total:
            current_subpool_ratio = float(latest_pool_size) / float(total)
    return {
        "visited_candidate_count": int(visited),
        "visited_candidate_ratio": round(visited_ratio, 6),
        "remaining_candidate_count": int(remaining),
        "current_subpool_ratio": (
            round(float(current_subpool_ratio), 6)
            if current_subpool_ratio is not None
            else None
        ),
    }


def _information_gain_proxy(history: list[dict[str, Any]], window: int = 6) -> float:
    recent = [float(row.get("result", 0.0)) for row in history[-window:]]
    if len(recent) <= 1:
        return 0.0
    mean = sum(recent) / len(recent)
    variance = sum((value - mean) ** 2 for value in recent) / len(recent)
    return round(variance ** 0.5, 6)


def _no_improvement_rounds(history: list[dict[str, Any]], goal: str = "maximize") -> int:
    curve = _best_progress(history, goal=goal)
    if len(curve) <= 1:
        return 0
    current_best = curve[-1]
    count = 0
    for value in reversed(curve[:-1]):
        if value == current_best:
            count += 1
            continue
        break
    return count


def _resolve_key_dimensions(
    search_space,  # noqa: ANN001
    search_space_meta: dict[str, Any] | None,
) -> list[str]:
    meta = search_space_meta or {}
    key_dims = [
        str(name)
        for name in list(meta.get("key_dimensions", []) or [])
        if isinstance(name, str) and str(name).strip()
    ]
    if key_dims:
        return key_dims
    scaffold_dims = [
        str(name)
        for name in list(meta.get("scaffold_dims", []) or [])
        if isinstance(name, str) and str(name).strip()
    ]
    if scaffold_dims:
        return scaffold_dims
    if search_space is None:
        return []
    return [
        str(getattr(param, "name", ""))
        for param in list(search_space)[:2]
        if str(getattr(param, "name", "")).strip()
    ]


def _dominant_value_stats(
    recent: list[dict[str, Any]],
    dims: list[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    ratios: dict[str, float] = {}
    dominant_values: dict[str, Any] = {}
    for name in dims:
        counts: dict[str, int] = {}
        last_value = None
        for candidate in recent:
            if name not in candidate:
                continue
            value = candidate.get(name)
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
            last_value = value
        if not counts:
            ratios[name] = 0.0
            dominant_values[name] = last_value
            continue
        dominant_key, dominant_count = max(counts.items(), key=lambda item: item[1])
        dominant_values[name] = dominant_key
        ratios[name] = round(dominant_count / len(recent), 6)
    return ratios, dominant_values


def _recent_scaffold_concentration(
    history: list[dict[str, Any]],
    search_space_meta: dict[str, Any] | None,
    window: int = 6,
) -> dict[str, Any]:
    recent = [row.get("candidate", {}) for row in history[-window:] if isinstance(row.get("candidate"), dict)]
    if not recent:
        return {"ratio": 0.0, "dominant_scaffold": {}}
    meta = search_space_meta or {}
    scaffold_dims = [
        str(name)
        for name in list(meta.get("scaffold_dims", []) or [])
        if isinstance(name, str) and str(name).strip()
    ]
    feature_columns = list(meta.get("feature_columns", []))
    if not scaffold_dims:
        scaffold_dims = [name for name in feature_columns[:2] if isinstance(name, str)]
    if not scaffold_dims and recent:
        scaffold_dims = list(recent[0].keys())[:2]
    if not scaffold_dims:
        return {"ratio": 0.0, "dominant_scaffold": {}}
    counts: dict[tuple[str, ...], int] = {}
    scaffold_values: dict[tuple[str, ...], dict[str, Any]] = {}
    for candidate in recent:
        key = tuple(str(candidate.get(name)) for name in scaffold_dims)
        counts[key] = counts.get(key, 0) + 1
        scaffold_values[key] = {name: candidate.get(name) for name in scaffold_dims}
    dominant_key, dominant_count = max(counts.items(), key=lambda item: item[1])
    return {
        "ratio": round(dominant_count / len(recent), 6),
        "dominant_scaffold": scaffold_values.get(dominant_key, {}),
    }


def _recent_local_lock_structure(
    history: list[dict[str, Any]],
    search_space_meta: dict[str, Any] | None,
    *,
    window: int = 6,
    threshold: float = 0.67,
) -> dict[str, Any]:
    recent = [row.get("candidate", {}) for row in history[-window:] if isinstance(row.get("candidate"), dict)]
    if not recent:
        return {
            "key_dimensions": [],
            "dominant_value_ratios_by_dim": {},
            "dominant_values_by_dim": {},
            "recent_primary_dim_concentration": 0.0,
            "recent_secondary_dim_concentration": 0.0,
            "one_axis_sweep_detected": False,
            "one_axis_sweep_dimension": None,
            "one_axis_sweep_anchor_dims": [],
            "one_axis_sweep_anchor_values": {},
            "anchor_repeat_count": 0,
            "local_lock_score": 0.0,
        }

    meta = search_space_meta or {}
    feature_columns = [
        str(name)
        for name in list(meta.get("feature_columns", []) or [])
        if isinstance(name, str) and str(name).strip()
    ]
    if not feature_columns:
        feature_columns = list(recent[0].keys())
    key_dims = _resolve_key_dimensions(None, meta)
    ratios_by_dim, dominant_values_by_dim = _dominant_value_stats(recent, feature_columns)
    primary_dim = key_dims[0] if key_dims else None
    secondary_dim = key_dims[1] if len(key_dims) > 1 else None
    primary_conc = float(ratios_by_dim.get(primary_dim, 0.0) or 0.0) if primary_dim else 0.0
    secondary_conc = float(ratios_by_dim.get(secondary_dim, 0.0) or 0.0) if secondary_dim else 0.0

    best_sweep: dict[str, Any] | None = None
    for varying_dim in feature_columns:
        anchor_dims = [name for name in feature_columns if name != varying_dim]
        if not anchor_dims:
            continue
        counts: dict[tuple[str, ...], int] = {}
        anchor_values: dict[tuple[str, ...], dict[str, Any]] = {}
        varying_values: dict[tuple[str, ...], set[str]] = {}
        for candidate in recent:
            key = tuple(str(candidate.get(name)) for name in anchor_dims)
            counts[key] = counts.get(key, 0) + 1
            anchor_values[key] = {name: candidate.get(name) for name in anchor_dims}
            varying_values.setdefault(key, set()).add(str(candidate.get(varying_dim)))
        if not counts:
            continue
        dominant_key, dominant_count = max(counts.items(), key=lambda item: item[1])
        sweep_ratio = dominant_count / len(recent)
        sweep_unique_values = len(varying_values.get(dominant_key, set()))
        if sweep_ratio < threshold or sweep_unique_values < 2:
            continue
        candidate_sweep = {
            "one_axis_sweep_detected": True,
            "one_axis_sweep_dimension": varying_dim,
            "one_axis_sweep_anchor_dims": anchor_dims,
            "one_axis_sweep_anchor_values": anchor_values.get(dominant_key, {}),
            "anchor_repeat_count": int(dominant_count),
            "sweep_ratio": round(float(sweep_ratio), 6),
        }
        if best_sweep is None or (
            candidate_sweep["anchor_repeat_count"],
            candidate_sweep["sweep_ratio"],
        ) > (
            best_sweep["anchor_repeat_count"],
            best_sweep["sweep_ratio"],
        ):
            best_sweep = candidate_sweep

    local_lock_score = max(
        primary_conc,
        secondary_conc,
        float((best_sweep or {}).get("sweep_ratio", 0.0) or 0.0),
    )
    return {
        "key_dimensions": key_dims,
        "dominant_value_ratios_by_dim": ratios_by_dim,
        "dominant_values_by_dim": dominant_values_by_dim,
        "recent_primary_dim_concentration": round(primary_conc, 6),
        "recent_secondary_dim_concentration": round(secondary_conc, 6),
        "one_axis_sweep_detected": bool(best_sweep),
        "one_axis_sweep_dimension": (best_sweep or {}).get("one_axis_sweep_dimension"),
        "one_axis_sweep_anchor_dims": list((best_sweep or {}).get("one_axis_sweep_anchor_dims", [])),
        "one_axis_sweep_anchor_values": dict((best_sweep or {}).get("one_axis_sweep_anchor_values", {})),
        "anchor_repeat_count": int((best_sweep or {}).get("anchor_repeat_count", 0) or 0),
        "local_lock_score": round(float(local_lock_score), 6),
    }


def _scaffold_plane_lock_signal(
    *,
    no_improvement_rounds: int,
    coverage_weighted_key_dim_ratio: float,
    coverage_min_key_dim_ratio: float,
    underexplored_dimensions: list[str],
    consecutive_failed_action_family_rounds: int = 0,
    last_action_family: str | None = None,
) -> dict[str, Any]:
    stall_component = min(1.0, max(0.0, float(no_improvement_rounds)) / 8.0)
    weighted_gap_component = max(0.0, 1.0 - float(coverage_weighted_key_dim_ratio))
    min_gap_component = max(0.0, 1.0 - float(coverage_min_key_dim_ratio))
    failed_shape_component = (
        min(1.0, float(consecutive_failed_action_family_rounds) / 4.0)
        if str(last_action_family or "") == "shape_only"
        else 0.0
    )
    underexplored_component = 1.0 if list(underexplored_dimensions or []) else 0.0
    score = max(
        0.55 * stall_component + 0.45 * weighted_gap_component,
        0.45 * stall_component + 0.35 * min_gap_component + 0.20 * max(
            failed_shape_component,
            0.5 * underexplored_component,
        ),
    )
    detected = bool(
        no_improvement_rounds >= 4
        and (
            coverage_weighted_key_dim_ratio < 0.34
            or coverage_min_key_dim_ratio < 0.30
            or bool(underexplored_dimensions)
        )
    )
    return {
        "scaffold_plane_lock_score": round(float(min(1.0, max(0.0, score))), 6),
        "scaffold_plane_lock_detected": detected,
    }


def _compact_history_tail(
    history: list[dict[str, Any]],
    *,
    window: int = 6,
    reflection_max_chars: int = 160,
) -> list[dict[str, Any]]:
    tail: list[dict[str, Any]] = []
    for row in history[-window:]:
        tail.append(
            {
                "iteration": row.get("iteration"),
                "stage": row.get("stage"),
                "trigger_reasons": row.get("trigger_reasons", []),
                "controller_mode": row.get("controller_mode", "bo_direct"),
                "intervention_type": row.get("intervention_type", "none"),
                "action_package": (
                    {
                        "intent": (row.get("action_package") or {}).get("intent"),
                        "shortlist_policy": (row.get("action_package") or {}).get("shortlist_policy"),
                        "selection_policy": (row.get("action_package") or {}).get("selection_policy"),
                        "verification_policy": (row.get("action_package") or {}).get("verification_policy"),
                    }
                    if isinstance(row.get("action_package"), dict)
                    else None
                ),
                "subspace_active": row.get("subspace_active", False),
                "active_variables": row.get("active_variables", []),
                "candidate": row.get("candidate", {}),
                "result": row.get("result"),
                "improved_best": row.get("improved_best"),
                "semantic_risk_level": row.get("semantic_risk_level", "low"),
                "verification_status": (
                    (row.get("verification_pass") or {}).get("status")
                    if isinstance(row.get("verification_pass"), dict)
                    else None
                ),
                "knowledge_used": row.get("knowledge_used", False),
                "knowledge_source_types": row.get("knowledge_source_types", []),
                "reflection_insight": _compact_text(
                    (row.get("reflection") or {}).get("insight", ""),
                    max_chars=reflection_max_chars,
                ),
            }
        )
    return tail


def _controller_trace_view(trace: dict[str, Any]) -> dict[str, Any]:
    items = []
    for row in list(trace.get("items", []) or []):
        if not isinstance(row, dict):
            continue
        items.append(
            {
                key: value
                for key, value in row.items()
                if key != "reflection_insight"
            }
            | {
                "recent_plan_effective": bool(row.get("improved_best")),
            }
        )
    return {
        **trace,
        "items": items,
    }


def _controller_active_hypothesis_view(active: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_problem_state": active.get("current_problem_state"),
        "open_hypotheses": list(active.get("open_hypotheses", [])),
        "suggested_next_focus": list(active.get("suggested_next_focus", [])),
        "avoid_patterns": list(active.get("avoid_patterns", [])),
        "confidence": active.get("confidence", "low"),
        "last_action_package": active.get("last_action_package"),
        "last_hypothesis": active.get("last_hypothesis"),
        "last_diagnosis": active.get("last_diagnosis"),
        "last_coverage": active.get("last_coverage"),
        "last_semantic_assessment": active.get("last_semantic_assessment"),
        "last_verification_pass": active.get("last_verification_pass"),
    }


def _compact_knowledge_units(
    knowledge_units: list[dict[str, Any]] | None,
    *,
    max_items: int = 5,
    max_chars: int = 240,
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in (knowledge_units or [])[:max_items]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "id": item.get("id"),
                "source_type": item.get("source_type", "unknown"),
                "knowledge_type": item.get("knowledge_type", "unknown"),
                "confidence": item.get("confidence", 0.0),
                "score": item.get("score"),
                "content": _compact_text(item.get("content", ""), max_chars=max_chars),
            }
        )
    return compact


def _derive_verification_feedback_state(
    verification_pass: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(verification_pass, dict):
        return {
            "verification_status_last_round": "not_run",
            "verification_risk_flags_last_round": [],
            "verification_warns_extension": False,
            "verification_identity_uncertain": False,
        }
    risk_flags = [
        str(item).strip()
        for item in list(verification_pass.get("risk_flags", []) or [])
        if str(item).strip()
    ]
    reasoning = str(verification_pass.get("reasoning", "")).strip().lower()
    raw_signal = " ".join([*risk_flags, reasoning]).lower()
    warns_extension = any(
        token in raw_signal
        for token in (
            "extension",
            "sparse_evidence",
            "weak_evidence",
            "insufficient",
            "not_worth_extending",
            "late_budget_compatibility",
            "motif_vs_exact_tuple",
        )
    )
    identity_uncertain = any(
        token in raw_signal
        for token in (
            "identity",
            "attribution",
            "motif",
            "exact_tuple",
            "bundle",
        )
    )
    return {
        "verification_status_last_round": str(verification_pass.get("status", "pass")),
        "verification_risk_flags_last_round": risk_flags,
        "verification_warns_extension": warns_extension,
        "verification_identity_uncertain": identity_uncertain,
    }


def _action_family(action_package: dict[str, Any] | None) -> str:
    package = dict(action_package or {})
    selection_policy = str(package.get("selection_policy", "bo_top1") or "bo_top1")
    focus_policy = str(package.get("focus_policy", "full_space") or "full_space")
    if selection_policy == "bo_top1":
        return "direct"
    if selection_policy == "bo_top1_from_shaped_shortlist":
        return "shape_only_focus" if focus_policy == "temporary_focus" else "shape_only"
    if selection_policy == "select_from_shaped_shortlist":
        return "rerank_focus" if focus_policy == "temporary_focus" else "rerank"
    return "unknown"


def _row_execution_action(row: dict[str, Any]) -> str | None:
    executed = str(row.get("executed_execution_action", "") or "").strip()
    if executed:
        return executed
    requested = str(row.get("requested_execution_action", "") or "").strip()
    if requested:
        return requested
    package = dict(row.get("action_package", {}) or {})
    if not package:
        return None
    selection_policy = str(package.get("selection_policy", "bo_top1") or "bo_top1")
    focus_policy = str(package.get("focus_policy", "full_space") or "full_space")
    if selection_policy == "bo_top1":
        return "direct_bo_pick"
    if selection_policy == "bo_top1_from_shaped_shortlist":
        return "shape_only_bo_pick"
    if selection_policy == "select_from_shaped_shortlist":
        return "focused_shortlist_alt_pick" if focus_policy == "temporary_focus" else "shortlist_alt_pick"
    return None


def _execution_action_family(row: dict[str, Any]) -> str:
    execution_action = _row_execution_action(row)
    if execution_action == "direct_bo_pick":
        return "direct"
    if execution_action == "shape_only_bo_pick":
        return "shape_only"
    if execution_action == "focused_shortlist_alt_pick":
        return "alt_focus"
    if execution_action == "shortlist_alt_pick":
        return "alt_pick"
    return _action_family(row.get("action_package"))


def _consecutive_failed_action_metric(
    history: list[dict[str, Any]],
    *,
    extractor,
    fallback: Any = None,
) -> tuple[Any, int]:
    actionable_rows = [
        row for row in history
        if isinstance(row.get("action_package"), dict)
    ]
    if not actionable_rows:
        return fallback, 0
    last_value = extractor(actionable_rows[-1].get("action_package") or {})
    if actionable_rows[-1].get("improved_best"):
        return last_value, 0
    count = 0
    for row in reversed(actionable_rows):
        current_value = extractor(row.get("action_package") or {})
        if current_value != last_value:
            break
        if row.get("improved_best"):
            break
        count += 1
    return last_value, count


def _action_feedback_state(history: list[dict[str, Any]]) -> dict[str, Any]:
    actionable_rows = [
        row for row in history
        if isinstance(row.get("action_package"), dict) or _row_execution_action(row) is not None
    ]
    if not actionable_rows:
        return {
            "last_action_effective": False,
            "last_action_family": None,
            "last_requested_execution_action": None,
            "last_executed_execution_action": None,
            "last_contract_satisfied": None,
            "last_selection_policy": None,
            "last_shortlist_policy": None,
            "consecutive_failed_action_family_rounds": 0,
            "consecutive_failed_selection_policy_rounds": 0,
            "consecutive_failed_shortlist_policy_rounds": 0,
        }
    last_row = actionable_rows[-1]
    last_action_package = dict(last_row.get("action_package", {}) or {})
    last_action_family = _execution_action_family(last_row)
    consecutive_failed_action_family_rounds = 0
    if not bool(last_row.get("improved_best")):
        for row in reversed(actionable_rows):
            if _execution_action_family(row) != last_action_family:
                break
            if row.get("improved_best"):
                break
            consecutive_failed_action_family_rounds += 1
    last_selection_policy, consecutive_failed_selection_policy_rounds = _consecutive_failed_action_metric(
        actionable_rows,
        extractor=lambda package: str(package.get("selection_policy", "") or "") or None,
        fallback=None,
    )
    last_shortlist_policy, consecutive_failed_shortlist_policy_rounds = _consecutive_failed_action_metric(
        actionable_rows,
        extractor=lambda package: str(package.get("shortlist_policy", "") or "") or None,
        fallback=None,
    )
    return {
        "last_action_effective": bool(last_row.get("improved_best")),
        "last_action_family": last_action_family,
        "last_requested_execution_action": str(last_row.get("requested_execution_action") or "") or None,
        "last_executed_execution_action": str(last_row.get("executed_execution_action") or "") or None,
        "last_contract_satisfied": last_row.get("contract_satisfied"),
        "last_selection_policy": last_selection_policy,
        "last_shortlist_policy": last_shortlist_policy,
        "consecutive_failed_action_family_rounds": consecutive_failed_action_family_rounds,
        "consecutive_failed_selection_policy_rounds": consecutive_failed_selection_policy_rounds,
        "consecutive_failed_shortlist_policy_rounds": consecutive_failed_shortlist_policy_rounds,
        "last_action_package": last_action_package or None,
    }


def _empty_state(prompt_config: PromptConfig) -> dict[str, Any]:
    return {
        "run_identity": {
            "iteration": 0,
            "observations": 0,
            "total_budget": None,
            "remaining_budget": None,
        },
        "optimization_status": {
            "best_observation": None,
            "best_value": None,
            "best_progress_curve_tail": [],
            "best_improvement_last_3": None,
            "no_improvement_rounds": 0,
        },
        "search_dynamics": {
            "recent_duplicate_ratio": 0.0,
            "recent_result_std": 0.0,
            "recent_feasibility_counts": {"accept": 0, "revise": 0, "reject": 0},
            "current_stage_streak": 0,
            "visited_candidate_ratio": None,
            "visited_candidate_count": None,
            "remaining_candidate_count": None,
            "current_subpool_ratio": None,
            "candidate_pool_size": None,
        },
        "coverage_state": {
            "coverage_overall_ratio": 0.0,
            "coverage_dimension_ratios": {},
            "coverage_dimension_metric_types": {},
            "underexplored_dimensions": [],
            "key_dimensions": [],
            "key_dimension_ratios": {},
            "coverage_min_key_dim_ratio": 0.0,
            "coverage_mean_key_dim_ratio": 0.0,
            "coverage_weighted_key_dim_ratio": 0.0,
        },
        "structure_state": {
            "recent_scaffold_concentration": 0.0,
            "dominant_scaffold": {},
            "recent_primary_dim_concentration": 0.0,
            "recent_secondary_dim_concentration": 0.0,
            "dominant_values_by_dim": {},
            "one_axis_sweep_detected": False,
            "one_axis_sweep_dimension": None,
            "one_axis_sweep_anchor_dims": [],
            "one_axis_sweep_anchor_values": {},
            "anchor_repeat_count": 0,
            "local_lock_score": 0.0,
        },
        "active_hypothesis_state": {
            "current_problem_state": "full_space_search",
            "open_hypotheses": [],
            "suggested_next_focus": [],
            "avoid_patterns": [],
            "trusted_patterns": [],
            "confidence": "low",
            "summary": "",
            "last_action_package": None,
            "last_hypothesis": None,
            "last_diagnosis": None,
            "last_coverage": None,
            "last_semantic_assessment": None,
            "last_reflection": None,
            "note_history": [],
        },
        "verification_feedback_state": {
            "verification_status_last_round": "not_run",
            "verification_risk_flags_last_round": [],
            "verification_warns_extension": False,
            "verification_identity_uncertain": False,
        },
        "action_feedback_state": {
            "last_action_effective": False,
            "last_action_family": None,
            "last_requested_execution_action": None,
            "last_executed_execution_action": None,
            "last_contract_satisfied": None,
            "last_selection_policy": None,
            "last_shortlist_policy": None,
            "consecutive_failed_action_family_rounds": 0,
            "consecutive_failed_selection_policy_rounds": 0,
            "consecutive_failed_shortlist_policy_rounds": 0,
        },
        "recent_decision_trace": {
            "items": [],
            "window_size": int(prompt_config.history_tail_window),
            "compressed_prefix": {
                "compressed_round_count": 0,
                "last_improving_iteration": None,
                "controller_modes_seen": [],
                "action_packages_seen": [],
                "stages_seen": [],
                "knowledge_used_rounds": 0,
            },
            "compression": {
                "trigger": None,
                "compression_count": 0,
                "last_compressed_iteration": 0,
            },
        },
        "external_context_refs": {
            "knowledge_units": [],
            "knowledge_query": "",
            "knowledge_meta": {},
        },
        "state_meta": {
            "version": "v0.6.0",
            "history_length": 0,
            "history_tail_window": int(prompt_config.history_tail_window),
        },
    }


def _build_node_state_views(
    context: dict[str, Any],
    state: dict[str, Any],
    *,
    controller_reflection_input_mode: str = "full",
) -> dict[str, dict[str, Any]]:
    common = {
        "iteration": context.get("iteration"),
        "observations": context.get("observations"),
        "total_budget": context.get("total_budget"),
        "remaining_budget": context.get("remaining_budget"),
        "no_improvement_rounds": context.get("no_improvement_rounds", 0),
        "num_history": context.get("num_history"),
        "best_observation": context.get("best_observation"),
        "best_value": context.get("best_value"),
        "search_space_meta": context.get("search_space_meta", {}),
        "knowledge_units": context.get("knowledge_units", []),
        "knowledge_query": context.get("knowledge_query", ""),
        "knowledge_meta": context.get("knowledge_meta", {}),
        "working_memory_summary": context.get("working_memory_summary", {}),
        "verification_feedback_state": state.get("verification_feedback_state", {}),
        "action_feedback_state": state.get("action_feedback_state", {}),
    }
    return {
        "hypothesis_action": {
            **common,
            "optimization_status": state.get("optimization_status", {}),
            "coverage_state": state.get("coverage_state", {}),
            "structure_state": state.get("structure_state", {}),
            "active_hypothesis_state": state.get("active_hypothesis_state", {}),
            "recent_decision_trace": state.get("recent_decision_trace", {}),
        },
        "stagnation_diagnosis": {
            **common,
            "optimization_status": state.get("optimization_status", {}),
            "search_dynamics": state.get("search_dynamics", {}),
            "coverage_state": state.get("coverage_state", {}),
            "structure_state": state.get("structure_state", {}),
            "recent_decision_trace": state.get("recent_decision_trace", {}),
        },
        "coverage_insight": {
            **common,
            "search_dynamics": state.get("search_dynamics", {}),
            "coverage_state": state.get("coverage_state", {}),
            "structure_state": state.get("structure_state", {}),
            "recent_decision_trace": state.get("recent_decision_trace", {}),
        },
        "controller_plan": {
            **common,
            "optimization_status": state.get("optimization_status", {}),
            "search_dynamics": state.get("search_dynamics", {}),
            "coverage_state": state.get("coverage_state", {}),
            "structure_state": state.get("structure_state", {}),
            "active_hypothesis_state": (
                _controller_active_hypothesis_view(state.get("active_hypothesis_state", {}))
                if str(controller_reflection_input_mode).strip().lower() != "full"
                else state.get("active_hypothesis_state", {})
            ),
            "recent_decision_trace": (
                _controller_trace_view(state.get("recent_decision_trace", {}))
                if str(controller_reflection_input_mode).strip().lower() != "full"
                else state.get("recent_decision_trace", {})
            ),
        },
        "semantic_assessment": {
            **common,
            "optimization_status": state.get("optimization_status", {}),
            "active_hypothesis_state": state.get("active_hypothesis_state", {}),
            "recent_decision_trace": state.get("recent_decision_trace", {}),
        },
        "verification_pass": {
            **common,
            "optimization_status": state.get("optimization_status", {}),
            "active_hypothesis_state": state.get("active_hypothesis_state", {}),
            "recent_decision_trace": state.get("recent_decision_trace", {}),
        },
        "reflection_action": {
            **common,
            "optimization_status": state.get("optimization_status", {}),
            "search_dynamics": state.get("search_dynamics", {}),
            "active_hypothesis_state": state.get("active_hypothesis_state", {}),
            "recent_decision_trace": state.get("recent_decision_trace", {}),
        },
        "search_constraints": {
            **common,
            "optimization_status": state.get("optimization_status", {}),
            "search_dynamics": state.get("search_dynamics", {}),
            "coverage_state": state.get("coverage_state", {}),
            "structure_state": state.get("structure_state", {}),
            "active_hypothesis_state": state.get("active_hypothesis_state", {}),
            "recent_decision_trace": state.get("recent_decision_trace", {}),
        },
        "shortlist_rerank": {
            **common,
            "optimization_status": state.get("optimization_status", {}),
            "search_dynamics": state.get("search_dynamics", {}),
            "structure_state": state.get("structure_state", {}),
            "active_hypothesis_state": state.get("active_hypothesis_state", {}),
            "recent_decision_trace": state.get("recent_decision_trace", {}),
        },
    }


class OnlineDecisionState:
    """Current-run state with fixed slots for downstream decision nodes."""

    def __init__(
        self,
        prompt_config: PromptConfig | None = None,
        *,
        controller_reflection_input_mode: str = "full",
    ) -> None:
        self.prompt_config = prompt_config or PromptConfig()
        self.controller_reflection_input_mode = str(controller_reflection_input_mode or "full")
        self._state = _empty_state(self.prompt_config)

    def summarize(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def load_working_memory_summary(self, summary: dict[str, Any] | None) -> None:
        if not summary:
            return
        research_state = dict(summary.get("research_state", {}) or {})
        active = dict(self._state["active_hypothesis_state"])
        active["current_problem_state"] = research_state.get(
            "current_phase",
            active["current_problem_state"],
        )
        active["open_hypotheses"] = list(research_state.get("open_hypotheses", []))
        active["suggested_next_focus"] = list(research_state.get("suggested_next_focus", []))
        active["trusted_patterns"] = list(research_state.get("trusted_patterns", []))
        active["confidence"] = research_state.get("confidence", active["confidence"])
        active["summary"] = research_state.get("summary", active["summary"])
        active["last_action_package"] = summary.get("last_action_package")
        active["last_hypothesis"] = summary.get("last_hypothesis")
        active["last_diagnosis"] = summary.get("last_diagnosis")
        active["last_coverage"] = summary.get("last_coverage")
        active["last_semantic_assessment"] = summary.get("last_semantic_assessment")
        active["last_verification_pass"] = summary.get("last_verification_pass")
        active["last_reflection"] = summary.get("last_reflection")
        active["note_history"] = list(summary.get("notes", []))[-10:]
        self._state["active_hypothesis_state"] = active
        verification_feedback = dict(summary.get("verification_feedback_state", {}) or {})
        if verification_feedback:
            self._state["verification_feedback_state"] = {
                **dict(self._state.get("verification_feedback_state", {})),
                **verification_feedback,
            }
        action_feedback = dict(summary.get("action_feedback_state", {}) or {})
        if action_feedback:
            self._state["action_feedback_state"] = {
                **dict(self._state.get("action_feedback_state", {})),
                **action_feedback,
            }

    def refresh_from_history(
        self,
        *,
        bootstrap_history: list[dict[str, Any]] | None = None,
        history: list[dict[str, Any]],
        best_observation: dict[str, Any] | None,
        current_best: float | None = None,
        iteration: int,
        observations: int,
        total_budget: int | None,
        search_space=None,  # noqa: ANN001
        search_space_meta: dict[str, Any] | None = None,
        goal: str = "maximize",
        knowledge_units: list[dict[str, Any]] | None = None,
        knowledge_query: str | None = None,
        knowledge_meta: dict[str, Any] | None = None,
    ) -> None:
        combined_history = [*(bootstrap_history or []), *history]
        best_curve = _best_progress(combined_history, goal=goal)
        delta_last_3 = None
        if len(best_curve) >= 4:
            if goal == "minimize":
                delta_last_3 = best_curve[-4] - best_curve[-1]
            else:
                delta_last_3 = best_curve[-1] - best_curve[-4]
        coverage = _coverage_snapshot(
            combined_history,
            search_space=search_space,
            search_space_meta=search_space_meta,
        )
        finite_pool_progress = _finite_pool_progress(combined_history, search_space_meta)
        scaffold = _recent_scaffold_concentration(combined_history, search_space_meta)
        local_lock = _recent_local_lock_structure(combined_history, search_space_meta)
        trace_window = int(self.prompt_config.history_tail_window)
        history_tail = _compact_history_tail(
            combined_history,
            window=trace_window,
            reflection_max_chars=self.prompt_config.history_reflection_max_chars,
        )
        older_history = (
            combined_history[:-trace_window]
            if len(combined_history) > trace_window
            else []
        )
        previous_trace = dict(self._state.get("recent_decision_trace", {}))
        previous_prefix = dict(previous_trace.get("compressed_prefix", {}))
        previous_compression = dict(previous_trace.get("compression", {}))
        action_feedback = _action_feedback_state(history)
        scaffold_plane_lock = _scaffold_plane_lock_signal(
            no_improvement_rounds=_no_improvement_rounds(combined_history, goal=goal),
            coverage_weighted_key_dim_ratio=float(coverage["weighted_key_dim_ratio"] or 0.0),
            coverage_min_key_dim_ratio=float(coverage["min_key_dim_ratio"] or 0.0),
            underexplored_dimensions=list(coverage["underexplored"] or []),
            consecutive_failed_action_family_rounds=int(
                action_feedback.get("consecutive_failed_action_family_rounds", 0) or 0
            ),
            last_action_family=str(action_feedback.get("last_action_family") or ""),
        )

        compression_trigger = None
        if older_history:
            compression_trigger = "history_window"
        if len(history) >= 2:
            last_mode = str(history[-1].get("controller_mode", ""))
            prev_mode = str(history[-2].get("controller_mode", ""))
            last_stage = str(history[-1].get("stage", ""))
            prev_stage = str(history[-2].get("stage", ""))
            if last_mode != prev_mode or last_stage != prev_stage:
                compression_trigger = "phase_change"

        last_improving_iteration = None
        knowledge_used_rounds = 0
        controller_modes_seen: set[str] = set()
        action_packages_seen: set[str] = set()
        stages_seen: set[str] = set()
        for row in older_history:
            if row.get("improved_best"):
                last_improving_iteration = row.get("iteration")
            if row.get("knowledge_used"):
                knowledge_used_rounds += 1
            if row.get("controller_mode"):
                controller_modes_seen.add(str(row.get("controller_mode")))
            if isinstance(row.get("action_package"), dict):
                action_package = row["action_package"]
                action_packages_seen.add(
                    ":".join(
                        [
                            str(action_package.get("intent", "")),
                            str(action_package.get("selection_policy", "")),
                        ]
                    ).strip(":")
                )
            if row.get("stage"):
                stages_seen.add(str(row.get("stage")))

        prefix_count = len(older_history)
        compression_count = int(previous_compression.get("compression_count", 0) or 0)
        previous_prefix_count = int(previous_prefix.get("compressed_round_count", 0) or 0)
        if prefix_count > previous_prefix_count or compression_trigger == "phase_change":
            compression_count += 1

        best_value = current_best if current_best is not None else (best_curve[-1] if best_curve else None)

        self._state["run_identity"] = {
            "iteration": iteration,
            "observations": observations,
            "total_budget": total_budget,
            "remaining_budget": (
                max(0, int(total_budget) - int(observations))
                if total_budget is not None
                else None
            ),
        }
        self._state["optimization_status"] = {
            "best_observation": best_observation,
            "best_value": best_value,
            "best_progress_curve_tail": best_curve[-10:],
            "best_improvement_last_3": delta_last_3,
            "no_improvement_rounds": _no_improvement_rounds(combined_history, goal=goal),
        }
        self._state["search_dynamics"] = {
            "recent_duplicate_ratio": _recent_duplicate_ratio(combined_history, window=trace_window),
            "recent_result_std": _information_gain_proxy(combined_history, window=trace_window),
            "recent_feasibility_counts": _recent_feasibility_counts(combined_history, window=trace_window),
            "current_stage_streak": _stage_streak(combined_history),
            "visited_candidate_ratio": finite_pool_progress["visited_candidate_ratio"],
            "visited_candidate_count": finite_pool_progress["visited_candidate_count"],
            "remaining_candidate_count": finite_pool_progress["remaining_candidate_count"],
            "current_subpool_ratio": finite_pool_progress["current_subpool_ratio"],
            "candidate_pool_size": (
                combined_history[-1].get("candidate_pool_size")
                if combined_history
                else None
            ),
        }
        self._state["coverage_state"] = {
            "coverage_overall_ratio": coverage["overall_ratio"],
            "coverage_dimension_ratios": coverage["dimension_ratios"],
            "coverage_dimension_metric_types": coverage["dimension_metric_types"],
            "underexplored_dimensions": coverage["underexplored"],
            "key_dimensions": coverage["key_dimensions"],
            "key_dimension_ratios": coverage["key_dimension_ratios"],
            "coverage_min_key_dim_ratio": coverage["min_key_dim_ratio"],
            "coverage_mean_key_dim_ratio": coverage["mean_key_dim_ratio"],
            "coverage_weighted_key_dim_ratio": coverage["weighted_key_dim_ratio"],
        }
        self._state["structure_state"] = {
            "recent_scaffold_concentration": scaffold["ratio"],
            "dominant_scaffold": scaffold["dominant_scaffold"],
            "recent_primary_dim_concentration": local_lock["recent_primary_dim_concentration"],
            "recent_secondary_dim_concentration": local_lock["recent_secondary_dim_concentration"],
            "dominant_values_by_dim": local_lock["dominant_values_by_dim"],
            "one_axis_sweep_detected": local_lock["one_axis_sweep_detected"],
            "one_axis_sweep_dimension": local_lock["one_axis_sweep_dimension"],
            "one_axis_sweep_anchor_dims": local_lock["one_axis_sweep_anchor_dims"],
            "one_axis_sweep_anchor_values": local_lock["one_axis_sweep_anchor_values"],
            "anchor_repeat_count": local_lock["anchor_repeat_count"],
            "local_lock_score": local_lock["local_lock_score"],
            "scaffold_plane_lock_score": scaffold_plane_lock["scaffold_plane_lock_score"],
            "scaffold_plane_lock_detected": scaffold_plane_lock["scaffold_plane_lock_detected"],
        }
        self._state["recent_decision_trace"] = {
            "items": history_tail,
            "window_size": trace_window,
            "compressed_prefix": {
                "compressed_round_count": prefix_count,
                "last_improving_iteration": last_improving_iteration,
                "controller_modes_seen": sorted(controller_modes_seen),
                "action_packages_seen": sorted(item for item in action_packages_seen if item),
                "stages_seen": sorted(stages_seen),
                "knowledge_used_rounds": knowledge_used_rounds,
            },
            "compression": {
                "trigger": compression_trigger,
                "compression_count": compression_count,
                "last_compressed_iteration": (
                    iteration
                    if compression_trigger is not None
                    else int(previous_compression.get("last_compressed_iteration", 0) or 0)
                ),
            },
        }
        self._state["external_context_refs"] = {
            "knowledge_units": _compact_knowledge_units(
                knowledge_units,
                max_items=self.prompt_config.knowledge_max_items,
                max_chars=self.prompt_config.knowledge_max_chars,
            ),
            "knowledge_query": knowledge_query or "",
            "knowledge_meta": deepcopy(knowledge_meta or {}),
        }
        self._state["action_feedback_state"] = action_feedback
        self._state["state_meta"] = {
            "version": "v0.6.0",
            "history_length": len(combined_history),
            "bootstrap_history_length": len(bootstrap_history or []),
            "decision_history_length": len(history),
            "history_tail_window": trace_window,
        }

    def apply_round_outputs(
        self,
        *,
        diagnosis: dict[str, Any] | None = None,
        hypothesis_action: dict[str, Any] | None = None,
        coverage_insight: dict[str, Any] | None = None,
        semantic_assessment: dict[str, Any] | None = None,
        verification_pass: dict[str, Any] | None = None,
        reflection_action: dict[str, Any] | None = None,
        controller_plan: dict[str, Any] | None = None,
    ) -> None:
        active = dict(self._state["active_hypothesis_state"])
        verification_feedback = dict(self._state.get("verification_feedback_state", {}))
        action_package = (
            dict(controller_plan.get("action_package", {}) or {})
            if isinstance(controller_plan, dict)
            else {}
        )
        active["last_action_package"] = action_package or None
        if hypothesis_action is not None:
            active["last_hypothesis"] = hypothesis_action
            active["open_hypotheses"] = list(hypothesis_action.get("hypotheses", []))
            active["suggested_next_focus"] = list(
                hypothesis_action.get("suggested_focus_variables", [])
            )
        if diagnosis is not None:
            active["last_diagnosis"] = diagnosis
            current_problem_state = "full_space_search"
            if diagnosis.get("is_stagnating", False):
                current_problem_state = "stagnation_recovery"
            execution_action = str(
                (controller_plan or {}).get("executed_execution_action")
                or (controller_plan or {}).get("requested_execution_action")
                or action_package.get("requested_execution_action")
                or ""
            ).strip()
            if execution_action == "focused_shortlist_alt_pick" or action_package.get("focus_policy") == "temporary_focus":
                current_problem_state = "focused_probe"
            elif execution_action == "shortlist_alt_pick":
                current_problem_state = "shortlist_reassessment"
            elif action_package.get("intent") == "probe":
                current_problem_state = "probe_reassessment"
            elif action_package.get("selection_policy") == "select_from_shaped_shortlist":
                current_problem_state = "shortlist_reassessment"
            elif controller_plan and controller_plan.get("intervention_type") == "bo_rerank_topk":
                current_problem_state = "rerank_reassessment"
            active["current_problem_state"] = current_problem_state
        if coverage_insight is not None:
            active["last_coverage"] = coverage_insight
        if semantic_assessment is not None:
            active["last_semantic_assessment"] = semantic_assessment
        active["last_verification_pass"] = verification_pass
        verification_feedback = _derive_verification_feedback_state(verification_pass)
        if reflection_action is not None:
            active["last_reflection"] = reflection_action
            suggested_focus = reflection_action.get("suggested_focus", [])
            if suggested_focus:
                active["suggested_next_focus"] = list(suggested_focus)
            active["avoid_patterns"] = list(reflection_action.get("avoid_pattern", []))
            active["summary"] = reflection_action.get("insight", active.get("summary", ""))
            active["confidence"] = reflection_action.get("confidence", active.get("confidence", "low"))
            note = {
                "insight": reflection_action.get("insight", ""),
                "next_step_hypothesis": reflection_action.get("next_step_hypothesis", ""),
                "confidence": reflection_action.get("confidence", "low"),
            }
            active["note_history"] = (list(active.get("note_history", [])) + [note])[-10:]
        self._state["active_hypothesis_state"] = active
        self._state["verification_feedback_state"] = verification_feedback

    def working_memory_summary(self) -> dict[str, Any]:
        active = dict(self._state["active_hypothesis_state"])
        coverage_state = dict(self._state["coverage_state"])
        return {
            "research_state": {
                "current_phase": active.get("current_problem_state", "full_space_search"),
                "promising_regions": [],
                "weak_regions": coverage_state.get("underexplored_dimensions", []),
                "trusted_patterns": active.get("trusted_patterns", []),
                "open_hypotheses": active.get("open_hypotheses", []),
                "suggested_next_focus": active.get("suggested_next_focus", []),
                "confidence": active.get("confidence", "low"),
                "summary": active.get("summary", ""),
            },
            "last_hypothesis": active.get("last_hypothesis"),
            "last_diagnosis": active.get("last_diagnosis"),
            "last_coverage": active.get("last_coverage"),
            "last_action_package": active.get("last_action_package"),
            "last_semantic_assessment": active.get("last_semantic_assessment"),
            "last_verification_pass": active.get("last_verification_pass"),
            "last_reflection": active.get("last_reflection"),
            "verification_feedback_state": dict(self._state.get("verification_feedback_state", {})),
            "action_feedback_state": dict(self._state.get("action_feedback_state", {})),
            "notes": list(active.get("note_history", [])),
        }

    def build_decision_context(self, *, search_space_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        run_identity = dict(self._state["run_identity"])
        optimization = dict(self._state["optimization_status"])
        search_dynamics = dict(self._state["search_dynamics"])
        coverage = dict(self._state["coverage_state"])
        structure = dict(self._state["structure_state"])
        external = dict(self._state["external_context_refs"])
        trace = dict(self._state["recent_decision_trace"])
        verification_feedback = dict(self._state.get("verification_feedback_state", {}))
        action_feedback = dict(self._state.get("action_feedback_state", {}))
        active = dict(self._state["active_hypothesis_state"])
        context = {
            "iteration": run_identity.get("iteration"),
            "observations": run_identity.get("observations"),
            "total_budget": run_identity.get("total_budget"),
            "remaining_budget": run_identity.get("remaining_budget"),
            "num_history": self._state["state_meta"].get("history_length", 0),
            "best_observation": optimization.get("best_observation"),
            "best_value": optimization.get("best_value"),
            "best_progress_curve_tail": optimization.get("best_progress_curve_tail", []),
            "best_improvement_last_3": optimization.get("best_improvement_last_3"),
            "no_improvement_rounds": optimization.get("no_improvement_rounds", 0),
            "recent_duplicate_ratio": search_dynamics.get("recent_duplicate_ratio", 0.0),
            "recent_result_std": search_dynamics.get("recent_result_std", 0.0),
            "recent_feasibility_counts": search_dynamics.get("recent_feasibility_counts", {}),
            "current_stage_streak": search_dynamics.get("current_stage_streak", 0),
            "visited_candidate_ratio": search_dynamics.get("visited_candidate_ratio"),
            "visited_candidate_count": search_dynamics.get("visited_candidate_count"),
            "remaining_candidate_count": search_dynamics.get("remaining_candidate_count"),
            "current_subpool_ratio": search_dynamics.get("current_subpool_ratio"),
            "coverage_overall_ratio": coverage.get("coverage_overall_ratio", 0.0),
            "coverage_dimension_ratios": coverage.get("coverage_dimension_ratios", {}),
            "coverage_dimension_metric_types": coverage.get("coverage_dimension_metric_types", {}),
            "underexplored_dimensions": coverage.get("underexplored_dimensions", []),
            "key_dimensions": coverage.get("key_dimensions", []),
            "key_dimension_ratios": coverage.get("key_dimension_ratios", {}),
            "coverage_min_key_dim_ratio": coverage.get("coverage_min_key_dim_ratio", 0.0),
            "coverage_mean_key_dim_ratio": coverage.get("coverage_mean_key_dim_ratio", 0.0),
            "coverage_weighted_key_dim_ratio": coverage.get("coverage_weighted_key_dim_ratio", 0.0),
            "recent_scaffold_concentration": structure.get("recent_scaffold_concentration", 0.0),
            "dominant_scaffold": structure.get("dominant_scaffold", {}),
            "recent_primary_dim_concentration": structure.get("recent_primary_dim_concentration", 0.0),
            "recent_secondary_dim_concentration": structure.get("recent_secondary_dim_concentration", 0.0),
            "dominant_values_by_dim": structure.get("dominant_values_by_dim", {}),
            "one_axis_sweep_detected": structure.get("one_axis_sweep_detected", False),
            "one_axis_sweep_dimension": structure.get("one_axis_sweep_dimension"),
            "one_axis_sweep_anchor_dims": structure.get("one_axis_sweep_anchor_dims", []),
            "one_axis_sweep_anchor_values": structure.get("one_axis_sweep_anchor_values", {}),
            "anchor_repeat_count": structure.get("anchor_repeat_count", 0),
            "local_lock_score": structure.get("local_lock_score", 0.0),
            "scaffold_plane_lock_score": structure.get("scaffold_plane_lock_score", 0.0),
            "scaffold_plane_lock_detected": structure.get("scaffold_plane_lock_detected", False),
            "scaffold_concentration": structure,
            "last_action_package": active.get("last_action_package"),
            "last_action_effective": action_feedback.get("last_action_effective", False),
            "last_action_family": action_feedback.get("last_action_family"),
            "last_requested_execution_action": action_feedback.get("last_requested_execution_action"),
            "last_executed_execution_action": action_feedback.get("last_executed_execution_action"),
            "last_contract_satisfied": action_feedback.get("last_contract_satisfied"),
            "last_selection_policy": action_feedback.get("last_selection_policy"),
            "last_shortlist_policy": action_feedback.get("last_shortlist_policy"),
            "consecutive_failed_action_family_rounds": action_feedback.get(
                "consecutive_failed_action_family_rounds",
                0,
            ),
            "consecutive_failed_selection_policy_rounds": action_feedback.get(
                "consecutive_failed_selection_policy_rounds",
                0,
            ),
            "consecutive_failed_shortlist_policy_rounds": action_feedback.get(
                "consecutive_failed_shortlist_policy_rounds",
                0,
            ),
            "verification_status_last_round": verification_feedback.get("verification_status_last_round", "not_run"),
            "verification_risk_flags_last_round": verification_feedback.get("verification_risk_flags_last_round", []),
            "verification_warns_extension": verification_feedback.get("verification_warns_extension", False),
            "verification_identity_uncertain": verification_feedback.get("verification_identity_uncertain", False),
            "search_space_meta": search_space_meta or {},
            "recent_history_tail": trace.get("items", []),
            "trace_compression": trace.get("compression", {}),
            "trace_compressed_prefix": trace.get("compressed_prefix", {}),
            "working_memory_summary": self.working_memory_summary(),
            "knowledge_units": external.get("knowledge_units", []),
            "knowledge_query": external.get("knowledge_query", ""),
            "knowledge_meta": external.get("knowledge_meta", {}),
            "online_decision_state": self.summarize(),
        }
        context["node_state_views"] = _build_node_state_views(
            context,
            self._state,
            controller_reflection_input_mode=self.controller_reflection_input_mode,
        )
        return context
