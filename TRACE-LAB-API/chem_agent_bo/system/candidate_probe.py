"""Finite-pool candidate probe helpers.

The probe is a candidate-surfacing utility: it proposes extra legal finite-pool
candidates for a shortlist, but it does not select the final experiment.
"""

from __future__ import annotations

from typing import Any


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_key(candidate: dict[str, Any], feature_columns: list[str]) -> tuple[str, ...]:
    return tuple(str(candidate.get(name)) for name in feature_columns)


def _ranked_history(
    history: list[dict[str, Any]],
    *,
    goal: str,
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in history
        if isinstance(row.get("candidate"), dict) and _float_or_none(row.get("result")) is not None
    ]
    return sorted(
        rows,
        key=lambda row: float(row.get("result")),
        reverse=str(goal).strip().lower() != "minimize",
    )


def _history_value_scores(
    rows: list[dict[str, Any]],
    *,
    goal: str,
) -> dict[int, float]:
    values = [_float_or_none(row.get("result")) for row in rows]
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {}
    low = min(numeric)
    high = max(numeric)
    span = high - low
    scores: dict[int, float] = {}
    for index, value in enumerate(values):
        if value is None:
            continue
        if span <= 1e-12:
            scores[index] = 1.0
        elif str(goal).strip().lower() == "minimize":
            scores[index] = (high - float(value)) / span
        else:
            scores[index] = (float(value) - low) / span
    return scores


def build_probe_candidates(
    *,
    candidate_records: list[dict[str, Any]],
    history: list[dict[str, Any]],
    shortlist_candidates: list[dict[str, Any]],
    feature_columns: list[str],
    scaffold_dims: list[str],
    allowed_keys: set[tuple[str, ...]] | None = None,
    include_map: dict[str, list[str]] | None = None,
    include_mode: str = "hard",
    goal: str = "maximize",
    max_candidates: int = 4,
    top_history: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return candidate records that should be added to the visible shortlist.

    Scoring is deliberately evidence-only and label-free for unobserved candidates.
    It rewards candidates that recombine dimension values seen in high observed
    conditions, while excluding already observed candidates and existing shortlist
    candidates.
    """

    feature_columns = [str(name) for name in feature_columns if str(name).strip()]
    scaffold_dims = [name for name in scaffold_dims if name in feature_columns]
    if not candidate_records or not feature_columns:
        return [], {"enabled": False, "reason": "missing_candidate_records_or_features"}

    max_candidates = max(0, int(max_candidates or 0))
    if max_candidates <= 0:
        return [], {"enabled": False, "reason": "max_candidates_zero"}

    ranked = _ranked_history(history, goal=goal)
    support_rows = ranked[: max(1, int(top_history or 1))]
    if not support_rows:
        return [], {"enabled": False, "reason": "missing_observed_support"}
    history_value_scores = _history_value_scores(support_rows, goal=goal)

    observed_keys = {
        _candidate_key(row["candidate"], feature_columns)
        for row in history
        if isinstance(row.get("candidate"), dict)
    }
    shortlist_keys = {
        _candidate_key(item["candidate"], feature_columns)
        for item in shortlist_candidates
        if isinstance(item.get("candidate"), dict)
    }
    legal_keys = allowed_keys if allowed_keys is not None else {
        tuple(str(v) for v in item.get("key", ()))
        for item in candidate_records
        if item.get("key")
    }
    normalized_include_map = {
        str(name): {str(value) for value in values}
        for name, values in dict(include_map or {}).items()
        if str(name) in feature_columns and values
    }
    normalized_include_mode = str(include_mode or "hard").strip().lower()
    if normalized_include_mode not in {"hard", "soft", "hybrid"}:
        normalized_include_mode = "hard"

    dim_support: dict[str, dict[str, float]] = {name: {} for name in feature_columns}
    pair_support: dict[tuple[str, str], float] = {}
    scaffold_support: dict[tuple[str, ...], float] = {}
    max_rank = max(1, len(support_rows))
    for rank, row in enumerate(support_rows):
        candidate = row["candidate"]
        weight = float(max_rank - rank) / float(max_rank)
        for name in feature_columns:
            value = str(candidate.get(name))
            dim_support[name][value] = max(dim_support[name].get(value, 0.0), weight)
        for left_index, left_name in enumerate(feature_columns):
            for right_name in feature_columns[left_index + 1 :]:
                pair = (str(candidate.get(left_name)), str(candidate.get(right_name)))
                pair_support[pair] = max(pair_support.get(pair, 0.0), weight)
        if scaffold_dims:
            scaffold_key = tuple(str(candidate.get(name)) for name in scaffold_dims)
            scaffold_support[scaffold_key] = max(scaffold_support.get(scaffold_key, 0.0), weight)

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for order, record in enumerate(candidate_records):
        candidate = record.get("candidate")
        key = tuple(str(value) for value in record.get("key", ()))
        if not isinstance(candidate, dict) or not key:
            continue
        if key not in legal_keys or key in observed_keys or key in shortlist_keys:
            continue
        if normalized_include_mode == "hard" and normalized_include_map and any(
            str(candidate.get(name)) not in allowed_values
            for name, allowed_values in normalized_include_map.items()
        ):
            continue

        dim_hits = {
            name: dim_support.get(name, {}).get(str(candidate.get(name)), 0.0)
            for name in feature_columns
        }
        dim_score = sum(dim_hits.values()) / max(1, len(feature_columns))
        pair_values = []
        for left_index, left_name in enumerate(feature_columns):
            for right_name in feature_columns[left_index + 1 :]:
                pair_values.append(
                    pair_support.get((str(candidate.get(left_name)), str(candidate.get(right_name))), 0.0)
                )
        pair_score = sum(pair_values) / max(1, len(pair_values))
        scaffold_key = tuple(str(candidate.get(name)) for name in scaffold_dims)
        scaffold_score = scaffold_support.get(scaffold_key, 0.0) if scaffold_dims else 0.0
        best_match_fraction = 0.0
        analogue_value_score = 0.0
        best_analogue_match_fraction = 0.0
        best_analogue_result = None
        for support_index, row in enumerate(support_rows):
            support_candidate = row["candidate"]
            match_fraction = sum(
                1 for name in feature_columns if str(candidate.get(name)) == str(support_candidate.get(name))
            ) / max(1, len(feature_columns))
            best_match_fraction = max(best_match_fraction, match_fraction)
            value_score = history_value_scores.get(support_index, 0.0)
            analogue_score = float(value_score) * float(match_fraction)
            if analogue_score > analogue_value_score:
                analogue_value_score = analogue_score
                best_analogue_match_fraction = match_fraction
                best_analogue_result = _float_or_none(row.get("result"))

        if normalized_include_map:
            direction_hits = sum(
                1
                for name, allowed_values in normalized_include_map.items()
                if str(candidate.get(name)) in allowed_values
            )
            direction_score = direction_hits / max(1, len(normalized_include_map))
        else:
            direction_score = 0.0
        score = (
            0.34 * dim_score
            + 0.18 * pair_score
            + 0.10 * best_match_fraction
            + 0.04 * scaffold_score
            + 0.24 * analogue_value_score
            + 0.10 * direction_score
        )
        scored.append(
            (
                score,
                order,
                {
                    **record,
                    "probe_score": round(float(score), 6),
                    "probe_evidence": {
                        "dim_support": {k: round(v, 4) for k, v in dim_hits.items() if v > 0},
                        "pair_score": round(float(pair_score), 4),
                        "best_match_fraction": round(float(best_match_fraction), 4),
                        "scaffold_score": round(float(scaffold_score), 4),
                        "analogue_value_score": round(float(analogue_value_score), 4),
                        "best_analogue_match_fraction": round(float(best_analogue_match_fraction), 4),
                        "best_analogue_result": best_analogue_result,
                        "direction_constraints": {
                            key: sorted(values)
                            for key, values in normalized_include_map.items()
                        },
                        "direction_include_mode": normalized_include_mode,
                    },
                },
            )
        )

    scored.sort(key=lambda item: (-item[0], item[1]))
    if normalized_include_mode == "hybrid" and normalized_include_map:
        direction_quota = max(1, min(max_candidates, (max_candidates + 1) // 2))
        selected_records: list[dict[str, Any]] = []
        selected_keys: set[tuple[str, ...]] = set()
        for _, _, item in scored:
            candidate = item.get("candidate", {})
            if not isinstance(candidate, dict):
                continue
            matches_direction = all(
                str(candidate.get(name)) in allowed_values
                for name, allowed_values in normalized_include_map.items()
            )
            if not matches_direction:
                continue
            selected_records.append(item)
            selected_keys.add(tuple(str(value) for value in item.get("key", ())))
            if len(selected_records) >= direction_quota:
                break
        for _, _, item in scored:
            key = tuple(str(value) for value in item.get("key", ()))
            if key in selected_keys:
                continue
            selected_records.append(item)
            selected_keys.add(key)
            if len(selected_records) >= max_candidates:
                break
        selected = selected_records
    else:
        selected = [item for _, _, item in scored[:max_candidates]]
    return selected, {
        "enabled": True,
        "reason": None,
        "max_candidates": max_candidates,
        "top_history": int(top_history or 0),
        "include_map": {
            key: sorted(values)
            for key, values in normalized_include_map.items()
        },
        "include_mode": normalized_include_mode,
        "support_history_count": len(support_rows),
        "scored_candidate_count": len(scored),
        "selected_candidate_count": len(selected),
    }


def build_local_calibration_candidates(
    *,
    candidate_records: list[dict[str, Any]],
    history: list[dict[str, Any]],
    shortlist_candidates: list[dict[str, Any]],
    feature_columns: list[str],
    allowed_keys: set[tuple[str, ...]] | None = None,
    goal: str = "maximize",
    max_candidates: int = 4,
    top_history: int = 6,
    min_anchor_matches: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Surface near-neighbor candidates around high-value observed anchors.

    The scoring only uses observed high-value anchors. It preserves most axes
    from a strong observed condition and varies one remaining axis, which makes
    it useful as a bounded early local-calibration probe.
    """

    feature_columns = [str(name) for name in feature_columns if str(name).strip()]
    if not candidate_records or not feature_columns:
        return [], {"enabled": False, "reason": "missing_candidate_records_or_features"}

    max_candidates = max(0, int(max_candidates or 0))
    if max_candidates <= 0:
        return [], {"enabled": False, "reason": "max_candidates_zero"}
    min_anchor_matches = max(1, min(len(feature_columns), int(min_anchor_matches or 1)))

    support_rows = _ranked_history(history, goal=goal)[: max(1, int(top_history or 1))]
    if not support_rows:
        return [], {"enabled": False, "reason": "missing_observed_support"}

    observed_keys = {
        _candidate_key(row["candidate"], feature_columns)
        for row in history
        if isinstance(row.get("candidate"), dict)
    }
    shortlist_keys = {
        _candidate_key(item["candidate"], feature_columns)
        for item in shortlist_candidates
        if isinstance(item.get("candidate"), dict)
    }
    legal_keys = allowed_keys if allowed_keys is not None else {
        tuple(str(value) for value in item.get("key", ()))
        for item in candidate_records
        if item.get("key")
    }

    numeric_support = [
        float(value)
        for value in (_float_or_none(row.get("result")) for row in support_rows)
        if value is not None
    ]
    low = min(numeric_support) if numeric_support else 0.0
    high = max(numeric_support) if numeric_support else 1.0
    span = max(1e-12, high - low)

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for order, record in enumerate(candidate_records):
        candidate = record.get("candidate")
        key = tuple(str(value) for value in record.get("key", ()))
        if not isinstance(candidate, dict) or not key:
            continue
        if key not in legal_keys or key in observed_keys or key in shortlist_keys:
            continue

        best_anchor: dict[str, Any] | None = None
        best_score: tuple[float, float, float] | None = None
        for rank, row in enumerate(support_rows):
            anchor = row.get("candidate")
            anchor_value = _float_or_none(row.get("result"))
            if not isinstance(anchor, dict) or anchor_value is None:
                continue
            match_count = sum(
                1
                for name in feature_columns
                if str(candidate.get(name)) == str(anchor.get(name))
            )
            if match_count < min_anchor_matches or match_count >= len(feature_columns):
                continue
            if str(goal).strip().lower() == "minimize":
                value_score = (high - float(anchor_value)) / span
            else:
                value_score = (float(anchor_value) - low) / span
            match_fraction = match_count / max(1, len(feature_columns))
            rank_score = float(len(support_rows) - rank) / max(1, len(support_rows))
            score_tuple = (float(value_score), float(match_fraction), float(rank_score))
            if best_score is None or score_tuple > best_score:
                best_score = score_tuple
                best_anchor = row
        if best_score is None or best_anchor is None:
            continue

        value_score, match_fraction, rank_score = best_score
        score = 0.55 * value_score + 0.35 * match_fraction + 0.10 * rank_score
        anchor_candidate = best_anchor["candidate"]
        dim_support = {
            name: 1.0
            for name in feature_columns
            if str(candidate.get(name)) == str(anchor_candidate.get(name))
        }
        scored.append(
            (
                score,
                order,
                {
                    **record,
                    "probe_score": round(float(score), 6),
                    "probe_evidence": {
                        "dim_support": dim_support,
                        "pair_score": 0.0,
                        "best_match_fraction": round(float(match_fraction), 4),
                        "scaffold_score": 0.0,
                        "analogue_value_score": round(float(value_score * match_fraction), 4),
                        "best_analogue_match_fraction": round(float(match_fraction), 4),
                        "best_analogue_result": _float_or_none(best_anchor.get("result")),
                        "direction_constraints": {},
                        "direction_include_mode": "local_calibration",
                    },
                    "candidate_tool": "local_calibration",
                },
            )
        )

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [item for _, _, item in scored[:max_candidates]]
    return selected, {
        "enabled": True,
        "reason": None,
        "tool": "local_calibration",
        "max_candidates": max_candidates,
        "top_history": int(top_history or 0),
        "support_history_count": len(support_rows),
        "scored_candidate_count": len(scored),
        "selected_candidate_count": len(selected),
        "min_anchor_matches": min_anchor_matches,
    }


def build_axis_companion_candidates(
    *,
    candidate_records: list[dict[str, Any]],
    history: list[dict[str, Any]],
    shortlist_candidates: list[dict[str, Any]],
    feature_columns: list[str],
    allowed_keys: set[tuple[str, ...]] | None = None,
    goal: str = "maximize",
    max_candidates: int = 4,
    top_history: int = 6,
    axis_name: str = "Base",
    prefer_patterns: list[str] | tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Surface one-axis companions around high-value observed anchors.

    This is stricter than local calibration: a candidate must match a strong
    observed anchor on every finite-pool dimension except ``axis_name``. It is
    meant for cases where the run has already found a promising route and only
    needs a bounded check of one chemically meaningful axis, such as Base.
    """

    feature_columns = [str(name) for name in feature_columns if str(name).strip()]
    axis_name = str(axis_name or "").strip()
    if not candidate_records or not feature_columns or axis_name not in feature_columns:
        return [], {"enabled": False, "reason": "missing_candidate_records_or_axis"}

    max_candidates = max(0, int(max_candidates or 0))
    if max_candidates <= 0:
        return [], {"enabled": False, "reason": "max_candidates_zero"}

    support_rows = _ranked_history(history, goal=goal)[: max(1, int(top_history or 1))]
    if not support_rows:
        return [], {"enabled": False, "reason": "missing_observed_support"}

    observed_keys = {
        _candidate_key(row["candidate"], feature_columns)
        for row in history
        if isinstance(row.get("candidate"), dict)
    }
    shortlist_keys = {
        _candidate_key(item["candidate"], feature_columns)
        for item in shortlist_candidates
        if isinstance(item.get("candidate"), dict)
    }
    legal_keys = allowed_keys if allowed_keys is not None else {
        tuple(str(value) for value in item.get("key", ()))
        for item in candidate_records
        if item.get("key")
    }

    numeric_support = [
        float(value)
        for value in (_float_or_none(row.get("result")) for row in support_rows)
        if value is not None
    ]
    low = min(numeric_support) if numeric_support else 0.0
    high = max(numeric_support) if numeric_support else 1.0
    span = max(1e-12, high - low)
    prefer_patterns = tuple(str(pattern) for pattern in prefer_patterns if str(pattern).strip())

    companion_dims = [name for name in feature_columns if name != axis_name]
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for order, record in enumerate(candidate_records):
        candidate = record.get("candidate")
        key = tuple(str(value) for value in record.get("key", ()))
        if not isinstance(candidate, dict) or not key:
            continue
        if key not in legal_keys or key in observed_keys or key in shortlist_keys:
            continue

        best_anchor: dict[str, Any] | None = None
        best_score: tuple[float, float, float] | None = None
        candidate_axis_value = str(candidate.get(axis_name))
        for rank, row in enumerate(support_rows):
            anchor = row.get("candidate")
            anchor_value = _float_or_none(row.get("result"))
            if not isinstance(anchor, dict) or anchor_value is None:
                continue
            if candidate_axis_value == str(anchor.get(axis_name)):
                continue
            if any(str(candidate.get(name)) != str(anchor.get(name)) for name in companion_dims):
                continue
            if str(goal).strip().lower() == "minimize":
                value_score = (high - float(anchor_value)) / span
            else:
                value_score = (float(anchor_value) - low) / span
            rank_score = float(len(support_rows) - rank) / max(1, len(support_rows))
            axis_prefer_score = 1.0 if any(pattern in candidate_axis_value for pattern in prefer_patterns) else 0.0
            score_tuple = (float(axis_prefer_score), float(value_score), float(rank_score))
            if best_score is None or score_tuple > best_score:
                best_score = score_tuple
                best_anchor = row
        if best_score is None or best_anchor is None:
            continue

        axis_prefer_score, value_score, rank_score = best_score
        score = 0.52 * axis_prefer_score + 0.36 * value_score + 0.12 * rank_score
        anchor_candidate = best_anchor["candidate"]
        dim_support = {
            name: 1.0
            for name in companion_dims
            if str(candidate.get(name)) == str(anchor_candidate.get(name))
        }
        scored.append(
            (
                score,
                order,
                {
                    **record,
                    "probe_score": round(float(score), 6),
                    "probe_evidence": {
                        "dim_support": dim_support,
                        "pair_score": 0.0,
                        "best_match_fraction": round(float(len(companion_dims) / max(1, len(feature_columns))), 4),
                        "scaffold_score": 0.0,
                        "analogue_value_score": round(float(value_score), 4),
                        "best_analogue_match_fraction": round(
                            float(len(companion_dims) / max(1, len(feature_columns))),
                            4,
                        ),
                        "best_analogue_result": _float_or_none(best_anchor.get("result")),
                        "direction_constraints": {axis_name: list(prefer_patterns)},
                        "direction_include_mode": "axis_companion",
                        "axis_name": axis_name,
                        "axis_prefer_score": round(float(axis_prefer_score), 4),
                    },
                    "candidate_tool": "axis_companion",
                },
            )
        )

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [item for _, _, item in scored[:max_candidates]]
    return selected, {
        "enabled": True,
        "reason": None,
        "tool": "axis_companion",
        "axis_name": axis_name,
        "prefer_patterns": list(prefer_patterns),
        "max_candidates": max_candidates,
        "top_history": int(top_history or 0),
        "support_history_count": len(support_rows),
        "scored_candidate_count": len(scored),
        "selected_candidate_count": len(selected),
    }


def build_suzuki_local_calibration_candidates(
    *,
    candidate_records: list[dict[str, Any]],
    history: list[dict[str, Any]],
    shortlist_candidates: list[dict[str, Any]],
    feature_columns: list[str],
    allowed_keys: set[tuple[str, ...]] | None = None,
    goal: str = "maximize",
    max_candidates: int = 4,
    top_history: int = 6,
    anchor_threshold: float = 0.0,
    prefer_ligands: tuple[str, ...] = (),
    prefer_bases: tuple[str, ...] = (),
    prefer_solvents: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build route-preserving Suzuki local-calibration candidates.

    The tool anchors on the best observed Suzuki route, keeps
    electrophile/nucleophile/catalyst fixed, and changes exactly one of
    ligand/base/solvent. Ranking is label-free for unobserved candidates.
    """

    feature_columns = [str(name) for name in feature_columns if str(name).strip()]
    required = {"electrophile", "nucleophile", "catalyst", "ligand", "base", "solvent"}
    if not required.issubset(set(feature_columns)):
        return [], {
            "enabled": False,
            "reason": "suzuki_feature_columns_not_available",
            "tool": "suzuki_local_calibration",
        }
    max_candidates = max(0, int(max_candidates or 0))
    if max_candidates <= 0:
        return [], {
            "enabled": False,
            "reason": "max_candidates_zero",
            "tool": "suzuki_local_calibration",
        }
    ranked = _ranked_history(history, goal=goal)
    if not ranked:
        return [], {
            "enabled": False,
            "reason": "missing_observed_support",
            "tool": "suzuki_local_calibration",
        }
    anchor_row = ranked[0]
    anchor = anchor_row.get("candidate")
    anchor_value = _float_or_none(anchor_row.get("result"))
    if not isinstance(anchor, dict) or anchor_value is None:
        return [], {
            "enabled": False,
            "reason": "missing_anchor",
            "tool": "suzuki_local_calibration",
        }
    if float(anchor_threshold or 0.0) > 0.0 and float(anchor_value) < float(anchor_threshold):
        return [], {
            "enabled": False,
            "reason": "anchor_below_threshold",
            "tool": "suzuki_local_calibration",
            "anchor_result": float(anchor_value),
            "anchor_threshold": float(anchor_threshold),
        }

    observed_keys = {
        _candidate_key(row["candidate"], feature_columns)
        for row in history
        if isinstance(row.get("candidate"), dict)
    }
    shortlist_keys = {
        _candidate_key(item["candidate"], feature_columns)
        for item in shortlist_candidates
        if isinstance(item.get("candidate"), dict)
    }
    legal_keys = allowed_keys if allowed_keys is not None else {
        tuple(str(v) for v in item.get("key", ()))
        for item in candidate_records
        if item.get("key")
    }

    support_rows = ranked[: max(1, int(top_history or 1))]
    support_values: dict[str, set[str]] = {name: set() for name in feature_columns}
    for row in support_rows:
        candidate = row.get("candidate")
        if not isinstance(candidate, dict):
            continue
        for name in feature_columns:
            support_values[name].add(str(candidate.get(name)))

    preference_map = {
        "ligand": {str(value): idx for idx, value in enumerate(prefer_ligands)},
        "base": {str(value): idx for idx, value in enumerate(prefer_bases)},
        "solvent": {str(value): idx for idx, value in enumerate(prefer_solvents)},
    }
    local_axes = ("ligand", "base", "solvent")
    route_axes = ("electrophile", "nucleophile", "catalyst")
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for order, record in enumerate(candidate_records):
        candidate = record.get("candidate")
        key = tuple(str(value) for value in record.get("key", ()))
        if not isinstance(candidate, dict) or not key:
            continue
        if key not in legal_keys or key in observed_keys or key in shortlist_keys:
            continue
        if any(str(candidate.get(axis)) != str(anchor.get(axis)) for axis in route_axes):
            continue
        changed_local_axes = [
            axis for axis in local_axes if str(candidate.get(axis)) != str(anchor.get(axis))
        ]
        if len(changed_local_axes) != 1:
            continue
        changed_axis = changed_local_axes[0]
        pref_scores = []
        pref_hit_count = 0
        for axis in local_axes:
            value = str(candidate.get(axis))
            axis_prefs = preference_map.get(axis, {})
            if value in axis_prefs:
                pref_hit_count += 1
                pref_scores.append(1.0 - 0.05 * float(axis_prefs[value]))
            elif value in support_values.get(axis, set()):
                pref_scores.append(0.35)
            else:
                pref_scores.append(0.0)
        changed_value = str(candidate.get(changed_axis))
        if changed_value in preference_map.get(changed_axis, {}):
            changed_axis_pref = 1.0
        elif changed_value in support_values.get(changed_axis, set()):
            changed_axis_pref = 0.35
        else:
            changed_axis_pref = 0.0
        support_fraction = sum(
            1 for name in feature_columns if str(candidate.get(name)) in support_values.get(name, set())
        ) / max(1, len(feature_columns))
        anchor_changed_value = str(anchor.get(changed_axis))
        anchor_changed_supported = (
            anchor_changed_value in preference_map.get(changed_axis, {})
            or anchor_changed_value in support_values.get(changed_axis, set())
        )
        replacement_supported = changed_axis_pref > 0.0
        support_penalty = -0.25 if anchor_changed_supported and not replacement_supported else 0.0
        score = (
            0.50 * (sum(pref_scores) / max(1, len(pref_scores)))
            + 0.30 * changed_axis_pref
            + 0.20 * support_fraction
            + support_penalty
        )
        scored.append(
            (
                score,
                order,
                {
                    **record,
                    "probe_score": round(float(score), 6),
                    "probe_evidence": {
                        "tool": "suzuki_local_calibration",
                        "anchor_result": float(anchor_value),
                        "anchor_candidate": dict(anchor),
                        "route_axes_fixed": list(route_axes),
                        "changed_axis": changed_axis,
                        "preference_hit_count": pref_hit_count,
                        "changed_axis_pref": round(float(changed_axis_pref), 4),
                        "support_fraction": round(float(support_fraction), 4),
                        "axis_preferences": {
                            axis: list(preference_map.get(axis, {}).keys())
                            for axis in local_axes
                        },
                    },
                    "candidate_tool": "suzuki_local_calibration",
                },
            )
        )

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [item for _, _, item in scored[:max_candidates]]
    for rank, item in enumerate(selected, start=1):
        item["candidate_probe_rank"] = rank
    return selected, {
        "enabled": True,
        "reason": None if selected else "no_suzuki_local_calibration_candidate",
        "tool": "suzuki_local_calibration",
        "anchor_result": float(anchor_value),
        "anchor_candidate": dict(anchor),
        "anchor_threshold": float(anchor_threshold or 0.0),
        "max_candidates": max_candidates,
        "top_history": int(top_history or 0),
        "support_history_count": len(support_rows),
        "scored_candidate_count": len(scored),
        "selected_candidate_count": len(selected),
        "axis_preferences": {
            "ligand": list(prefer_ligands),
            "base": list(prefer_bases),
            "solvent": list(prefer_solvents),
        },
    }


def merge_probe_candidates_into_shortlist(
    *,
    shortlist_candidates: list[dict[str, Any]],
    probe_candidates: list[dict[str, Any]],
    history: list[dict[str, Any]],
    feature_columns: list[str],
    scaffold_dims: list[str],
    max_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Append probe candidates after planner candidates and reindex the shortlist."""

    max_size = max(1, int(max_size or len(shortlist_candidates) or 1))
    existing = {
        _candidate_key(item["candidate"], feature_columns)
        for item in shortlist_candidates
        if isinstance(item.get("candidate"), dict)
    }
    merged = [dict(item) for item in shortlist_candidates]
    added = 0
    for probe_rank, item in enumerate(probe_candidates):
        candidate = item.get("candidate")
        if not isinstance(candidate, dict):
            continue
        key = _candidate_key(candidate, feature_columns)
        if key in existing:
            continue
        existing.add(key)
        merged.append(
            {
                "candidate": dict(candidate),
                "bo_rank": item.get("bo_rank"),
                "probe_score": item.get("probe_score"),
                "probe_evidence": item.get("probe_evidence"),
                "candidate_id": item.get("candidate_id"),
                "candidate_probe_rank": int(item.get("candidate_probe_rank", probe_rank)),
                "pool_source": "candidate_probe_pool",
                "shortlist_source": "candidate_probe_injected",
            }
        )
        added += 1
        if len(merged) >= max_size:
            break

    recent_scaffold_counts: dict[tuple[str, ...], int] = {}
    for row in history:
        candidate = row.get("candidate")
        if not isinstance(candidate, dict):
            continue
        key = tuple(str(candidate.get(name)) for name in scaffold_dims)
        recent_scaffold_counts[key] = recent_scaffold_counts.get(key, 0) + 1

    for idx, item in enumerate(merged):
        candidate = item.get("candidate", {})
        scaffold_key = tuple(str(candidate.get(name)) for name in scaffold_dims)
        item["candidate_index"] = idx
        item.setdefault("scaffold_key", list(scaffold_key))
        item.setdefault("recent_scaffold_hits", int(recent_scaffold_counts.get(scaffold_key, 0)))
        item.setdefault("main_pool_rank", None)
        item.setdefault("diversity_pool_rank", None)
        item["is_main_bo_top1"] = bool(idx == 0 and item.get("pool_source") == "main_pool")

    return merged, {
        "added_candidate_count": added,
        "post_merge_shortlist_size": len(merged),
        "max_size": max_size,
    }
