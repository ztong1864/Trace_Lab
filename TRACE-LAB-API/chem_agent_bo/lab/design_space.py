"""Design-space parsing for real-lab TRACE projects."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from olympus.campaigns import ParameterSpace
from olympus.objects import ParameterContinuous

try:
    from olympus.objects import ParameterCategorical
except ImportError:  # pragma: no cover
    from olympus.objects import Parameter as _OlympusParameter

    def ParameterCategorical(*, name: str, options: list[str]):  # type: ignore[misc]
        return _OlympusParameter(kind="categorical", name=name, options=options)


DESCRIPTOR_PREFIX = "descriptor__"
CONTROLLED_CONDITION_ROLES = {
    "controlled",
    "controlled_condition",
    "stage_control",
    "held_condition",
}
NORMALIZED_COLUMNS = (
    "variable",
    "type",
    "value",
    "unit",
    "low",
    "high",
    "step",
    "role",
    "stage",
    "active",
    "is_fixed",
    "fixed_value",
    "descriptor_group",
    "constraint_group",
    "source",
)


@dataclass
class DesignOption:
    value: str
    unit: str = ""
    descriptors: dict[str, Any] = field(default_factory=dict)
    source: str = ""


@dataclass
class DesignVariable:
    name: str
    kind: str = "categorical"
    options: list[DesignOption] = field(default_factory=list)
    unit: str = ""
    low: float | None = None
    high: float | None = None
    step: float | None = None
    role: str = ""
    stage: str = "A"
    active: bool = True
    is_fixed: bool = False
    fixed_value: str = ""
    descriptor_group: str = ""
    constraint_group: str = ""

    def option_values(self) -> list[str]:
        return [str(option.value) for option in self.options]

    def option_display_values(self) -> list[str]:
        return [
            _format_condition_value(option.value, option.unit or self.unit)
            for option in self.options
        ]

    def default_condition_value(self) -> str:
        if self.fixed_value:
            return _format_condition_value(self.fixed_value, self.unit)
        if self.options:
            option = self.options[0]
            return _format_condition_value(option.value, option.unit or self.unit)
        if self.kind == "continuous" and self.low is not None:
            return _format_condition_value(str(self.low), self.unit)
        return ""

    @property
    def optimization_active(self) -> bool:
        return bool(self.active) and not self.is_fixed and self.kind != "fixed"

    @property
    def controlled_condition(self) -> bool:
        return (
            not self.optimization_active
            and not self.is_fixed
            and self.kind != "fixed"
            and _role_key(self.role) in CONTROLLED_CONDITION_ROLES
        )


class DesignSpace:
    """A lab design space with descriptors and an Olympus parameter space."""

    def __init__(self, variables: list[DesignVariable]) -> None:
        if not variables:
            raise ValueError("DesignSpace requires at least one variable.")
        self.variables = variables
        self.variable_names = [
            variable.name for variable in variables if variable.optimization_active
        ]
        self.condition_names = [
            variable.name
            for variable in variables
            if (
                variable.optimization_active
                or variable.is_fixed
                or variable.kind == "fixed"
                or variable.controlled_condition
            )
        ]

    @classmethod
    def from_long_records(cls, records: Iterable[dict[str, Any]]) -> "DesignSpace":
        grouped: dict[str, DesignVariable] = {}
        for raw in records:
            row = {str(key).strip(): value for key, value in raw.items()}
            variable_name = _clean(row.get("variable") or row.get("Variable"))
            if not variable_name:
                continue
            kind = _normalize_kind(row.get("type") or row.get("kind") or "categorical")
            is_fixed = _parse_bool(row.get("is_fixed"), default=(kind == "fixed"))
            if is_fixed:
                kind = "fixed"
            active = _parse_bool(row.get("active"), default=True) and kind != "fixed"
            fixed_value = _clean(row.get("fixed_value") or row.get("fixed") or "")
            variable = grouped.get(variable_name)
            if variable is None:
                variable = DesignVariable(
                    name=variable_name,
                    kind=kind or "categorical",
                    unit=_clean(row.get("unit")),
                    low=_parse_float(row.get("low")),
                    high=_parse_float(row.get("high")),
                    step=_parse_float(row.get("step") or row.get("stride")),
                    role=_clean(row.get("role")),
                    stage=_clean(row.get("stage")) or "A",
                    active=active,
                    is_fixed=is_fixed,
                    fixed_value=fixed_value,
                    descriptor_group=_clean(row.get("descriptor_group")),
                    constraint_group=_clean(row.get("constraint_group")),
                )
                grouped[variable_name] = variable
            elif is_fixed and not variable.is_fixed:
                variable.kind = "fixed"
                variable.active = False
                variable.is_fixed = True
            if fixed_value and not variable.fixed_value:
                variable.fixed_value = fixed_value
            value = _clean(row.get("value") or row.get("option") or row.get("name"))
            if variable.kind == "fixed" and not variable.fixed_value and value:
                variable.fixed_value = value
            descriptors = {
                key[len(DESCRIPTOR_PREFIX) :]: value
                for key, value in row.items()
                if key.startswith(DESCRIPTOR_PREFIX) and _clean(value) != ""
            }
            for key, value_item in row.items():
                if key in NORMALIZED_COLUMNS or key.startswith(DESCRIPTOR_PREFIX):
                    continue
                if _clean(value_item) != "":
                    descriptors[key] = value_item
            if value:
                if value not in {item.value for item in variable.options}:
                    variable.options.append(
                        DesignOption(
                            value=value,
                            unit=_clean(row.get("unit")) or variable.unit,
                            descriptors=descriptors,
                            source=_clean(row.get("source")),
                        )
                    )
            if variable.kind == "continuous":
                variable.low = variable.low if variable.low is not None else _parse_float(row.get("low"))
                variable.high = variable.high if variable.high is not None else _parse_float(row.get("high"))
                variable.step = variable.step if variable.step is not None else _parse_float(row.get("step") or row.get("stride"))
        variables = list(grouped.values())
        _apply_collaborator_stage_defaults(variables)
        for variable in variables:
            if variable.kind == "fixed":
                if not variable.fixed_value:
                    if variable.options:
                        variable.fixed_value = variable.options[0].value
                    else:
                        raise ValueError(f"Fixed variable `{variable.name}` requires fixed_value.")
                variable.active = False
                variable.is_fixed = True
                if not variable.options:
                    variable.options.append(
                        DesignOption(
                            value=variable.fixed_value,
                            unit=variable.unit,
                            source="fixed_condition",
                        )
                    )
                continue
            if variable.kind != "continuous" and not variable.options:
                raise ValueError(f"Variable `{variable.name}` has no options.")
            if variable.kind == "continuous" and (variable.low is None or variable.high is None):
                raise ValueError(f"Continuous variable `{variable.name}` requires low/high.")
        return cls(variables)

    @classmethod
    def read_csv(cls, path: str | Path) -> "DesignSpace":
        rows = _read_csv_rows(path)
        return cls.from_long_records(rows)

    @classmethod
    def from_collaborator_csvs(
        cls,
        *,
        additive_csv: str | Path,
        solvent_csv: str | Path,
        tempo_csv: str | Path,
        extra_options: dict[str, list[str]] | None = None,
        include_stage_a_defaults: bool = True,
    ) -> "DesignSpace":
        records: list[dict[str, Any]] = []
        records.extend(
            _records_from_descriptor_csv(
                additive_csv,
                variable="Additive",
                value_column="Formula",
                source="collaborator:additive_desc",
            )
        )
        records.extend(
            _records_from_descriptor_csv(
                solvent_csv,
                variable="Solvent",
                value_column="solvent",
                source="collaborator:solvent_desc",
            )
        )
        tempo_rows = _read_csv_rows(tempo_csv)
        tempo_value_col = _first_non_descriptor_column(tempo_rows, preferred=("", "tempo", "TEMPO"))
        records.extend(
            _records_from_descriptor_rows(
                tempo_rows,
                variable="TEMPO derivative",
                value_column=tempo_value_col,
                source="collaborator:tempo_desc",
                fallback_value_prefix="TEMPO",
            )
        )
        stage_a_defaults = {
            "M(NO3)3": {
                "type": "fixed",
                "values": ["Fe(NO3)3.9H2O"],
                "stage": "A",
                "active": False,
                "role": "fixed_condition",
            },
            "Component ratio": {
                "type": "categorical",
                "values": ["5/5/10", "6/5/10"],
                "stage": "A",
                "active": False,
                "role": "controlled_condition",
                "fixed_value": "6/5/10",
            },
            "Solvent volume": {
                "type": "discrete_numeric",
                "values": ["3", "4"],
                "unit": "mL",
                "stage": "A",
                "active": False,
                "role": "controlled_condition",
                "fixed_value": "3",
            },
            "Temperature": {
                "type": "discrete_numeric",
                "values": ["25", "50"],
                "unit": "C",
                "stage": "A",
                "active": False,
                "role": "controlled_condition",
                "fixed_value": "50",
            },
        }
        merged_extra: dict[str, Any] = {}
        if include_stage_a_defaults:
            merged_extra.update(stage_a_defaults)
        if extra_options:
            merged_extra.update(extra_options)
        for variable, spec in merged_extra.items():
            if isinstance(spec, dict):
                values = list(spec.get("values") or [])
                kind = str(spec.get("type") or "categorical")
                stage = str(spec.get("stage") or "A")
                active = bool(spec.get("active", True))
                role = str(spec.get("role") or "")
                unit = str(spec.get("unit") or "")
                fixed_value = str(
                    spec.get("fixed_value") or spec.get("default") or ""
                )
            else:
                values = list(spec)
                kind = "categorical"
                stage = "A"
                active = True
                role = ""
                unit = ""
                fixed_value = ""
            for value in values:
                records.append(
                    {
                        "variable": variable,
                        "type": kind,
                        "value": str(value),
                        "unit": unit,
                        "stage": stage,
                        "active": str(active).lower(),
                        "is_fixed": str(kind == "fixed").lower(),
                        "fixed_value": str(value) if kind == "fixed" else fixed_value,
                        "role": role,
                        "source": "stage_a_template",
                    }
                )
        return cls.from_long_records(records)

    def to_long_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for variable in self.variables:
            if variable.kind == "continuous":
                records.append(
                    {
                        "variable": variable.name,
                        "type": variable.kind,
                        "value": "",
                        "unit": variable.unit,
                        "low": "" if variable.low is None else variable.low,
                        "high": "" if variable.high is None else variable.high,
                        "step": "" if variable.step is None else variable.step,
                        "role": variable.role,
                        "stage": variable.stage,
                        "active": str(variable.active).lower(),
                        "is_fixed": str(variable.is_fixed).lower(),
                        "fixed_value": variable.fixed_value,
                        "descriptor_group": variable.descriptor_group,
                        "constraint_group": variable.constraint_group,
                        "source": "",
                    }
                )
                continue
            for option in variable.options:
                row: dict[str, Any] = {
                    "variable": variable.name,
                    "type": variable.kind,
                    "value": option.value,
                    "unit": option.unit or variable.unit,
                    "low": "" if variable.low is None else variable.low,
                    "high": "" if variable.high is None else variable.high,
                    "step": "" if variable.step is None else variable.step,
                    "role": variable.role,
                    "stage": variable.stage,
                    "active": str(variable.active).lower(),
                    "is_fixed": str(variable.is_fixed).lower(),
                    "fixed_value": variable.fixed_value,
                    "descriptor_group": variable.descriptor_group,
                    "constraint_group": variable.constraint_group,
                    "source": option.source,
                }
                for key, value in sorted(option.descriptors.items()):
                    row[f"{DESCRIPTOR_PREFIX}{key}"] = value
                records.append(row)
        return records

    def write_csv(self, path: str | Path) -> None:
        records = self.to_long_records()
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = list(NORMALIZED_COLUMNS)
        descriptor_fields = sorted(
            {
                key
                for record in records
                for key in record
                if key not in fieldnames
            }
        )
        fieldnames.extend(descriptor_fields)
        with path_obj.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    def param_space(
        self,
        *,
        include_descriptors: bool = False,
        max_descriptor_count: int | None = None,
    ) -> ParameterSpace:
        space = ParameterSpace()
        for variable in self.variables:
            if not variable.optimization_active:
                continue
            if variable.kind == "continuous":
                space.add(
                    ParameterContinuous(
                        name=variable.name,
                        low=float(variable.low),
                        high=float(variable.high),
                    )
                )
            else:
                # Treat finite numeric grids as categorical for the first lab
                # release. This keeps Atlas/Random behavior stable and avoids
                # implying continuous interpolation where chemists supplied a
                # fixed set of executable conditions.
                kwargs: dict[str, Any] = {
                    "name": variable.name,
                    "options": variable.option_values(),
                }
                if include_descriptors:
                    matrix = self.numeric_descriptor_matrix(
                        variable.name,
                        max_descriptor_count=max_descriptor_count,
                    )
                    if matrix["descriptor_keys"]:
                        kwargs["descriptors"] = matrix["matrix"]
                space.add(ParameterCategorical(**kwargs))
        return space

    def numeric_descriptor_matrix(
        self,
        variable_name: str,
        *,
        max_descriptor_count: int | None = None,
    ) -> dict[str, Any]:
        variable = self._variable(variable_name)
        keys = _complete_numeric_descriptor_keys(variable)
        if max_descriptor_count is not None:
            keys = keys[: max(0, int(max_descriptor_count))]
        matrix: list[list[float]] = []
        for option in variable.options:
            row: list[float] = []
            for key in keys:
                parsed = _parse_descriptor_float(option.descriptors.get(key))
                row.append(0.0 if parsed is None else parsed)
            matrix.append(row)
        return {
            "variable": variable.name,
            "option_values": variable.option_values(),
            "descriptor_keys": keys,
            "matrix": matrix,
            "shape": [len(matrix), len(keys)],
            "source_count": len({option.source for option in variable.options if option.source}),
        }

    def descriptor_matrices(
        self,
        *,
        variable_names: list[str] | None = None,
        max_descriptor_count: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        selected = set(variable_names or self.variable_names)
        return {
            variable.name: self.numeric_descriptor_matrix(
                variable.name,
                max_descriptor_count=max_descriptor_count,
            )
            for variable in self.variables
            if variable.name in selected and variable.kind != "continuous"
        }

    def descriptor_overview(
        self,
        *,
        variable_names: list[str] | None = None,
        max_descriptor_names: int = 12,
    ) -> dict[str, Any]:
        selected = set(variable_names or self.variable_names)
        variables: dict[str, Any] = {}
        for variable in self.variables:
            if variable.name not in selected or variable.kind == "continuous":
                continue
            stats = _descriptor_key_stats(variable)
            complete_numeric = [
                key
                for key, item in stats.items()
                if item["numeric_count"] == len(variable.options) and len(variable.options) > 0
            ]
            any_numeric = [
                key for key, item in stats.items() if item["numeric_count"] > 0
            ]
            variables[variable.name] = {
                "option_count": len(variable.options),
                "raw_descriptor_count": len(stats),
                "numeric_descriptor_count": len(any_numeric),
                "complete_numeric_descriptor_count": len(complete_numeric),
                "sample_descriptor_names": complete_numeric[: max(0, int(max_descriptor_names))],
                "matrix_shape": [len(variable.options), len(complete_numeric)],
                "descriptor_sources": sorted(
                    {option.source for option in variable.options if option.source}
                )[:5],
                "matrix_available": bool(complete_numeric),
            }
        return {
            "variables": variables,
            "enabled_variables": [
                name
                for name, item in variables.items()
                if item.get("matrix_available")
            ],
            "mode": "controller_metadata",
        }

    def planner_descriptor_eligibility(
        self,
        *,
        variable_names: list[str] | None = None,
    ) -> dict[str, Any]:
        selected = set(variable_names or self.variable_names)
        missing: list[str] = []
        enabled: list[str] = []
        ignored: list[str] = []
        for variable in self.variables:
            if variable.name not in selected:
                continue
            if variable.kind == "continuous":
                ignored.append(variable.name)
                continue
            if _complete_numeric_descriptor_keys(variable):
                enabled.append(variable.name)
            else:
                missing.append(variable.name)
        return {
            "eligible": bool(enabled) and not missing,
            "enabled_variables": enabled,
            "missing_descriptor_variables": missing,
            "ignored_continuous_variables": ignored,
            "mode": "all_active_categorical_variables_require_complete_numeric_descriptors",
        }

    def candidate_descriptor_profile(
        self,
        candidate: dict[str, Any],
        *,
        variable_names: list[str] | None = None,
        max_descriptors_per_variable: int = 6,
    ) -> dict[str, Any]:
        selected = set(variable_names or self.variable_names)
        profile: dict[str, Any] = {}
        for variable in self.variables:
            if variable.name not in selected or variable.kind == "continuous":
                continue
            option = _find_option(variable, candidate.get(variable.name))
            if option is None:
                continue
            keys = _complete_numeric_descriptor_keys(variable)[
                : max(0, int(max_descriptors_per_variable))
            ]
            descriptors = {
                key: _round_float(_parse_descriptor_float(option.descriptors.get(key)))
                for key in keys
                if _parse_descriptor_float(option.descriptors.get(key)) is not None
            }
            profile[variable.name] = {
                "value": option.value,
                "source": option.source,
                "numeric_descriptor_count": len(_complete_numeric_descriptor_keys(variable)),
                "descriptors": descriptors,
            }
        return profile

    def candidate_descriptor_contrast(
        self,
        candidate: dict[str, Any],
        reference_candidate: dict[str, Any],
        *,
        variable_names: list[str] | None = None,
        max_deltas_per_variable: int = 5,
    ) -> dict[str, Any]:
        selected = set(variable_names or self.variable_names)
        contrast: dict[str, Any] = {}
        for variable in self.variables:
            if variable.name not in selected or variable.kind == "continuous":
                continue
            candidate_value = candidate.get(variable.name)
            reference_value = reference_candidate.get(variable.name)
            if str(candidate_value) == str(reference_value):
                continue
            candidate_option = _find_option(variable, candidate_value)
            reference_option = _find_option(variable, reference_value)
            if candidate_option is None or reference_option is None:
                continue
            deltas: list[tuple[str, float]] = []
            squared = 0.0
            for key in _complete_numeric_descriptor_keys(variable):
                left = _parse_descriptor_float(reference_option.descriptors.get(key))
                right = _parse_descriptor_float(candidate_option.descriptors.get(key))
                if left is None or right is None:
                    continue
                delta = right - left
                squared += delta * delta
                if delta != 0:
                    deltas.append((key, delta))
            deltas.sort(key=lambda item: abs(item[1]), reverse=True)
            contrast[variable.name] = {
                "from": reference_option.value,
                "to": candidate_option.value,
                "shared_numeric_descriptor_count": len(_complete_numeric_descriptor_keys(variable)),
                "l2_distance": _round_float(math.sqrt(squared)),
                "top_descriptor_deltas": {
                    key: _round_float(value)
                    for key, value in deltas[: max(0, int(max_deltas_per_variable))]
                },
            }
        return contrast

    def _variable(self, variable_name: str) -> DesignVariable:
        for variable in self.variables:
            if variable.name == variable_name:
                return variable
        raise KeyError(f"Unknown design-space variable `{variable_name}`.")

    def fixed_conditions(self) -> dict[str, str]:
        return {
            variable.name: variable.default_condition_value()
            for variable in self.variables
            if variable.is_fixed or variable.kind == "fixed"
        }

    def controlled_conditions(self) -> dict[str, str]:
        return {
            variable.name: variable.default_condition_value()
            for variable in self.variables
            if variable.controlled_condition
        }

    def static_conditions(self) -> dict[str, str]:
        return {
            **self.fixed_conditions(),
            **self.controlled_conditions(),
        }

    def with_fixed_conditions(self, candidate: dict[str, Any]) -> dict[str, Any]:
        merged = dict(candidate)
        for name, value in self.static_conditions().items():
            merged.setdefault(name, value)
        return merged

    def describe(self) -> dict[str, Any]:
        variables = []
        for variable in self.variables:
            if variable.kind == "continuous":
                value_summary = _range_label(variable.low, variable.high, variable.unit)
            elif variable.is_fixed or variable.kind == "fixed":
                value_summary = variable.default_condition_value()
            elif variable.controlled_condition:
                value_summary = _controlled_value_summary(variable)
            elif variable.kind == "discrete_numeric":
                value_summary = _option_summary(variable)
            else:
                value_summary = str(len(variable.options))
            option_values = variable.option_display_values()
            variables.append(
                {
                    "name": variable.name,
                    "type": variable.kind,
                    "active": variable.optimization_active,
                    "is_fixed": variable.is_fixed or variable.kind == "fixed",
                    "is_controlled": variable.controlled_condition,
                    "stage": variable.stage,
                    "unit": variable.unit,
                    "option_count": len(variable.options),
                    "values": option_values[:20],
                    "values_truncated": len(option_values) > 20,
                    "low": variable.low,
                    "high": variable.high,
                    "step": variable.step,
                    "role": variable.role,
                    "fixed_value": variable.default_condition_value(),
                    "value_summary": value_summary,
                    "descriptor_count": _descriptor_count(variable),
                    "descriptor_names": _descriptor_names(variable, limit=16),
                }
            )
        active_variables = [
            item for item in variables if item["active"] and not item["is_fixed"]
        ]
        fixed_conditions = [item for item in variables if item["is_fixed"]]
        controlled_conditions = [item for item in variables if item["is_controlled"]]
        inactive_variables = [
            item
            for item in variables
            if not item["active"] and not item["is_fixed"] and not item["is_controlled"]
        ]
        stages = sorted({str(item["stage"] or "") for item in variables if item["stage"]})
        return {
            "variables": variables,
            "active_variables": list(self.variable_names),
            "active_variable_count": len(self.variable_names),
            "condition_variables": list(self.condition_names),
            "fixed_conditions": fixed_conditions,
            "fixed_condition_count": len(fixed_conditions),
            "controlled_conditions": controlled_conditions,
            "controlled_condition_count": len(controlled_conditions),
            "inactive_variables": inactive_variables,
            "inactive_variable_count": len(inactive_variables),
            "stages": stages,
            "stage_summary": {
                stage: [item for item in variables if item["stage"] == stage]
                for stage in stages
            },
        }


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _records_from_descriptor_csv(
    path: str | Path,
    *,
    variable: str,
    value_column: str,
    source: str,
) -> list[dict[str, Any]]:
    return _records_from_descriptor_rows(
        _read_csv_rows(path),
        variable=variable,
        value_column=value_column,
        source=source,
    )


def _records_from_descriptor_rows(
    rows: list[dict[str, Any]],
    *,
    variable: str,
    value_column: str,
    source: str,
    fallback_value_prefix: str = "option",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        value = _clean(row.get(value_column))
        if not value:
            value = f"{fallback_value_prefix}-{index}"
        record: dict[str, Any] = {
            "variable": variable,
            "type": "categorical",
            "value": value,
            "stage": "A",
            "source": source,
        }
        for key, item in row.items():
            if key == value_column:
                continue
            cleaned_key = _clean(key) or "id"
            if _clean(item) != "":
                record[f"{DESCRIPTOR_PREFIX}{cleaned_key}"] = item
        records.append(record)
    return records


def _first_non_descriptor_column(
    rows: list[dict[str, Any]],
    *,
    preferred: tuple[str, ...],
) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    for candidate in preferred:
        if candidate in columns:
            return candidate
    return columns[0]


def _apply_collaborator_stage_defaults(variables: list[DesignVariable]) -> None:
    """Interpret older collaborator templates with the newer staged semantics."""

    by_name = {variable.name: variable for variable in variables}
    metal = by_name.get("M(NO3)3")
    if metal is not None and len(metal.options) == 1 and _has_stage_template_source(metal):
        metal.kind = "fixed"
        metal.active = False
        metal.is_fixed = True
        metal.fixed_value = metal.fixed_value or metal.options[0].value
        metal.role = metal.role or "fixed_condition"
        metal.stage = metal.stage or "A"

    controlled_specs = {
        "Component ratio": ("categorical", "", "6/5/10"),
        "Solvent volume": ("discrete_numeric", "mL", "3"),
        "Temperature": ("discrete_numeric", "C", "50"),
    }
    for name, (kind, unit, default_value) in controlled_specs.items():
        variable = by_name.get(name)
        if variable is None or not _has_stage_template_source(variable):
            continue
        variable.kind = kind
        variable.stage = "A"
        variable.active = False
        variable.role = "controlled_condition"
        if unit and not variable.unit:
            variable.unit = unit
            for option in variable.options:
                option.unit = option.unit or unit
        variable.fixed_value = variable.fixed_value or _first_matching_option(
            variable,
            default_value,
        )


def _has_stage_template_source(variable: DesignVariable) -> bool:
    return any(option.source == "stage_a_template" for option in variable.options)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _role_key(value: Any) -> str:
    return _clean(value).lower().replace("-", "_").replace(" ", "_")


def _normalize_kind(value: Any) -> str:
    text = _clean(value).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "cat": "categorical",
        "category": "categorical",
        "categorical": "categorical",
        "discrete": "discrete_numeric",
        "discrete_numeric": "discrete_numeric",
        "numeric_grid": "discrete_numeric",
        "continuous": "continuous",
        "float": "continuous",
        "range": "continuous",
        "fixed": "fixed",
        "constant": "fixed",
        "fixed_condition": "fixed",
    }
    return aliases.get(text, text or "categorical")


def _parse_bool(value: Any, *, default: bool) -> bool:
    text = _clean(value).lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "y", "active"}


def _parse_float(value: Any) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _range_label(low: float | None, high: float | None, unit: str = "") -> str:
    left = "" if low is None else str(low)
    right = "" if high is None else str(high)
    label = f"{left}-{right}".strip("-")
    return f"{label} {unit}".strip()


def _format_condition_value(value: Any, unit: str = "") -> str:
    text = _clean(value)
    unit_text = _clean(unit)
    if not text or not unit_text:
        return text
    if text.lower().endswith(unit_text.lower()):
        return text
    return f"{text} {unit_text}"


def _option_summary(variable: DesignVariable, *, limit: int = 6) -> str:
    values = variable.option_display_values()
    if not values:
        return ""
    visible = values[:limit]
    suffix = "" if len(values) <= limit else f", ... ({len(values)} levels)"
    return ", ".join(visible) + suffix


def _controlled_value_summary(variable: DesignVariable) -> str:
    default = variable.default_condition_value()
    values = variable.option_display_values()
    if not values:
        return default
    allowed = ", ".join(values[:6])
    if len(values) > 6:
        allowed += f", ... ({len(values)} levels)"
    if default:
        return f"{default} (allowed: {allowed})"
    return allowed


def _first_matching_option(variable: DesignVariable, preferred: str) -> str:
    preferred_clean = _clean(preferred)
    preferred_key = _condition_match_key(preferred_clean)
    if not variable.options:
        return preferred_clean
    for option in variable.options:
        if _clean(option.value) == preferred_clean:
            return option.value
    for option in variable.options:
        if _condition_match_key(option.value) == preferred_key:
            return option.value
    for option in variable.options:
        if _format_condition_value(option.value, option.unit or variable.unit) == preferred_clean:
            return option.value
    return variable.options[0].value


def _condition_match_key(value: Any) -> str:
    text = _clean(value).lower()
    if not text:
        return ""
    first = text.split()[0]
    if _parse_float(first) is not None:
        return first
    return text


def _descriptor_key_stats(variable: DesignVariable) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for option in variable.options:
        for key, value in option.descriptors.items():
            if _clean(value) == "":
                continue
            item = stats.setdefault(key, {"present_count": 0, "numeric_count": 0})
            item["present_count"] += 1
            if _parse_descriptor_float(value) is not None:
                item["numeric_count"] += 1
    return stats


def _complete_numeric_descriptor_keys(variable: DesignVariable) -> list[str]:
    stats = _descriptor_key_stats(variable)
    option_count = len(variable.options)
    return sorted(
        key
        for key, item in stats.items()
        if option_count > 0 and item["numeric_count"] == option_count
    )


def _parse_descriptor_float(value: Any) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _round_float(value: float | None, *, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _find_option(variable: DesignVariable, value: Any) -> DesignOption | None:
    target = _clean(value)
    target_key = _condition_match_key(target)
    for option in variable.options:
        if _clean(option.value) == target:
            return option
    for option in variable.options:
        if _format_condition_value(option.value, option.unit or variable.unit) == target:
            return option
    for option in variable.options:
        if _condition_match_key(option.value) == target_key:
            return option
    return None


def _descriptor_count(variable: DesignVariable) -> int:
    keys = {
        key
        for option in variable.options
        for key, value in option.descriptors.items()
        if _clean(value) != ""
    }
    return len(keys)


def _descriptor_names(variable: DesignVariable, *, limit: int = 16) -> list[str]:
    keys = sorted(
        {
            key
            for option in variable.options
            for key, value in option.descriptors.items()
            if _clean(value) != ""
        }
    )
    return keys[: max(0, int(limit))]
