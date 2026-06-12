"""YAML loader for Agentic BO single-file config."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from chem_agent_bo.config.schema import (
    AgenticBOConfig,
    ExperimentConfig,
    OrchestratorSettings,
    PromptConfig,
    PromptSkillsConfig,
    ReactionOverrideConfig,
    RuntimeConfig,
)


SECTION_TYPES = {
    "runtime": RuntimeConfig,
    "experiment": ExperimentConfig,
    "orchestrator": OrchestratorSettings,
    "prompt": PromptConfig,
}


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "agent_bo.yaml"


def _normalize_bool(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return value


def _coerce_value(value: Any, expected_type: Any) -> Any:
    origin = get_origin(expected_type)
    if expected_type in {Any, object}:
        return value
    if origin is None:
        if expected_type is bool:
            return _normalize_bool(value)
        if is_dataclass(expected_type):
            if not isinstance(value, dict):
                raise TypeError(f"Expected dict for {expected_type.__name__}, got {type(value).__name__}")
            return _build_dataclass(expected_type, value)
        return value
    if origin in {list, tuple}:
        if not isinstance(value, list):
            raise TypeError(f"Expected list, got {type(value).__name__}")
        (inner_type,) = get_args(expected_type)[:1] or (Any,)
        converted = [_coerce_value(item, inner_type) for item in value]
        return tuple(converted) if origin is tuple else converted
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"Expected dict, got {type(value).__name__}")
        key_type, inner_type = get_args(expected_type)[:2] or (Any, Any)
        converted: dict[Any, Any] = {}
        for key, item in value.items():
            converted_key = _coerce_value(key, key_type) if key_type not in {Any, object} else key
            converted[converted_key] = _coerce_value(item, inner_type)
        return converted
    if origin is type(None):
        return None if value is None else value
    if origin is not None and type(None) in get_args(expected_type):
        inner_types = [item for item in get_args(expected_type) if item is not type(None)]
        if value is None or not inner_types:
            return None
        return _coerce_value(value, inner_types[0])
    return value


def _build_dataclass(cls: type[Any], payload: dict[str, Any]) -> Any:
    allowed = {field.name: field for field in fields(cls)}
    type_hints = get_type_hints(cls)
    unknown = set(payload) - set(allowed)
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise KeyError(f"Unknown keys for {cls.__name__}: {unknown_list}")
    kwargs: dict[str, Any] = {}
    for name, field_info in allowed.items():
        if name not in payload:
            continue
        value = payload[name]
        expected_type = type_hints.get(name, field_info.type)
        if cls is PromptConfig and name == "reaction_overrides":
            if not isinstance(value, dict):
                raise TypeError("prompt.reaction_overrides must be a mapping")
            kwargs[name] = {
                str(key): _build_dataclass(ReactionOverrideConfig, item)
                for key, item in value.items()
            }
            continue
        if cls is PromptConfig and name == "skills":
            if not isinstance(value, dict):
                raise TypeError("prompt.skills must be a mapping")
            kwargs[name] = _build_dataclass(PromptSkillsConfig, value)
            continue
        kwargs[name] = _coerce_value(value, expected_type)
    return cls(**kwargs)


def load_agentic_bo_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> AgenticBOConfig:
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("Top-level config must be a mapping")

    config = AgenticBOConfig()
    for section_name, section_type in SECTION_TYPES.items():
        section_payload = raw.get(section_name)
        if section_payload is None:
            continue
        if not isinstance(section_payload, dict):
            raise TypeError(f"Section '{section_name}' must be a mapping")
        setattr(config, section_name, _build_dataclass(section_type, section_payload))

    if overrides:
        _apply_overrides(config, overrides)
    return config


def _apply_overrides(config: AgenticBOConfig, overrides: dict[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target: Any = config
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
