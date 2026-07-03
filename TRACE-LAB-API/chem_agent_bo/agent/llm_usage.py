"""LLM usage and cost accounting helpers."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any


@dataclass(frozen=True)
class LLMPrice:
    input_per_1m: float = 0.0
    output_per_1m: float = 0.0
    cached_input_per_1m: float = 0.0


PRICE_PROFILES: dict[str, LLMPrice] = {
    "openai_gpt54": LLMPrice(input_per_1m=2.50, output_per_1m=15.00, cached_input_per_1m=0.25),
    "openai_gpt54_mini": LLMPrice(input_per_1m=0.75, output_per_1m=4.50, cached_input_per_1m=0.075),
}


def resolve_price(
    *,
    model_name: str,
    pricing_profile: str = "auto",
    input_cost_per_1m: float | None = None,
    output_cost_per_1m: float | None = None,
    cached_input_cost_per_1m: float | None = None,
) -> tuple[str, LLMPrice]:
    """Resolve the price profile used for estimated cost accounting."""

    profile = str(pricing_profile or "auto").strip().lower()
    if profile == "auto":
        profile = "openai_gpt54_mini" if "mini" in str(model_name).lower() else "openai_gpt54"
    if profile == "custom":
        return profile, LLMPrice(
            input_per_1m=float(input_cost_per_1m or 0.0),
            output_per_1m=float(output_cost_per_1m or 0.0),
            cached_input_per_1m=float(cached_input_cost_per_1m or 0.0),
        )
    return profile, PRICE_PROFILES.get(profile, LLMPrice())


def estimate_cost_usd(
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_input_tokens: int | None,
    price: LLMPrice,
) -> float | None:
    if input_tokens is None and output_tokens is None and cached_input_tokens is None:
        return None
    cached = int(cached_input_tokens or 0)
    total_input = int(input_tokens or 0)
    non_cached_input = max(0, total_input - cached)
    output = int(output_tokens or 0)
    return (
        non_cached_input * price.input_per_1m
        + cached * price.cached_input_per_1m
        + output * price.output_per_1m
    ) / 1_000_000.0


def _get_nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_usage_from_result(result: Any) -> tuple[dict[str, int | None], str]:
    """Extract token usage from common LangChain/OpenAI result shapes."""

    usage_candidates: list[dict[str, Any]] = []
    if isinstance(result, dict):
        for key in ("usage_metadata", "token_usage", "usage"):
            value = result.get(key)
            if isinstance(value, dict):
                usage_candidates.append(value)
        messages = result.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                metadata = getattr(message, "usage_metadata", None)
                if isinstance(metadata, dict):
                    usage_candidates.append(metadata)
                response_metadata = getattr(message, "response_metadata", None)
                if isinstance(response_metadata, dict):
                    for key in ("token_usage", "usage", "usage_metadata"):
                        value = response_metadata.get(key)
                        if isinstance(value, dict):
                            usage_candidates.append(value)
    for usage in usage_candidates:
        input_tokens = _int_or_none(
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or usage.get("prompt_token_count")
        )
        output_tokens = _int_or_none(
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or usage.get("completion_token_count")
        )
        total_tokens = _int_or_none(usage.get("total_tokens") or usage.get("total_token_count"))
        cached_input_tokens = _int_or_none(
            usage.get("cached_input_tokens")
            or _get_nested(usage, "input_token_details", "cache_read")
            or _get_nested(usage, "prompt_tokens_details", "cached_tokens")
        )
        if input_tokens is None and total_tokens is not None and output_tokens is not None:
            input_tokens = max(0, total_tokens - output_tokens)
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
        if input_tokens is not None or output_tokens is not None or total_tokens is not None:
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_input_tokens": cached_input_tokens,
                "total_tokens": total_tokens,
            }, "response_metadata"
    return {
        "input_tokens": None,
        "output_tokens": None,
        "cached_input_tokens": None,
        "total_tokens": None,
    }, "unavailable"


def now() -> float:
    return time.time()

