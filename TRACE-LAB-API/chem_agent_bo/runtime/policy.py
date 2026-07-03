"""Action capability gates shared by benchmark and lab runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LAB_ALLOWED_ACTIONS = {
    "direct_bo_pick",
    "shape_only_bo_pick",
    "shortlist_alt_pick",
}

LAB_ORACLE_ONLY_ACTIONS = {
    "finite_pool_candidate_probe",
    "mask_scaffold_corridor_resuggest",
    "mask_dominant_resuggest",
    "mask_low_repeat_resuggest",
}


@dataclass(frozen=True)
class ActionCapabilityResult:
    requested_action: str
    executed_action: str
    fallback_reason: str | None = None
    allowed_actions: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.requested_action != self.executed_action

    def to_trace(self) -> dict[str, Any]:
        return {
            "requested_action": self.requested_action,
            "executed_action": self.executed_action,
            "fallback_reason": self.fallback_reason,
            "allowed_actions": list(self.allowed_actions),
            "changed": self.changed,
        }


class ActionCapabilityPolicy:
    """Resolve whether a requested controller action is executable."""

    def __init__(
        self,
        *,
        mode: str,
        allowed_actions: set[str] | None = None,
        oracle_only_actions: set[str] | None = None,
    ) -> None:
        self.mode = str(mode or "benchmark")
        self.allowed_actions = set(allowed_actions or LAB_ALLOWED_ACTIONS)
        self.oracle_only_actions = set(oracle_only_actions or LAB_ORACLE_ONLY_ACTIONS)

    @classmethod
    def for_lab(cls) -> "ActionCapabilityPolicy":
        return cls(mode="lab", allowed_actions=LAB_ALLOWED_ACTIONS)

    @classmethod
    def for_benchmark(cls) -> "ActionCapabilityPolicy":
        return cls(mode="benchmark", allowed_actions=set())

    def resolve(self, action_package: dict[str, Any] | None) -> ActionCapabilityResult:
        requested = str(
            (action_package or {}).get("requested_execution_action")
            or "direct_bo_pick"
        ).strip() or "direct_bo_pick"
        if self.mode != "lab":
            return ActionCapabilityResult(
                requested_action=requested,
                executed_action=requested,
                allowed_actions=tuple(sorted(self.allowed_actions)),
            )
        if requested in self.allowed_actions:
            return ActionCapabilityResult(
                requested_action=requested,
                executed_action=requested,
                allowed_actions=tuple(sorted(self.allowed_actions)),
            )
        if requested in self.oracle_only_actions:
            return ActionCapabilityResult(
                requested_action=requested,
                executed_action="shape_only_bo_pick",
                fallback_reason=f"{requested}_disabled_in_lab_mode",
                allowed_actions=tuple(sorted(self.allowed_actions)),
            )
        return ActionCapabilityResult(
            requested_action=requested,
            executed_action="direct_bo_pick",
            fallback_reason=f"{requested}_unsupported_in_lab_mode",
            allowed_actions=tuple(sorted(self.allowed_actions)),
        )
