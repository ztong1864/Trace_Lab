"""Lab-mode batch composition for TRACE.

The composer treats a real-lab ask as one portfolio-design decision. It keeps
BO/Atlas as the bounded source of admissible candidates, while asking the
controller to explain how the whole batch should be allocated across
experimental roles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BatchSlot:
    slot_id: int
    role: str
    candidate_index: int
    purpose: str = ""
    varied_variables: list[str] = field(default_factory=list)
    controlled_variables: list[str] = field(default_factory=list)
    rationale: str = ""
    risk_note: str = ""
    evidence_refs: list[str] = field(default_factory=list)


@dataclass
class BatchContract:
    batch_strategy: str = "planner_anchor_diversity"
    batch_rationale: str = ""
    global_constraints: list[str] = field(default_factory=list)
    slots: list[BatchSlot] = field(default_factory=list)


@dataclass
class BatchValidationReport:
    status: str = "valid"
    issues: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    fallback_used: bool = False


@dataclass
class BatchComposition:
    selected_items: list[dict[str, Any]]
    batch_contract: BatchContract
    validation_report: BatchValidationReport
    raw_contract: dict[str, Any] = field(default_factory=dict)


class LabBatchComposer:
    """Compose a whole real-lab batch under controller constraints."""

    def __init__(self, decision_engine) -> None:  # noqa: ANN001
        self.decision_engine = decision_engine

    def compose(
        self,
        *,
        candidate_pool: list[dict[str, Any]],
        batch_size: int,
        decision_context: dict[str, Any],
        controller_plan: dict[str, Any],
        diagnosis: dict[str, Any],
        hypothesis_action: dict[str, Any],
        coverage_insight: dict[str, Any],
        search_space,  # noqa: ANN001
        reaction_context: dict[str, Any],
    ) -> BatchComposition:
        requested = max(1, int(batch_size))
        if not candidate_pool:
            raise RuntimeError("LabBatchComposer requires a non-empty candidate pool.")

        report = BatchValidationReport()
        raw_contract: dict[str, Any] = {}
        try:
            raw_contract = self.decision_engine.compose_lab_batch(
                decision_context=decision_context,
                candidate_pool=candidate_pool,
                controller_plan=controller_plan,
                diagnosis=diagnosis,
                hypothesis_action=hypothesis_action,
                coverage_insight=coverage_insight,
                batch_size=requested,
                search_space=search_space,
                reaction_context=reaction_context,
            )
        except Exception as exc:  # noqa: BLE001
            report.status = "fallback"
            report.fallback_used = True
            report.issues.append(f"batch_composer_llm_error:{type(exc).__name__}: {exc}")
            raw_contract = {}

        contract = _contract_from_raw(raw_contract)
        selected_items = _items_from_contract(
            contract=contract,
            candidate_pool=candidate_pool,
            requested=requested,
            report=report,
        )
        if len(selected_items) < requested:
            report.repairs.append(
                f"filled_{requested - len(selected_items)}_missing_slots_with_fallback"
            )
            selected_items = _fill_with_fallback(
                selected_items=selected_items,
                candidate_pool=candidate_pool,
                requested=requested,
                decision_context=decision_context,
                contract=contract,
            )
        if not selected_items:
            report.status = "fallback"
            report.fallback_used = True
            contract, selected_items = _fallback_contract_and_items(
                candidate_pool=candidate_pool,
                requested=requested,
                decision_context=decision_context,
            )
        _attach_slot_metadata(selected_items, contract)
        if report.issues and report.status == "valid":
            report.status = "repaired"
        return BatchComposition(
            selected_items=selected_items[:requested],
            batch_contract=contract,
            validation_report=report,
            raw_contract=raw_contract,
        )


def _contract_from_raw(raw: dict[str, Any]) -> BatchContract:
    slots: list[BatchSlot] = []
    for idx, item in enumerate(raw.get("slots") or [], start=1):
        if not isinstance(item, dict):
            continue
        try:
            candidate_index = int(item.get("candidate_index", -1))
        except (TypeError, ValueError):
            candidate_index = -1
        slots.append(
            BatchSlot(
                slot_id=int(item.get("slot_id") or idx),
                role=str(item.get("role") or f"slot_{idx}"),
                candidate_index=candidate_index,
                purpose=str(item.get("purpose") or ""),
                varied_variables=[str(value) for value in item.get("varied_variables") or []],
                controlled_variables=[
                    str(value) for value in item.get("controlled_variables") or []
                ],
                rationale=str(item.get("rationale") or ""),
                risk_note=str(item.get("risk_note") or ""),
                evidence_refs=[str(value) for value in item.get("evidence_refs") or []],
            )
        )
    return BatchContract(
        batch_strategy=str(raw.get("batch_strategy") or "planner_anchor_diversity"),
        batch_rationale=str(raw.get("batch_rationale") or ""),
        global_constraints=[str(value) for value in raw.get("global_constraints") or []],
        slots=slots,
    )


def _items_from_contract(
    *,
    contract: BatchContract,
    candidate_pool: list[dict[str, Any]],
    requested: int,
    report: BatchValidationReport,
) -> list[dict[str, Any]]:
    by_index = {
        int(item.get("candidate_index", idx)): item
        for idx, item in enumerate(candidate_pool)
    }
    selected: list[dict[str, Any]] = []
    seen_candidates: set[tuple[tuple[str, str], ...]] = set()
    repaired_slots: list[BatchSlot] = []
    for slot in contract.slots:
        if len(selected) >= requested:
            break
        item = by_index.get(int(slot.candidate_index))
        if item is None:
            report.issues.append(f"invalid_candidate_index:{slot.candidate_index}")
            continue
        signature = _candidate_signature(item.get("candidate") or {})
        if signature in seen_candidates:
            report.issues.append(f"duplicate_candidate_index:{slot.candidate_index}")
            continue
        seen_candidates.add(signature)
        selected.append({**item, **_slot_dict(slot)})
        repaired_slots.append(slot)
    contract.slots = repaired_slots
    return selected


def _fill_with_fallback(
    *,
    selected_items: list[dict[str, Any]],
    candidate_pool: list[dict[str, Any]],
    requested: int,
    decision_context: dict[str, Any],
    contract: BatchContract,
) -> list[dict[str, Any]]:
    selected = list(selected_items)
    seen = {
        _candidate_signature(item.get("candidate") or {})
        for item in selected
    }
    dimensions = _batch_diversity_dimensions(candidate_pool, decision_context)
    if not selected:
        anchor = candidate_pool[0]
        role = _role_for_candidate(anchor, [], dimensions, first=True)
        selected.append({**anchor, **role})
        seen.add(_candidate_signature(anchor.get("candidate") or {}))
        contract.slots.append(_slot_from_item(anchor, role, len(contract.slots) + 1))
    while len(selected) < requested:
        remaining = [
            item
            for item in candidate_pool
            if _candidate_signature(item.get("candidate") or {}) not in seen
        ]
        if not remaining:
            break
        scored = [
            (_diversity_score(item, selected, dimensions), idx, item)
            for idx, item in enumerate(remaining)
        ]
        _, _, item = max(scored, key=lambda entry: entry[0])
        role = _role_for_candidate(item, selected, dimensions, first=False)
        selected.append({**item, **role})
        seen.add(_candidate_signature(item.get("candidate") or {}))
        contract.slots.append(_slot_from_item(item, role, len(contract.slots) + 1))
    return selected


def _fallback_contract_and_items(
    *,
    candidate_pool: list[dict[str, Any]],
    requested: int,
    decision_context: dict[str, Any],
) -> tuple[BatchContract, list[dict[str, Any]]]:
    contract = BatchContract(
        batch_strategy="fallback_planner_anchor_diversity",
        batch_rationale=(
            "Fallback batch composition keeps the planner anchor and fills the "
            "remaining slots with diverse legal candidates."
        ),
        global_constraints=[
            "select only candidates from the planner pool",
            "avoid duplicate candidates",
        ],
    )
    items = _fill_with_fallback(
        selected_items=[],
        candidate_pool=candidate_pool,
        requested=requested,
        decision_context=decision_context,
        contract=contract,
    )
    return contract, items


def _attach_slot_metadata(
    selected_items: list[dict[str, Any]],
    contract: BatchContract,
) -> None:
    slots_by_index = {int(slot.candidate_index): slot for slot in contract.slots}
    for rank, item in enumerate(selected_items, start=1):
        candidate_index = int(item.get("candidate_index", rank - 1) or 0)
        slot = slots_by_index.get(candidate_index)
        if slot is None:
            role = str(item.get("batch_role") or "batch_member")
            slot = BatchSlot(
                slot_id=rank,
                role=role,
                candidate_index=candidate_index,
                purpose=str(item.get("batch_role_reason") or ""),
                rationale=str(item.get("batch_role_reason") or ""),
            )
        item["batch_role"] = slot.role
        item["batch_role_reason"] = slot.purpose or slot.rationale
        item["batch_slot"] = _slot_dict(slot)
        item["batch_selection_strategy"] = contract.batch_strategy


def _slot_from_item(item: dict[str, Any], role: dict[str, str], slot_id: int) -> BatchSlot:
    candidate_index = int(item.get("candidate_index", slot_id - 1) or 0)
    return BatchSlot(
        slot_id=slot_id,
        role=str(role.get("batch_role") or "batch_member"),
        candidate_index=candidate_index,
        purpose=str(role.get("batch_role_reason") or ""),
        varied_variables=_changed_variables(item.get("candidate") or {}, {}),
        rationale=str(role.get("batch_role_reason") or ""),
    )


def _slot_dict(slot: BatchSlot) -> dict[str, Any]:
    return {
        "batch_slot_id": int(slot.slot_id),
        "batch_role": slot.role,
        "batch_role_reason": slot.purpose or slot.rationale,
        "batch_slot_purpose": slot.purpose,
        "batch_slot_rationale": slot.rationale,
        "batch_slot_risk_note": slot.risk_note,
        "batch_slot_evidence_refs": list(slot.evidence_refs),
        "batch_slot_varied_variables": list(slot.varied_variables),
        "batch_slot_controlled_variables": list(slot.controlled_variables),
    }


def contract_to_dict(contract: BatchContract) -> dict[str, Any]:
    return {
        "batch_strategy": contract.batch_strategy,
        "batch_rationale": contract.batch_rationale,
        "global_constraints": list(contract.global_constraints),
        "slots": [
            {
                "slot_id": slot.slot_id,
                "role": slot.role,
                "candidate_index": slot.candidate_index,
                "purpose": slot.purpose,
                "varied_variables": list(slot.varied_variables),
                "controlled_variables": list(slot.controlled_variables),
                "rationale": slot.rationale,
                "risk_note": slot.risk_note,
                "evidence_refs": list(slot.evidence_refs),
            }
            for slot in contract.slots
        ],
    }


def validation_report_to_dict(report: BatchValidationReport) -> dict[str, Any]:
    return {
        "status": report.status,
        "issues": list(report.issues),
        "repairs": list(report.repairs),
        "fallback_used": bool(report.fallback_used),
    }


def _batch_diversity_dimensions(
    candidate_pool: list[dict[str, Any]],
    decision_context: dict[str, Any],
) -> list[str]:
    underexplored = [
        str(item)
        for item in decision_context.get("underexplored_dimensions", [])
        if str(item).strip()
    ]
    first_candidate = next(
        (item.get("candidate") for item in candidate_pool if isinstance(item.get("candidate"), dict)),
        {},
    )
    all_dims = list(first_candidate.keys())
    return underexplored + [dim for dim in all_dims if dim not in underexplored]


def _diversity_score(
    item: dict[str, Any],
    selected: list[dict[str, Any]],
    dimensions: list[str],
) -> tuple[int, int, int, int, int]:
    candidate = item.get("candidate") or {}
    selected_candidates = [entry.get("candidate") or {} for entry in selected]
    novelty = 0
    for dim in dimensions:
        selected_values = {str(entry.get(dim, "")) for entry in selected_candidates}
        if str(candidate.get(dim, "")) not in selected_values:
            novelty += 1
    pair_dims = [dim for dim in ("Additive", "Solvent", "TEMPO derivative", "Catalyst") if dim in candidate]
    pair = tuple(str(candidate.get(dim, "")) for dim in pair_dims[:2])
    selected_pairs = {
        tuple(str(entry.get(dim, "")) for dim in pair_dims[:2])
        for entry in selected_candidates
    }
    pair_novelty = 1 if pair_dims and pair not in selected_pairs else 0
    descriptor_novelty = 1 if _descriptor_distance(item) > 0 else 0
    rank = int(item.get("bo_rank", item.get("candidate_index", 0)) or 0)
    return (
        descriptor_novelty,
        pair_novelty,
        novelty,
        -rank,
        -int(item.get("candidate_index", rank) or rank),
    )


def _role_for_candidate(
    item: dict[str, Any],
    selected: list[dict[str, Any]],
    dimensions: list[str],
    *,
    first: bool,
) -> dict[str, str]:
    candidate = item.get("candidate") or {}
    if first:
        return {
            "batch_role": "planner_anchor",
            "batch_role_reason": "keeps the planner's highest-priority candidate visible in the batch",
        }
    anchor = (selected[0].get("candidate") or {}) if selected else {}
    changed = _changed_variables(candidate, anchor, dimensions=dimensions)
    descriptor_phrase = _descriptor_contrast_phrase(item)
    if descriptor_phrase:
        role = "descriptor_contrast"
    elif "Additive" in changed:
        role = "additive_contrast"
    elif "Catalyst" in changed:
        role = "catalyst_contrast"
    elif "Solvent" in changed:
        role = "solvent_contrast"
    elif "TEMPO derivative" in changed:
        role = "tempo_probe"
    elif changed:
        role = f"{changed[0].lower().replace(' ', '_')}_probe"
    else:
        role = "local_refinement"
    reason = (
        "adds batch diversity relative to the planner anchor"
        if changed
        else "keeps a local refinement near the planner anchor"
    )
    if changed:
        reason += f" by changing {', '.join(changed[:3])}"
    if descriptor_phrase:
        reason += f"; descriptor contrast: {descriptor_phrase}"
    return {"batch_role": role, "batch_role_reason": reason}


def _descriptor_distance(item: dict[str, Any]) -> float:
    contrast = item.get("descriptor_contrast_to_anchor") or {}
    distances = []
    for payload in contrast.values():
        if not isinstance(payload, dict):
            continue
        try:
            distances.append(float(payload.get("l2_distance") or 0.0))
        except (TypeError, ValueError):
            continue
    return max(distances or [0.0])


def _descriptor_contrast_phrase(item: dict[str, Any], *, max_variables: int = 2) -> str:
    contrast = item.get("descriptor_contrast_to_anchor") or {}
    phrases: list[str] = []
    for variable, payload in contrast.items():
        if not isinstance(payload, dict):
            continue
        deltas = payload.get("top_descriptor_deltas") or {}
        if not deltas:
            continue
        top_keys = list(deltas.keys())[:2]
        if top_keys:
            phrases.append(f"{variable} shifts {', '.join(top_keys)}")
        if len(phrases) >= max_variables:
            break
    return "; ".join(phrases)


def _changed_variables(
    candidate: dict[str, Any],
    anchor: dict[str, Any],
    *,
    dimensions: list[str] | None = None,
) -> list[str]:
    dims = list(dimensions or candidate.keys())
    return [dim for dim in dims if str(candidate.get(dim, "")) != str(anchor.get(dim, ""))]


def _candidate_signature(candidate: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in candidate.items()))
