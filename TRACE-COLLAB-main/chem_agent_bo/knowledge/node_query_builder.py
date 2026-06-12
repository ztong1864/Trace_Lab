"""Node-aware reviewed knowledge query construction."""

from __future__ import annotations

from typing import Any


class ReviewedKnowledgeQueryBuilder:
    """Build compact node-specific requests for reviewed knowledge access."""

    @staticmethod
    def _recent_trigger_tags(node_state_view: dict[str, Any]) -> list[str]:
        trace = (node_state_view.get("recent_decision_trace") or {}).get("items", [])
        if not trace:
            return []
        latest = trace[-1]
        return [str(item) for item in latest.get("trigger_reasons", [])]

    def build(
        self,
        *,
        node_name: str,
        node_state_view: dict[str, Any],
        reaction_context: dict[str, Any],
        candidate: dict[str, Any] | None = None,
        result: float | None = None,
        dataset_meta: dict[str, Any] | None = None,
        sample_pool: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        dataset = str(reaction_context.get("dataset", "")).strip()
        reaction_type = str(reaction_context.get("reaction_type", "")).strip()
        coverage_state = node_state_view.get("coverage_state", {})
        active_state = node_state_view.get("active_hypothesis_state", {})
        variable_tags = list(coverage_state.get("underexplored_dimensions", []))
        if dataset_meta:
            variable_tags.extend(str(item) for item in dataset_meta.get("scaffold_dims", []) if str(item).strip())
        if candidate:
            variable_tags.extend(str(key) for key in candidate.keys())
        if sample_pool:
            for item in sample_pool[:3]:
                if isinstance(item, dict):
                    variable_tags.extend(str(key) for key in item.keys())
        trigger_tags = self._recent_trigger_tags(node_state_view)
        planner_name = str((reaction_context or {}).get("backend", "")).strip().lower()
        if planner_name:
            trigger_tags.append(f"planner_{planner_name}")
        no_improvement_rounds = int(node_state_view.get("no_improvement_rounds", 0) or 0)
        if no_improvement_rounds >= 2:
            trigger_tags.append("stagnation")
        local_lock_score = float(node_state_view.get("local_lock_score", 0.0) or 0.0)
        if local_lock_score >= 0.5:
            trigger_tags.append("local_lock")
        remaining_budget = int(node_state_view.get("remaining_budget", 0) or 0)
        if 0 < remaining_budget <= 6:
            trigger_tags.append("late_budget")
        underexplored = [
            str(item).strip()
            for item in coverage_state.get("underexplored_dimensions", [])
            if str(item).strip()
        ]
        if underexplored == ["Additive"]:
            trigger_tags.append("underexplored_additive_only")
        if underexplored:
            trigger_tags.extend(f"underexplored_{item}" for item in underexplored)
        dataset_lower = dataset.lower()
        if dataset_lower.startswith("buchwald_task_"):
            trigger_tags.append("buchwald_product_subtask")
        if dataset_lower == "buchwald_task_1":
            trigger_tags.append("low_ceiling_task")
        focus_terms = active_state.get("suggested_next_focus", [])
        focus_text = "; ".join(str(item) for item in focus_terms[:3])
        best_value = node_state_view.get("best_value")
        query_parts = [
            dataset,
            reaction_type,
            node_name.replace("_", " "),
        ]
        if best_value is not None:
            query_parts.append(f"best={best_value}")
        if focus_text:
            query_parts.append(f"focus={focus_text[:180]}")
        if candidate:
            candidate_text = ", ".join(f"{key}={value}" for key, value in candidate.items())
            query_parts.append(f"candidate={candidate_text[:220]}")
        if result is not None:
            query_parts.append(f"observed_result={result}")
        if dataset_meta and dataset_meta.get("description"):
            query_parts.append(f"context={str(dataset_meta.get('description'))[:180]}")
        query = " | ".join(part for part in query_parts if part)
        return {
            "node_name": node_name,
            "dataset": dataset,
            "reaction_type": reaction_type,
            "query": query,
            "variable_tags": list(dict.fromkeys(variable_tags))[:8],
            "trigger_tags": list(dict.fromkeys(trigger_tags))[:10],
        }
