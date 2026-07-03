"""Build retrieval query and metadata filters from optimization state."""

from __future__ import annotations

from typing import Any


class KnowledgeQueryBuilder:
    """Create compact retrieval requests for sparse high-level intervention."""

    @staticmethod
    def enable_document_retrieval_when(
        *,
        trigger_reasons: list[str],
        underexplored_dimensions: list[str],
        iteration: int,
    ) -> bool:
        if trigger_reasons:
            return True
        if underexplored_dimensions:
            return True
        # Periodic evidence check even when no explicit trigger.
        return iteration % 10 == 0

    def build(
        self,
        *,
        dataset: str,
        best_observation: dict[str, Any] | None,
        underexplored_dimensions: list[str] | None,
        trigger_reasons: list[str] | None,
        working_memory_summary: dict[str, Any] | None,
        iteration: int,
    ) -> dict[str, Any]:
        underexplored = list(underexplored_dimensions or [])
        triggers = list(trigger_reasons or [])
        wm = working_memory_summary or {}
        last_diag = wm.get("last_diagnosis") or {}
        last_hypo = wm.get("last_hypothesis") or {}
        best_text = ""
        if best_observation:
            parts = [f"{k}={v}" for k, v in best_observation.items() if k != "yield"]
            best_text = ", ".join(parts)
        rule_query = " | ".join(
            [
                f"dataset={dataset}",
                f"triggers={', '.join(triggers) if triggers else 'none'}",
                f"underexplored={', '.join(underexplored) if underexplored else 'none'}",
            ]
        )
        hypothesis_focus = "; ".join(last_hypo.get("hypotheses", [])[:2]).strip()
        document_terms = [
            f"{dataset} reaction optimization",
            "mechanism",
            "variable interaction",
            "catalyst",
            "base",
        ]
        if underexplored:
            document_terms.append("underexplored: " + ", ".join(underexplored[:3]))
        if best_text:
            document_terms.append("current_best: " + best_text)
        if hypothesis_focus:
            document_terms.append("hypothesis: " + hypothesis_focus[:220])
        diagnosis_text = str(last_diag.get("reasoning", "")).strip()
        if diagnosis_text:
            document_terms.append("diagnosis: " + diagnosis_text[:220])
        document_query = " | ".join(document_terms)
        route = "hybrid" if triggers or underexplored else "rule"
        if len(triggers) > 0 and len(underexplored) == 0 and "duplicate_high" not in triggers:
            route = "rule"
        enable_document_retrieval = self.enable_document_retrieval_when(
            trigger_reasons=triggers,
            underexplored_dimensions=underexplored,
            iteration=iteration,
        )
        filters = {
            "dataset": dataset,
            "variable_tags": underexplored,
            "trigger_tags": triggers,
        }
        return {
            "rule_query": rule_query,
            "document_query": document_query,
            "filters": filters,
            "route": route,
            "enable_document_retrieval": enable_document_retrieval,
        }
