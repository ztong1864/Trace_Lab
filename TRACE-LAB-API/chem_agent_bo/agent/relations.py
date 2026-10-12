"""How a candidate relates to the experiments already measured, computed by code.

The semantic assessment used to leave this to the LLM, which compared conditions field by field and
called one-variable probes "an exact repeat of the best observation" (L1, round_002: LiCl for NaCl,
5 for 4 mol% Fe, ABNO for AZADO). Duplicates and differences are facts the code can state exactly.
"""
from __future__ import annotations

import re
from typing import Any

# Phrases with which a comment claims the candidate repeats a measured experiment.
_REPEAT_CLAIM_RE = re.compile(
    r"\b(exact(ly)?\s+(repeat|duplicate|replicat)\w*|repeats?\s+(the|an?)\s+[\w\s-]{0,40}(condition|observation|experiment)"
    r"|duplicates?\s+(the|an?)\s+[\w\s-]{0,40}(condition|observation|experiment)|identical\s+to\s+(the|an?)\s+"
    r"[\w\s-]{0,30}(observed|measured|best)|same\s+as\s+the\s+[\w\s-]{0,20}(best|measured|observed)"
    r"|already\s+(been\s+)?(measured|tested|run))\b",
    re.IGNORECASE,
)


def _same(a: Any, b: Any) -> bool:
    left, right = str(a if a is not None else "").strip(), str(b if b is not None else "").strip()
    if left == right:
        return True
    try:
        return float(left) == float(right)
    except ValueError:
        return False


def relation_to_history(candidate: dict[str, Any], history: list[dict[str, Any]], *,
                        goal: str = "maximize") -> dict[str, Any]:
    """{exact_duplicate_of, closest_measured, measured_runs} for one candidate.

    Only the variables the history row was optimized over (`active_variables`) are compared.
    Among equally close runs the best one is preferred."""
    rows = [row for row in history or [] if isinstance(row.get("candidate"), dict)]
    if not rows:
        return {}
    maximize = str(goal or "maximize").lower() != "minimize"
    best_choice: tuple[Any, ...] | None = None
    closest: dict[str, Any] | None = None
    exact: dict[str, Any] | None = None
    for row in rows:
        measured = row["candidate"]
        variables = list(row.get("active_variables") or measured.keys())
        differences = [
            {"variable": name, "measured": measured.get(name), "candidate": candidate.get(name)}
            for name in variables
            if not _same(measured.get(name), candidate.get(name))
        ]
        try:
            result = float(row.get("result"))
        except (TypeError, ValueError):
            result = None
        summary = {"iteration": row.get("iteration"), "result": result, "differences": differences,
                   "difference_count": len(differences)}
        if not differences and exact is None:
            exact = {"iteration": row.get("iteration"), "result": result}
        score = (len(differences), -(result if result is not None else float("-inf")) if maximize
                 else (result if result is not None else float("inf")))
        if best_choice is None or score < best_choice:
            best_choice, closest = score, summary
    return {"exact_duplicate_of": exact, "closest_measured": closest, "measured_runs": len(rows)}


def describe_differences(relation: dict[str, Any]) -> str:
    """'Additive: NaCl -> LiCl; Solvent: tBuOH -> DCE' for the closest measured run."""
    closest = (relation or {}).get("closest_measured") or {}
    return "; ".join(f"{item['variable']}: {item['measured']} -> {item['candidate']}"
                     for item in closest.get("differences") or [])


def relation_sentence(relation: dict[str, Any]) -> str:
    """One code-made sentence for a recommendation's rationale."""
    if not relation:
        return ""
    exact = relation.get("exact_duplicate_of")
    if exact:
        return (f"This is a repeat measurement of iteration {exact.get('iteration')} "
                f"(measured result {exact.get('result')}).")
    closest = relation.get("closest_measured") or {}
    if not closest:
        return ""
    count = closest.get("difference_count", 0)
    return (f"Compared with the closest measured run (iteration {closest.get('iteration')}, result "
            f"{closest.get('result')}), {count} {'variable differs' if count == 1 else 'variables differ'}: "
            f"{describe_differences(relation)}.")


def claims_repeat(text: str) -> bool:
    return bool(_REPEAT_CLAIM_RE.search(str(text or "")))
