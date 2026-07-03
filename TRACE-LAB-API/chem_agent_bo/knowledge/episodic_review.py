"""Run-local episodic review staging for milestone 3."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


def _normalize_confidence(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"high", "medium", "low"}:
        return text
    return "low"


def _initial_staging_bucket(
    *,
    benchmark_read_forbidden: bool,
    leakage_risk: str,
    confidence: str,
    contains_exact_candidate: bool,
    contains_result_value: bool,
) -> tuple[str, str]:
    if benchmark_read_forbidden or leakage_risk == "high":
        return (
            "pending",
            "benchmark_or_leakage_sensitive_requires_manual_review",
        )
    if confidence == "low":
        return ("reject", "low_confidence_reflection")
    if not contains_exact_candidate and not contains_result_value:
        return ("generalizable", "strategy_level_without_answer_content")
    return ("keep", "useful_run_local_experience")


@dataclass
class EpisodicReviewCandidate:
    """Structured experience candidate pending or entering review staging."""

    episode_id: str
    run_id: str
    dataset: str
    iteration: int
    source_node: str
    situation_pattern: dict[str, Any]
    action_pattern: dict[str, Any]
    observed_outcome: dict[str, Any]
    reflection_summary: str
    confidence: str
    abstraction_level: str
    contains_exact_candidate: bool
    contains_result_value: bool
    leakage_risk: str
    review_status: str
    staging_bucket: str
    staging_reason: str
    benchmark_read_forbidden: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "dataset": self.dataset,
            "iteration": self.iteration,
            "source_node": self.source_node,
            "situation_pattern": dict(self.situation_pattern),
            "action_pattern": dict(self.action_pattern),
            "observed_outcome": dict(self.observed_outcome),
            "reflection_summary": self.reflection_summary,
            "confidence": self.confidence,
            "abstraction_level": self.abstraction_level,
            "contains_exact_candidate": self.contains_exact_candidate,
            "contains_result_value": self.contains_result_value,
            "leakage_risk": self.leakage_risk,
            "review_status": self.review_status,
            "staging_bucket": self.staging_bucket,
            "staging_reason": self.staging_reason,
            "benchmark_read_forbidden": self.benchmark_read_forbidden,
        }


def build_episodic_review_candidate(
    *,
    run_id: str,
    dataset: str,
    iteration: int,
    protocol_mode: str,
    trigger_reasons: list[str],
    controller_plan: dict[str, Any] | None,
    diagnosis: dict[str, Any] | None,
    reflection_action: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    result: float | None,
    improved_best: bool,
) -> EpisodicReviewCandidate | None:
    reflection = reflection_action or {}
    insight = str(reflection.get("insight", "")).strip()
    if not insight:
        return None
    contains_exact_candidate = bool(candidate)
    contains_result_value = result is not None
    leakage_risk = "high" if (contains_exact_candidate or contains_result_value) else "low"
    confidence = _normalize_confidence(reflection.get("confidence", "low"))
    benchmark_read_forbidden = str(protocol_mode).strip().lower() == "benchmark_clean"
    staging_bucket, staging_reason = _initial_staging_bucket(
        benchmark_read_forbidden=benchmark_read_forbidden,
        leakage_risk=leakage_risk,
        confidence=confidence,
        contains_exact_candidate=contains_exact_candidate,
        contains_result_value=contains_result_value,
    )
    return EpisodicReviewCandidate(
        episode_id=f"{run_id}__iter_{iteration:03d}",
        run_id=run_id,
        dataset=dataset,
        iteration=int(iteration),
        source_node="reflection_action",
        situation_pattern={
            "trigger_reasons": list(trigger_reasons),
            "diagnosis_type": str((diagnosis or {}).get("stagnation_type", "none")),
            "diagnosis_recommendation": str(
                (diagnosis or {}).get("recommended_intervention", "none")
            ),
        },
        action_pattern={
            "controller_mode": str((controller_plan or {}).get("intervention_type", "bo_direct")),
            "action_package": dict((controller_plan or {}).get("action_package", {}) or {}),
            "focus_variables": list((controller_plan or {}).get("focus_variables", [])),
            "suggested_focus": list(reflection.get("suggested_focus", [])),
            "avoid_pattern": list(reflection.get("avoid_pattern", [])),
        },
        observed_outcome={
            "improved_best": bool(improved_best),
            "result": result,
        },
        reflection_summary=insight,
        confidence=confidence,
        abstraction_level="strategy",
        contains_exact_candidate=contains_exact_candidate,
        contains_result_value=contains_result_value,
        leakage_risk=leakage_risk,
        review_status="pending_review",
        staging_bucket=staging_bucket,
        staging_reason=staging_reason,
        benchmark_read_forbidden=benchmark_read_forbidden,
    )


class EpisodicReviewQueue:
    """Run-local append-only staging area for episodic review candidates."""

    _BUCKET_NAMES = ("pending", "keep", "reject", "generalizable")

    def __init__(self, output_path: str | Path, *, reset: bool = True) -> None:
        raw_path = Path(output_path)
        if raw_path.suffix == ".jsonl":
            self.aggregate_path = raw_path
            self.staging_dir = raw_path.parent / "episodic_review_staging"
        else:
            self.staging_dir = raw_path
            self.aggregate_path = self.staging_dir / "all_candidates.jsonl"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        self.bucket_paths = {
            bucket: self.staging_dir / f"{bucket}.jsonl"
            for bucket in self._BUCKET_NAMES
        }
        self.review_log_path = self.staging_dir / "review_log.jsonl"
        self.summary_path = self.staging_dir / "summary.json"
        self._records: list[dict[str, Any]] = []
        self._bucket_counts = {bucket: 0 for bucket in self._BUCKET_NAMES}
        if reset:
            self.aggregate_path.write_text("", encoding="utf-8")
            for path in self.bucket_paths.values():
                path.write_text("", encoding="utf-8")
            self.review_log_path.write_text("", encoding="utf-8")
        else:
            self.aggregate_path.touch(exist_ok=True)
            for path in self.bucket_paths.values():
                path.touch(exist_ok=True)
            self.review_log_path.touch(exist_ok=True)
            self._load_existing_records()
        self._sync_summary()

    def _load_existing_records(self) -> None:
        if not self.aggregate_path.exists():
            return
        for line in self.aggregate_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            self._records.append(payload)
            bucket = str(payload.get("staging_bucket", "pending")).strip().lower()
            if bucket not in self._bucket_counts:
                bucket = "pending"
            self._bucket_counts[bucket] = self._bucket_counts.get(bucket, 0) + 1

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _sync_summary(self) -> None:
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2, ensure_ascii=False, default=str)

    def append(self, candidate: EpisodicReviewCandidate) -> None:
        payload = candidate.to_dict()
        bucket = str(payload.get("staging_bucket", "pending")).strip().lower()
        if bucket not in self.bucket_paths:
            bucket = "pending"
            payload["staging_bucket"] = bucket
        self._records.append(payload)
        self._bucket_counts[bucket] = self._bucket_counts.get(bucket, 0) + 1
        self._append_jsonl(self.aggregate_path, payload)
        self._append_jsonl(self.bucket_paths[bucket], payload)
        self._sync_summary()

    def record_manual_decision(
        self,
        *,
        episode_id: str,
        new_bucket: str,
        reviewer: str,
        note: str = "",
        decision_reason: str = "",
        redacted_summary: str = "",
        generalized_form: str = "",
    ) -> None:
        bucket = str(new_bucket).strip().lower()
        if bucket not in self.bucket_paths:
            raise ValueError(f"Unsupported episodic review bucket: {new_bucket}")
        payload = {
            "episode_id": episode_id,
            "new_bucket": bucket,
            "reviewer": reviewer,
            "note": note,
            "decision_reason": decision_reason,
            "redacted_summary": redacted_summary,
            "generalized_form": generalized_form,
        }
        self._append_jsonl(self.review_log_path, payload)

    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def summary(self) -> dict[str, Any]:
        leakage_counts: dict[str, int] = {}
        for row in self._records:
            risk = str(row.get("leakage_risk", "unknown"))
            leakage_counts[risk] = leakage_counts.get(risk, 0) + 1
        return {
            "candidate_count": len(self._records),
            "leakage_risk_counts": leakage_counts,
            "aggregate_queue_path": str(self.aggregate_path),
            "staging_dir": str(self.staging_dir),
            "bucket_counts": dict(self._bucket_counts),
            "bucket_paths": {name: str(path) for name, path in self.bucket_paths.items()},
            "review_log_path": str(self.review_log_path),
            "summary_path": str(self.summary_path),
        }
