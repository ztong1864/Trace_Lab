"""Shared schemas for question-driven source acquisition."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class SourceRecord:
    source_id: str
    question_id: str
    provider: str
    source_type: str
    query: str
    title: str = ""
    url: str = ""
    doi: str = ""
    local_path: str = ""
    raw_payload_path: str = ""
    retrieved_at: str = field(default_factory=utc_now_iso)
    status: str = "ok"
    error: str = ""
    content_preview: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def write_jsonl(path: Path, records: list[SourceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")


def read_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def question_text(question: dict[str, Any]) -> str:
    parts = [
        str(question.get("question", "")),
        str(question.get("reaction_family", "")),
        " ".join(str(item) for item in question.get("variable_tags", [])),
        " ".join(str(item) for item in question.get("optimization_goals", [])),
    ]
    return " ".join(part for part in parts if part).strip()
