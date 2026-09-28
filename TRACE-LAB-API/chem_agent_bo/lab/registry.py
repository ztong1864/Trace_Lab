"""Registry of lab projects that live outside the server's projects root.

A project is addressed by a *handle*: the name used in every API URL. Projects
inside the projects root use their folder name; projects registered from
elsewhere (e.g. a folder next to the raw lab data) are listed in
``<projects_root>/registry.yaml`` as ``handle -> absolute folder path`` and are
read and written in place.
"""

from __future__ import annotations

import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REGISTRY_FILENAME = "registry.yaml"
# Letters, digits, "-" and "_" only, so a handle can never be a path ("..", "a/b", "C:").
_HANDLE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_REGISTRY_LOCKS: dict[str, threading.Lock] = {}
_REGISTRY_LOCKS_GUARD = threading.Lock()


def validate_handle(value: Any) -> str:
    handle = str(value or "").strip()
    if not _HANDLE_PATTERN.match(handle):
        raise ValueError(
            f"Invalid project handle `{handle}`: use 1-128 letters, digits, '-' or '_', "
            "starting with a letter or digit."
        )
    return handle


def default_handle(folder: str | Path) -> str:
    """A valid handle derived from a folder name (other characters become '_')."""
    name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(folder).name)
    return validate_handle(name.strip("_-") or "project")


def _registry_lock(path: Path) -> threading.Lock:
    key = os.path.normcase(str(path.resolve()))
    with _REGISTRY_LOCKS_GUARD:
        return _REGISTRY_LOCKS.setdefault(key, threading.Lock())


def _same_folder(a: str | Path, b: str | Path) -> bool:
    return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))


class ProjectRegistry:
    """Handles of registered outside projects, stored next to the projects root."""

    def __init__(self, projects_root: str | Path) -> None:
        self.projects_root = Path(projects_root)
        self.path = self.projects_root / REGISTRY_FILENAME

    def entries(self) -> dict[str, dict[str, str]]:
        if not self.path.exists():
            return {}
        payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        projects = payload.get("projects") or {}
        if not isinstance(projects, dict):
            raise ValueError(f"Malformed project registry: {self.path}")
        return {str(handle): dict(entry or {}) for handle, entry in projects.items()}

    def resolve(self, handle: str) -> Path | None:
        entry = self.entries().get(handle)
        return Path(entry["path"]) if entry and entry.get("path") else None

    def handle_for(self, folder: str | Path) -> str | None:
        for handle, entry in self.entries().items():
            if entry.get("path") and _same_folder(entry["path"], folder):
                return handle
        return None

    def add(self, handle: str, folder: str | Path) -> dict[str, str]:
        handle = validate_handle(handle)
        folder = Path(folder).resolve()
        with _registry_lock(self.path):
            entries = self.entries()
            if handle in entries:
                raise ValueError(
                    f"Handle `{handle}` is already registered for {entries[handle].get('path')}; "
                    "choose another handle."
                )
            if (self.projects_root / handle).exists():
                raise ValueError(
                    f"Handle `{handle}` is already used by a project folder in the projects root; "
                    "choose another handle."
                )
            existing = next(
                (name for name, entry in entries.items() if entry.get("path") and _same_folder(entry["path"], folder)),
                None,
            )
            if existing:
                raise ValueError(f"{folder} is already registered as `{existing}`.")
            entry = {"path": str(folder), "registered_at": datetime.now(timezone.utc).isoformat()}
            entries[handle] = entry
            self._write(entries)
            return entry

    def remove(self, handle: str) -> dict[str, str]:
        handle = validate_handle(handle)
        with _registry_lock(self.path):
            entries = self.entries()
            if handle not in entries:
                raise KeyError(f"No registered project with handle `{handle}`.")
            entry = entries.pop(handle)
            self._write(entries)
            return entry

    def _write(self, entries: dict[str, dict[str, str]]) -> None:
        self.projects_root.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.tmp")
        temp.write_text(
            yaml.safe_dump({"projects": entries}, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(temp, self.path)
