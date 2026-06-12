"""Runtime I/O helpers for readable experiment execution."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterator


@contextmanager
def capture_third_party_output(
    *,
    enabled: bool,
    log_path: str | Path | None,
    label: str,
) -> Iterator[None]:
    """Redirect noisy library stdout/stderr to a run-local log file."""
    if not enabled or log_path is None:
        yield
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== {label} =====\n")
        handle.flush()
        with redirect_stdout(handle), redirect_stderr(handle):
            yield
