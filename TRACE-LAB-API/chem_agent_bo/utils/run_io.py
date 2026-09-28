"""Runtime I/O helpers for readable experiment execution."""

from __future__ import annotations

import contextvars
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO


# The log file that stdout/stderr writes from the current request should go to.
# A context variable (not a process-wide swap) keeps concurrent requests in the
# web server's thread pool from redirecting, restoring or closing each other's
# streams.
_capture_target: contextvars.ContextVar[TextIO | None] = contextvars.ContextVar(
    "trace_third_party_capture_target",
    default=None,
)
_install_lock = threading.Lock()
_active_captures = 0
_saved_streams: tuple[Any, Any] | None = None


class _RoutingStream:
    """Stand-in for sys.stdout/sys.stderr while any capture is active.

    Writes go to the calling context's capture file, or to the original stream
    when the caller has no capture. Libraries that keep a reference to this
    object keep working after every capture has ended.
    """

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback

    def _target(self) -> Any:
        return _capture_target.get() or self._fallback

    def write(self, text: str) -> int:
        return self._target().write(text)

    def writelines(self, lines) -> None:  # noqa: ANN001
        self._target().writelines(lines)

    def flush(self) -> None:
        self._target().flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target(), name)


def _enter_routing() -> None:
    global _active_captures, _saved_streams
    with _install_lock:
        if _active_captures == 0:
            _saved_streams = (sys.stdout, sys.stderr)
            sys.stdout = _RoutingStream(sys.stdout)
            sys.stderr = _RoutingStream(sys.stderr)
        _active_captures += 1


def _exit_routing() -> None:
    global _active_captures, _saved_streams
    with _install_lock:
        _active_captures -= 1
        if _active_captures == 0 and _saved_streams is not None:
            # Restore only streams that are still ours; leave anything a caller
            # installed in the meantime alone.
            if isinstance(sys.stdout, _RoutingStream):
                sys.stdout = _saved_streams[0]
            if isinstance(sys.stderr, _RoutingStream):
                sys.stderr = _saved_streams[1]
            _saved_streams = None


@contextmanager
def capture_third_party_output(
    *,
    enabled: bool,
    log_path: str | Path | None,
    label: str,
) -> Iterator[None]:
    """Send noisy library stdout/stderr from the current request to a run-local log file.

    Safe under concurrency: other threads/requests keep writing to their own log
    or to the original console, and nothing is left pointing at a closed file.
    """
    if not enabled or log_path is None:
        yield
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== {label} =====\n")
        handle.flush()
        _enter_routing()
        token = _capture_target.set(handle)
        try:
            yield
        finally:
            _capture_target.reset(token)
            _exit_routing()
