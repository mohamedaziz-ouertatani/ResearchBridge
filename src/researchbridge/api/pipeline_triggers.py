"""Subprocess-based triggers for CLI-driven pipelines (ingestion/extraction/embedding).

Reuses the existing CLI entry points as subprocesses rather than
reimplementing pipeline-triggering logic in the API layer - the same
`rb-ingest`/`rb-ingest-springer`/`rb-ingest-semantic-scholar`/
`rb-ingest-core`/`rb-extract`/`rb-embed`/`rb-retrieval-evaluate` commands an
operator would run by hand, just launched from a button instead of a
terminal. One subprocess per pipeline key at a time; a second trigger
while one is running is refused rather than queued or duplicated.

The registry is in-process memory, not persisted - it resets on server
restart. That's a deliberate tradeoff, not an oversight: the *_runs
tables (see admin_routes.py's pipeline_status) remain the durable
record of what ran and when; this registry only answers "is something
I spawned still alive right now."
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil

from researchbridge.pipeline_logging import LOG_FILE_ENV_VAR, LOGS_DIR

REPO_ROOT = Path(__file__).resolve().parents[3]

_RUNNING: dict[str, subprocess.Popen] = {}

# Without this, a triggered subprocess inherits the API server's console
# (and process group, on Windows) - so ANY interrupt delivered to that
# console (uvicorn's own --reload restarting itself on a source-file
# change; a Ctrl+C in whatever terminal is hosting it) broadcasts to every
# attached process and kills the subprocess too, mid-run, with a bare
# KeyboardInterrupt that has nothing to do with the pipeline itself. This
# gives the child its own process group (Windows) / session (POSIX) so a
# signal aimed at the parent's console no longer reaches it.
_DETACH_FROM_PARENT_CONSOLE: dict[str, object] = (
    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    if sys.platform == "win32"
    else {"start_new_session": True}
)


class PipelineAlreadyRunning(Exception):
    def __init__(self, key: str) -> None:
        super().__init__(f"{key} is already running")
        self.key = key


def is_running(key: str) -> bool:
    proc = _RUNNING.get(key)
    return proc is not None and proc.poll() is None


def tail_log(key: str, lines: int = 200) -> str:
    """The last `lines` lines of the most recent log file for this pipeline
    key, or "" if none exists yet. Finds "most recent" by filename (the
    timestamp trigger() and configure_pipeline_logging() both embed sorts
    lexicographically), not mtime - no extra state to track beyond what's
    already on disk, and it keeps working across a server restart the way
    the in-process _RUNNING registry deliberately doesn't (see module
    docstring). This is also why a run started directly from a terminal
    is tailable here too, as long as its CLI entry point called
    configure_pipeline_logging(key) - see pipeline_logging.py."""
    candidates = sorted(LOGS_DIR.glob(f"{key}-*.log"))
    if not candidates:
        return ""
    content = candidates[-1].read_text(encoding="utf-8", errors="replace")
    return "\n".join(content.splitlines()[-lines:])


def stop(key: str, fallback_pid: int | None = None) -> bool:
    """Terminate the subprocess running under `key`, if any.

    Returns True if something was actually running and got signaled, False
    if there was nothing to stop (already finished, or never started).
    Best-effort: on Windows, Popen.terminate() already maps to
    TerminateProcess - a hard kill, not a graceful SIGTERM - so there's no
    softer signal to try first the way there would be on POSIX. A short
    wait() still happens so `is_running(key)` reads False immediately after
    a caller awaits this, rather than racing the OS's own cleanup.

    fallback_pid (2026-09-05): this in-memory registry resets on server
    restart (see module docstring) - but the subprocess itself is
    deliberately detached from the parent's console specifically so it
    SURVIVES that restart, running on regardless. Before this fallback, a
    restart between trigger and stop left no way to actually stop such a
    run: the caller's own admin_routes.py already solves this exact gap
    for DETECTING liveness (has_running_db_row/_reconcile_stale_running_
    runs, both via psutil.pid_exists against the *_runs row's own stored
    pid), but stop() never got the equivalent - a "stop" button click
    would silently 409 ("not running") against a process that was very
    much still running. When the in-memory handle is gone, fall back to
    killing this OS-level pid directly if it's still alive."""
    proc = _RUNNING.get(key)
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        _RUNNING.pop(key, None)
        return True
    _RUNNING.pop(key, None)

    if fallback_pid is not None and psutil.pid_exists(fallback_pid):
        try:
            proc = psutil.Process(fallback_pid)
            proc.terminate()
            proc.wait(timeout=5)
        except psutil.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        except psutil.NoSuchProcess:
            return False
        return True

    return False


def trigger(key: str, module: str, args: list[str]) -> Path:
    """Launch `python -m <module> <args>` as a background subprocess.

    Raises PipelineAlreadyRunning if a process for this key is still alive.
    Returns the path of the log file the subprocess's stdout/stderr is
    redirected to.
    """
    if is_running(key):
        raise PipelineAlreadyRunning(key)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOGS_DIR / f"{key}-{timestamp}.log"
    log_file = log_path.open("w", buffering=1)

    proc = subprocess.Popen(
        # -u: unbuffered stdout/stderr. Without it, the child's own stdout is
        # block-buffered whenever it's not attached to a terminal (true here,
        # since it's redirected to a file) - progress logging wouldn't reach
        # disk until the buffer filled or the process exited, so the admin
        # page's log tail would sit empty for most of the run instead of
        # actually being live.
        [sys.executable, "-u", "-m", module, *args],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
        # Tells the child's own configure_pipeline_logging() call that its
        # stdout/stderr is already this exact file (via the redirect
        # above), so it adds only a console handler instead of also
        # opening a second FileHandler on the same path (see
        # pipeline_logging.py).
        env={**os.environ, LOG_FILE_ENV_VAR: str(log_path)},
        **_DETACH_FROM_PARENT_CONSOLE,
    )
    _RUNNING[key] = proc
    return log_path
