"""Subprocess-based triggers for CLI-driven pipelines (ingestion/extraction/embedding).

Reuses the existing CLI entry points as subprocesses rather than
reimplementing pipeline-triggering logic in the API layer - the same
`rb-ingest`/`rb-ingest-springer`/`rb-extract`/`rb-embed` commands an
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

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LOGS_DIR = REPO_ROOT / "logs"

_RUNNING: dict[str, subprocess.Popen] = {}


class PipelineAlreadyRunning(Exception):
    def __init__(self, key: str) -> None:
        super().__init__(f"{key} is already running")
        self.key = key


def is_running(key: str) -> bool:
    proc = _RUNNING.get(key)
    return proc is not None and proc.poll() is None


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
    log_file = log_path.open("w")

    proc = subprocess.Popen(
        [sys.executable, "-m", module, *args],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=REPO_ROOT,
    )
    _RUNNING[key] = proc
    return log_path
