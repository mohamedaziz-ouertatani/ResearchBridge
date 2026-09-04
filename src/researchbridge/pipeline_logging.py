"""Shared logging setup for every pipeline CLI entry point (rb-ingest*,
rb-extract, rb-embed, rb-citations-fetch, rb-gaps-detect, rb-fulltext-
fetch, rb-retrieval-evaluate, rb-extract-evaluate).

Lives outside both `api/` (which owns subprocess triggering) and any one
domain package - every CLI module needs this, so it belongs to none of
them specifically. `api/pipeline_triggers.py` imports LOGS_DIR and
LOG_FILE_ENV_VAR from here rather than the other way around, keeping the
dependency direction the API layer wrapping CLI entry points, not CLI
entry points depending on the API layer for something as basic as log
setup.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = REPO_ROOT / "logs"

# The env var api/pipeline_triggers.py's trigger() sets on a subprocess it
# spawns, naming the exact log file it's already redirecting that
# subprocess's stdout/stderr to at the OS level - configure_pipeline_
# logging() checks for it so a panel-triggered run doesn't ALSO open its
# own FileHandler on that same path (which would truncate/interleave
# against the fd it already inherited) alongside the redirect that
# already works today.
LOG_FILE_ENV_VAR = "RB_PIPELINE_LOG_FILE"


def configure_pipeline_logging(key: str, level: int = logging.INFO) -> None:
    """Sets up a pipeline CLI's logging so its progress ends up somewhere
    the admin panel's live-log tail (api/pipeline_triggers.py::tail_log)
    can find, regardless of how the process was started.

    A run launched via the admin panel's trigger button already gets this
    for free - trigger() redirects the whole subprocess's stdout/stderr to
    logs/{key}-<timestamp>.log before it even starts, so a plain
    logging.basicConfig(level=...) console handler is all that's needed
    (LOG_FILE_ENV_VAR being set is the signal that redirect already
    happened). A run started directly from a terminal has no such
    redirect - nothing before this function existed made its output reach
    the logs/ directory at all, so the admin panel's live-log tail had
    nothing to show for it even once it correctly detected the run as
    alive (see admin_routes.py's pid-based liveness check). This closes
    that gap by opening the same logs/{key}-<timestamp>.log file
    tail_log() already knows how to find, whenever LOG_FILE_ENV_VAR isn't
    already set.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if LOG_FILE_ENV_VAR not in os.environ:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        handlers.append(logging.FileHandler(LOGS_DIR / f"{key}-{timestamp}.log"))
    logging.basicConfig(level=level, handlers=handlers)
