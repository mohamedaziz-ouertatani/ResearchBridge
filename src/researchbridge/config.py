from __future__ import annotations

import os

from dotenv import load_dotenv


def load_config() -> None:
    """Load .env into the process environment, if present. Safe to call multiple times."""
    load_dotenv(override=False)


def ollama_enabled() -> bool:
    """Whether the optional local-LLM stages may run.

    Single definition for the whole project. Four independent copies of
    this predicate previously existed (qa/summarize.py, and the three
    assessment stages: opportunity_synthesis, application_relevance,
    dimensions_llm), each re-reading the same env var with its own
    fallback - so a change to one silently diverged from the rest.

    Default TRUE (2026-09-05): every stage gated on this fails safe
    (fail-open to a deterministic result, or fail-closed to NULL - never a
    crash or fabricated output) if Ollama isn't installed or running, so
    defaulting to "try it" costs a real deployment nothing but a timeout,
    while defaulting to "off" silently left every fresh clone's
    product-opportunity field NULL forever unless an operator happened to
    discover and flip this var. Set OLLAMA_ENABLED=false to opt back out.
    """
    return os.environ.get("OLLAMA_ENABLED", "true").lower() == "true"


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def ollama_model(default: str) -> str:
    """The configured model, or the CALLER's own fallback.

    The default is a parameter rather than a constant because the stages
    did not agree on one: qa/summarize.py fell back to "qwen2.5:3b" while
    the three assessment stages fell back to "phi3:mini", all behind the
    same OLLAMA_MODEL var. Consolidating to a single hard-coded default
    would have silently changed which model one of them runs, so each
    caller keeps its own and the divergence is visible here instead of
    being spread across four files.
    """
    return os.environ.get("OLLAMA_MODEL", default)


def ollama_timeout_seconds() -> float:
    return float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "20"))
