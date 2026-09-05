"""Local-LLM summarization layer over the extractive Q&A quotes.

Takes the exact QuoteHitOut list the client already received from
POST /api/ask and asks a local Ollama model to write a short synthesis
that only rephrases/connects those quotes, citing each by its [n]
index. The prompt constrains the model to the numbered quotes, and
every citation number the model DOES emit is checked against the hits
it was actually given (see extract_citations) - this is citation
existence validation, not a guarantee that the model never writes
unfounded prose. This is the one place in the app that calls a
generative model; see docs/superpowers/specs/
2026-08-26-ollama-summary-layer-design.md for why, and why it stays
optional/off-by-default.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import requests

from researchbridge.api.schemas import QuoteHitOut

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are given a question and a numbered list of verbatim quotes from research papers. "
    "Write a 3-5 sentence synthesis using ONLY information stated in these quotes - never add "
    "outside knowledge or infer anything not explicitly present. After every sentence, cite the "
    "quote number(s) it draws from in brackets, e.g. [1] or [1][3]. If the quotes don't address "
    "the question, say that plainly instead of guessing. Do not use any information not in the "
    "numbered quotes above - only use the quotes given."
)

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _escape_bracketed_numbers(text: str) -> str:
    """Neutralizes literal [n] substrings inside quote text (e.g. a
    bibliography reference copied verbatim from the source paper) so the
    citation-extraction regex can never mistake text that originated
    inside a quote for a citation marker the model itself emitted."""
    return _CITATION_PATTERN.sub(lambda m: f"({m.group(1)})", text)


def build_prompt(question: str, hits: list[QuoteHitOut]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the Ollama chat call."""
    numbered = "\n".join(
        f'[{i}] "{_escape_bracketed_numbers(hit.text)}" — {hit.paper_title}'
        for i, hit in enumerate(hits, start=1)
    )
    user_prompt = f"Question: {question}\n\nQuotes:\n{numbered}"
    return SYSTEM_PROMPT, user_prompt


def extract_citations(text: str, hit_count: int) -> list[int]:
    """Extracts [n] citation markers from the model's output, in order of
    first appearance, deduplicated. Raises ValueError if any citation
    number is outside 1..hit_count - the caller treats this identically
    to an unreachable model (retry, then fail closed)."""
    seen: list[int] = []
    for match in _CITATION_PATTERN.finditer(text):
        n = int(match.group(1))
        if n < 1 or n > hit_count:
            raise ValueError(f"citation [{n}] is out of range for {hit_count} quotes")
        if n not in seen:
            seen.append(n)
    return seen


@dataclass
class SummaryResult:
    summary: str
    citations: list[int]


class SummarizationUnavailable(Exception):
    """Raised when OLLAMA_ENABLED is false, or Ollama is unreachable/times
    out/produces an invalid summary after one retry. The route layer turns
    this into a 503 - never a partially-validated summary."""


def ollama_enabled() -> bool:
    # Default TRUE (2026-09-05): opportunity/application-relevance/summary
    # LLM stages are all fail-safe (fail-open or fail-closed to a
    # deterministic NULL/unfiltered result - never a crash or fabricated
    # output) if Ollama isn't actually installed/running, so defaulting to
    # "try it" costs a real deployment nothing but a timeout, while
    # defaulting to "off" silently left every fresh clone/deployment's
    # product-opportunity field NULL forever unless an operator happened to
    # discover and flip this var. Explicitly set OLLAMA_ENABLED=false to
    # opt back out.
    return os.environ.get("OLLAMA_ENABLED", "true").lower() == "true"


def _call_ollama(system_prompt: str, user_prompt: str, timeout: float) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

    response = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=timeout,
        proxies={"http": None, "https": None},
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def summarize_quotes(question: str, hits: list[QuoteHitOut]) -> SummaryResult:
    """Calls the local Ollama model to synthesize a grounded summary of
    the given hits. Retries once on an unreachable model or an
    out-of-range citation, then raises SummarizationUnavailable - never
    returns a summary whose citations weren't checked against hits."""
    if not ollama_enabled():
        raise SummarizationUnavailable("local LLM summarization is not enabled")

    system_prompt, user_prompt = build_prompt(question, hits)
    # 20, not 30 (2026-09-04): kept in sync with assessment/opportunity_
    # synthesis.py's and assessment/application_relevance.py's identical
    # default - all three share this one env var, so a split default would
    # only matter when OLLAMA_TIMEOUT_SECONDS is unset, which would then
    # silently give this call a different timeout than the other two for
    # no real reason.
    timeout = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "20"))

    for attempt in range(2):
        try:
            content = _call_ollama(system_prompt, user_prompt, timeout)
            citations = extract_citations(content, len(hits))
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            logger.warning("Ollama summarization attempt %d failed: %s", attempt + 1, exc)
            continue
        return SummaryResult(summary=content, citations=citations)

    raise SummarizationUnavailable("local LLM could not produce a valid grounded summary")
