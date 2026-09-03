"""Local-LLM synthesis of Direct/Adjacent/Speculative product opportunities
(blueprint Sec 33/49) from an assessment's already-grounded applications.

Every other assess_* function in this package is extractive/deterministic.
This is the one exception - see assessment/opportunities.py's docstring
for why nothing in the deterministic toolbox (extractive claims, embedding
similarity, deterministic clustering) can honestly produce a Direct/
Adjacent/Speculative product framing, which requires inventing a concept
that is not literally present in any paper. The precedent this follows is
qa/summarize.py: a local Ollama call, off by default (OLLAMA_ENABLED,
reused as-is - no new config), with every citation the model emits
checked against real indices before anything is shown, one retry, then
fail closed. See docs/superpowers/specs/
2026-09-03-opportunities-synthesis-design.md for the full design.

The Ollama HTTP-call/retry envelope is deliberately duplicated from
qa/summarize.py rather than imported, matching this codebase's own stated
precedent for the same tradeoff (assessment/applications.py's
_restates_own_task, duplicated from gaps/detect.py "since the two
modules' surrounding logic ... differ enough that sharing code across two
call sites would add more indirection than it saves") - the two prompts,
response shapes, and validation rules differ enough that a shared
abstraction would mostly be indirection.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Literal

import requests

logger = logging.getLogger(__name__)

Tier = Literal["direct", "adjacent", "speculative"]
_TIERS: tuple[Tier, ...] = ("direct", "adjacent", "speculative")


@dataclass
class SourceApplication:
    """One entry from a persisted assessment's potential_applications - the
    same {application, source_paper, paper_id, evidence_id} shape
    build_assessment() already writes (see build.py), deliberately NOT
    assessment/applications.py's ApplicationRecord: that dataclass exists
    for build_assessment()'s own evidence-linking, whereas this one is
    reconstructed from plain JSON already read back out of the database at
    synthesis time. evidence_id is optional only for backward compatibility
    with assessment rows persisted before this field was added (see
    assessment_routes.py's synthesize_assessment_opportunities, which skips
    evidence-linking a cited application that has none)."""

    application: str
    source_paper: str
    paper_id: str
    evidence_id: str | None = None

SYSTEM_PROMPT = (
    "You are given a numbered list of applications already identified for a research idea, each "
    "grounded in a specific paper. Propose exactly three product/technology opportunities that "
    "build on these applications: one Direct (a straightforward product built from one "
    "application as stated), one Adjacent (a broader product combining or extending the "
    "applications, still plausible from what's listed), and one Speculative (an ambitious, "
    "longer-horizon idea, clearly still connected to the applications). Do not invent a "
    "capability, technology, or claim that isn't implied by the numbered applications. Format "
    "your response as exactly three lines, in this exact order:\n"
    "Direct: <opportunity> [n]\n"
    "Adjacent: <opportunity> [n][m]\n"
    "Speculative: <opportunity> [n]\n"
    "where each line's [n] cites which application number(s) it draws from."
)

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_TIER_LINE_RE = re.compile(r"^\s*(direct|adjacent|speculative)\s*:\s*(.+)$", re.IGNORECASE)


def _escape_bracketed_numbers(text: str) -> str:
    """Same guard as qa/summarize.py's _escape_bracketed_numbers: neutralizes
    a literal [n] substring inside application text so citation extraction
    can't mistake it for a citation marker the model itself emitted."""
    return _CITATION_PATTERN.sub(lambda m: f"({m.group(1)})", text)


def build_prompt(applications: list[SourceApplication]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the Ollama chat call."""
    numbered = "\n".join(
        f'[{i}] "{_escape_bracketed_numbers(app.application)}" — {app.source_paper}'
        for i, app in enumerate(applications, start=1)
    )
    user_prompt = f"Applications:\n{numbered}"
    return SYSTEM_PROMPT, user_prompt


@dataclass
class SynthesizedOpportunity:
    tier: Tier
    opportunity: str
    source_application_indices: list[int] = field(default_factory=list)


def parse_response(text: str, application_count: int) -> list[SynthesizedOpportunity]:
    """Parses the model's three labeled lines into one SynthesizedOpportunity
    per tier, in Direct/Adjacent/Speculative order regardless of the order
    the model wrote them in. Raises ValueError if any tier is missing or
    duplicated, a line has no citation at all, or any cited index is
    outside 1..application_count - the caller treats this identically to
    an unreachable model (retry, then fail closed)."""
    by_tier: dict[Tier, SynthesizedOpportunity] = {}

    for raw_line in text.splitlines():
        match = _TIER_LINE_RE.match(raw_line)
        if not match:
            continue
        tier = match.group(1).lower()
        assert tier in _TIERS
        body = match.group(2).strip()

        citations: list[int] = []
        for cite_match in _CITATION_PATTERN.finditer(body):
            n = int(cite_match.group(1))
            if n < 1 or n > application_count:
                raise ValueError(f"citation [{n}] is out of range for {application_count} applications")
            if n not in citations:
                citations.append(n)
        if not citations:
            raise ValueError(f"{tier} opportunity has no citation")

        opportunity_text = _CITATION_PATTERN.sub("", body).strip()
        if not opportunity_text:
            raise ValueError(f"{tier} opportunity has no text beyond its citation")

        if tier in by_tier:
            raise ValueError(f"duplicate {tier} line in response")
        by_tier[tier] = SynthesizedOpportunity(
            tier=tier, opportunity=opportunity_text, source_application_indices=citations
        )

    missing = [t for t in _TIERS if t not in by_tier]
    if missing:
        raise ValueError(f"missing tier(s) in response: {', '.join(missing)}")

    return [by_tier[t] for t in _TIERS]


@dataclass
class SynthesisResult:
    opportunities: list[SynthesizedOpportunity]


class OpportunitySynthesisUnavailable(Exception):
    """Raised when OLLAMA_ENABLED is false, Ollama is unreachable/times out,
    or the response doesn't parse into three validly-cited tiers after one
    retry. The route layer turns this into a 503 - never a partially
    validated result persisted to potential_opportunities."""


def ollama_enabled() -> bool:
    """Same flag as qa/summarize.py.ollama_enabled() - deliberately reused,
    not a new setting: wherever an operator already turned on the Q&A
    summary layer, this becomes available too."""
    return os.environ.get("OLLAMA_ENABLED", "false").lower() == "true"


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


def synthesize_opportunities(applications: list[SourceApplication]) -> SynthesisResult:
    """Calls the local Ollama model to synthesize three grounded product
    opportunities from the given applications. Retries once on an
    unreachable model or an invalid/incomplete response, then raises
    OpportunitySynthesisUnavailable - never returns a result whose
    citations weren't checked against the given applications."""
    if not ollama_enabled():
        raise OpportunitySynthesisUnavailable("local LLM opportunity synthesis is not enabled")
    if not applications:
        raise OpportunitySynthesisUnavailable("no applications to ground opportunity synthesis in")

    system_prompt, user_prompt = build_prompt(applications)
    timeout = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "30"))

    for attempt in range(2):
        try:
            content = _call_ollama(system_prompt, user_prompt, timeout)
            opportunities = parse_response(content, len(applications))
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            logger.warning("Ollama opportunity synthesis attempt %d failed: %s", attempt + 1, exc)
            continue
        return SynthesisResult(opportunities=opportunities)

    raise OpportunitySynthesisUnavailable("local LLM could not produce a valid grounded synthesis")
