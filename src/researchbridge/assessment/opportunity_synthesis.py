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

Cross-model verification (item 8 of the assessment hardening list - this
was previously "only verified against the default LLM model," spot-
checked, not systematic): ran real synthesis calls against every locally
available model (qwen2.5:3b - the OLLAMA_MODEL default, phi3:mini,
qwen2.5-coder:7b, qwen2.5-coder:7b-instruct-q3_K_M) on real 1/2/5-
application cases pulled from the live DB. All 4 models produced
structurally valid, correctly-cited output on every case tried. This
verification is also what surfaced MIN_OPPORTUNITY_TEXT_LENGTH below - a
content-quality gap the structural checks alone couldn't catch, found
specifically because it was checked against more than one model's actual
output rather than assumed adequate from the default model alone.

Default model changed 2026-09-04 from qwen2.5:3b to phi3:mini, on the
strength of the same verification pass: on a real single-application case
(a terse "critical review" application with little to build an
opportunity from), qwen2.5:3b degenerated to a bare category word so
consistently (4/6, then 3/4, of repeated live calls at this module's own
temperature=0.2) that even the existing one retry wasn't enough - it
failed closed (OpportunitySynthesisUnavailable, surfaced to the user as a
503) on most attempts for this input shape. phi3:mini never reproduced
that pattern across the same repeated testing (nor did qwen2.5-coder:7b,
tied on reliability but picked over: phi3:mini's worst-case latency was
lower - 16.6s vs. 19.6s for the single-application case - and it's a
smaller download, 2.2GB vs. 4.7GB). MIN_OPPORTUNITY_TEXT_LENGTH stays as
the actual safety net either way - this module's job is to never persist
a bad result regardless of which model ends up configured, not to rely
on the model choice alone for correctness.
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

def _build_system_prompt(application_count: int) -> str:
    """Built per call (not a fixed module constant) so the valid citation
    range is stated explicitly, anchored to the real application count -
    found live against the actual model (qwen2.5:3b): an earlier fixed
    prompt's "Adjacent: <opportunity> [n][m]" example hardcoded a two-
    citation shape, and the model pattern-matched it literally even with
    only ONE application available, hallucinating a citation [2] that
    doesn't exist - a single-application assessment (a common real case,
    not a contrived edge case) then failed synthesis on every attempt,
    every retry, 100% of the time. Fixed by (a) using a single, consistent
    "[n]" example across all three tiers instead of implying Adjacent
    always needs two, and (b) stating the valid range out loud.

    2026-09-04: now also takes the idea text (see build_prompt) after a
    real live failure - the prompt previously never told the model what
    the actual research idea WAS, only showed it the numbered
    applications. For a single application that was itself a generic
    multi-item enumeration ("Machine learning can be used in many
    applications such as face detection, speech recognition, medical
    diagnostics, statistical arbitrage, traffic prediction, etc." - the
    only application found for a traffic-congestion-prediction idea), the
    model free-associated across the UNRELATED listed items with nothing
    anchoring it back to the actual idea, producing "Smart Healthcare
    Diagnostics" and "Autonomous Financial Advisory System" as the
    Adjacent/Speculative opportunities for a traffic app. Application.py's
    own Fix A/Fix B already found this exact enumeration-dilution failure
    mode for a different check (own-task-overlap) and fixed it by
    inspecting individual list items; this fixes the analogous problem at
    the synthesis boundary by giving the model the one piece of context
    it was missing - explicit instruction below to stay grounded in the
    idea now backs that up structurally, not just via better luck."""
    only = "only application" if application_count == 1 else f"applications, numbered 1 to {application_count},"
    return (
        f"You are given a research idea and {application_count} numbered {only} already identified for "
        "it, each grounded in a specific paper. Propose exactly three product/technology opportunities "
        "that build on these applications: one Direct (a straightforward product built from one "
        "application as stated), one Adjacent (a broader product combining or extending the "
        "applications, still plausible from what's listed), and one Speculative (an ambitious, "
        "longer-horizon idea, clearly still connected to the applications). Every opportunity must stay "
        "specific to the idea given below - if an application lists several unrelated examples (e.g. "
        "\"used in X, Y, and Z\"), only draw on the example(s) that are actually about THIS idea, never "
        "on the other unrelated examples in that same list. Do not invent a capability, technology, or "
        "claim that isn't implied by the numbered applications. Format your response as exactly three "
        "lines, in this exact order:\n"
        "Direct: <opportunity> [n]\n"
        "Adjacent: <opportunity> [n]\n"
        "Speculative: <opportunity> [n]\n"
        f"where each line's [n] cites which application number(s) it draws from - only use numbers "
        f"from 1 to {application_count}, and cite more than one only if more than one application "
        "genuinely supports that specific opportunity."
    )

_TIER_LINE_RE = re.compile(r"^\s*(direct|adjacent|speculative)\s*:\s*(.+)$", re.IGNORECASE)

# Strips purely cosmetic formatting a model might wrap a tier label in - a
# leading list/bullet marker ("1. ", "1) ", "- ", "* ", "• ") and any
# "**" markdown-bold markers anywhere in the line - before _TIER_LINE_RE
# ever sees it. Investigated live (2026-09-03 stress-testing pass, see
# docs/superpowers/specs/2026-09-03-opportunities-synthesis-design.md):
# 3/3 locally-tested models (qwen2.5:3b, phi3:mini, qwen2.5-coder:7b)
# consistently wrote plain "Direct: ..." with no markdown or numbering, so
# this is forward-hardening against a model swap, not a fix for an
# observed failure - every case this touches previously failed CLOSED
# (missing-tier ValueError), never produced wrong data. Formatting-only:
# does not change what counts as a valid tier word, a valid citation, or
# a valid range - it only changes what counts as the start of a line.
_LEADING_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)")


def _normalize_tier_line(line: str) -> str:
    return _LEADING_LIST_MARKER_RE.sub("", line).replace("**", "")

# Matches a single-number bracket like "[3]" - used only to escape a
# literal citation-shaped substring already present in an application's
# own text (e.g. a copied bibliography reference) before it's numbered
# into the prompt, so extraction can't mistake it for a marker the model
# itself emitted. Deliberately narrower than _CITATION_GROUP_RE below.
# Item 8 of the assessment hardening list (systematic cross-model
# verification, not just the default model): a structurally valid
# response can still be semantically empty. Verified live and repeatedly
# reproducible - NOT a one-off fluke: the default model (qwen2.5:3b),
# asked to synthesize opportunities from a single terse application, wrote
# bare category words as its "opportunity" text on 4/6 repeated calls at
# this module's own temperature=0.2 ("Direct: Evaluate [1]", "Adjacent:
# Compare [1]", "Speculative: Scale [1]"; also seen: "Metrics", "Benchmark
# Suite") - each one a syntactically perfect line (real tier, real
# citation, non-empty text) that parse_response's other checks all
# accept, yet none of these describe an actual product/technology
# concept. Cross-checked against the other 3 locally-available models on
# the identical prompt: phi3:mini and qwen2.5-coder:7b (both plain and
# the q3_K_M quant) never produced this pattern across repeated runs,
# always writing a genuine multi-concept phrase. A word-count minimum
# would incorrectly reject a genuinely fine single-token product name
# (e.g. "HealthWellnessPlatform", seen from the same default model on a
# different case) that just happens to be squished with no spaces, so
# this checks character length instead: every degenerate case observed
# was <=15 characters ("Evaluate"=8, "Metrics"=7, "Compare"=7, "Scale"=5,
# "Benchmark Suite"=15 - itself little more than a bare category name),
# while every genuine opportunity text collected across all 4 models
# (12+ real synthesis calls) was >=18 characters. 16 sits in that gap.
MIN_OPPORTUNITY_TEXT_LENGTH = 16

_SINGLE_BRACKET_NUMBER_RE = re.compile(r"\[(\d+)\]")

# Matches one bracket group containing one or more digits, comma-and/or
# space-separated - e.g. "[3]", "[3,4]", "[3, 4, 5]". Found live against
# the real model (qwen2.5:3b): it writes multi-citations BOTH as separate
# brackets ("[1][2]", what the prompt's own example shows) AND as a
# comma-separated list inside one bracket ("[1,2]", "[1, 3, 5]") - the
# first version of this parser only recognized the first style, so a line
# citing multiple applications the second way silently parsed as having
# NO citation at all and failed validation on every retry.
_CITATION_GROUP_RE = re.compile(r"\[([\d,\s]+)\]")


def _escape_bracketed_numbers(text: str) -> str:
    """Same guard as qa/summarize.py's _escape_bracketed_numbers: neutralizes
    a literal [n] substring inside application text so citation extraction
    can't mistake it for a citation marker the model itself emitted."""
    return _SINGLE_BRACKET_NUMBER_RE.sub(lambda m: f"({m.group(1)})", text)


def _extract_citation_numbers(text: str) -> list[int]:
    """Every citation number in `text`, in order of first appearance,
    deduplicated - handling both "[1][2]" and "[1,2]"/"[1, 2]" bracket
    styles (see _CITATION_GROUP_RE)."""
    numbers: list[int] = []
    for group in _CITATION_GROUP_RE.finditer(text):
        for piece in group.group(1).split(","):
            piece = piece.strip()
            if not piece:
                continue
            n = int(piece)
            if n not in numbers:
                numbers.append(n)
    return numbers


def build_prompt(idea_text: str, applications: list[SourceApplication]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the Ollama chat call.
    idea_text (2026-09-04): see _build_system_prompt's own docstring for
    the real enumeration-drift failure this fixes - without the idea in
    the prompt, the model has no way to tell a relevant listed example
    from an unrelated one in the same application's own text."""
    numbered = "\n".join(
        f'[{i}] "{_escape_bracketed_numbers(app.application)}" — {app.source_paper}'
        for i, app in enumerate(applications, start=1)
    )
    user_prompt = f'Idea: "{_escape_bracketed_numbers(idea_text)}"\n\nApplications:\n{numbered}'
    return _build_system_prompt(len(applications)), user_prompt


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
        line = _normalize_tier_line(raw_line)
        match = _TIER_LINE_RE.match(line)
        if not match:
            continue
        tier = match.group(1).lower()
        assert tier in _TIERS
        body = match.group(2).strip()

        citations = _extract_citation_numbers(body)
        for n in citations:
            if n < 1 or n > application_count:
                raise ValueError(f"citation [{n}] is out of range for {application_count} applications")
        if not citations:
            raise ValueError(f"{tier} opportunity has no citation")

        opportunity_text = _CITATION_GROUP_RE.sub("", body).strip()
        if not opportunity_text:
            raise ValueError(f"{tier} opportunity has no text beyond its citation")
        if len(opportunity_text) < MIN_OPPORTUNITY_TEXT_LENGTH:
            raise ValueError(
                f"{tier} opportunity text {opportunity_text!r} is too short to be a real "
                f"opportunity (<{MIN_OPPORTUNITY_TEXT_LENGTH} characters)"
            )

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


def to_persisted_opportunities(
    applications: list[SourceApplication], result: SynthesisResult
) -> tuple[list[dict], set[str]]:
    """Shapes a SynthesisResult into the JSON list persisted as
    ResearchAssessment.potential_opportunities, plus the set of cited
    evidence_id strings to link (role="opportunity"). Shared by both the
    on-demand endpoint (api/assessment_routes.py's
    synthesize_assessment_opportunities) and build_assessment()'s inline
    path (assessment/build.py), so the persisted shape can't drift between
    the two call sites."""
    opportunities_json = [
        {
            "tier": opp.tier,
            "opportunity": opp.opportunity,
            "source_applications": [
                {
                    "application": applications[i - 1].application,
                    "paper_id": applications[i - 1].paper_id,
                    "paper_title": applications[i - 1].source_paper,
                }
                for i in opp.source_application_indices
            ],
        }
        for opp in result.opportunities
    ]
    cited_evidence_ids = {
        applications[i - 1].evidence_id
        for opp in result.opportunities
        for i in opp.source_application_indices
        if applications[i - 1].evidence_id is not None
    }
    return opportunities_json, cited_evidence_ids


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
    model = os.environ.get("OLLAMA_MODEL", "phi3:mini")

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


def synthesize_opportunities(idea_text: str, applications: list[SourceApplication]) -> SynthesisResult:
    """Calls the local Ollama model to synthesize three grounded product
    opportunities from the given applications, anchored to idea_text (see
    build_prompt's own docstring for why the model needs the idea, not
    just the applications). Retries once on an unreachable model or an
    invalid/incomplete response, then raises OpportunitySynthesisUnavailable
    - never returns a result whose citations weren't checked against the
    given applications."""
    if not ollama_enabled():
        raise OpportunitySynthesisUnavailable("local LLM opportunity synthesis is not enabled")
    if not applications:
        raise OpportunitySynthesisUnavailable("no applications to ground opportunity synthesis in")

    system_prompt, user_prompt = build_prompt(idea_text, applications)
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
