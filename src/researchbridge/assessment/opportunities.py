"""Potential Product/Technology Opportunities (blueprint Sec 33/49).

Deliberately NOT implemented yet - left NULL/not_assessed by design, not
by omission. Sec 33's Direct/Adjacent/Speculative opportunity framing
requires genuine synthesis (turning a grounded application into a new
product/market idea) - the same further interpretive leap gaps/cluster.py
already refuses to take for research gaps ("generating that second, more
interpretive leap would mean inventing content beyond what's grounded in
the source papers... left to the human reviewer"). Applying that same
standard here: nothing in this project's toolbox (extractive claims,
embedding similarity, deterministic clustering) can produce a genuine
Direct/Adjacent/Speculative opportunity without either

(a) trivially relabeling an existing application as "Direct" - real, but
    not actually a distinct opportunity; Applications (assess_applications)
    already covers exactly this, or
(b) inventing content via a generative model - out of scope until/unless
    the project deliberately adds an LLM behind the Sec 29 provider
    abstraction and accepts the non-determinism and grounding risk that
    comes with it.

assess_opportunities() exists as a stable interface point (same shape as
the other assess_* functions in this package) so that build_assessment()
never has to special-case this field. Its own output is still always NULL
- this function itself was never given a generative path.

(b) was taken, narrowly and on-demand: see assessment/opportunity_
synthesis.py and docs/superpowers/specs/
2026-09-03-opportunities-synthesis-design.md. Originally deliberately NOT
called from build_assessment() at all, for exactly the invariant-
preservation reason above - kept reachable only via POST /api/assessments
/{id}/opportunities, after the assessment already existed.

2026-09-04: build_assessment() now DOES call it, but only when the
caller explicitly opts in via its enable_llm_stages parameter (see that
function's own docstring) - the API route layer passes True when
ollama_enabled() is true, so every assessment created through the real
API gets synthesis automatically; every other caller (tests, scripts,
benchmarks) keeps today's synchronous/deterministic/no-model-calls
default unless it explicitly asks otherwise. This module's own NULL
output is what build_assessment() falls back to when enable_llm_stages
is False, or when synthesis is enabled but fails (Ollama disabled/
unreachable, or an invalid response after one retry) - opportunity
synthesis fails CLOSED (see its own OpportunitySynthesisUnavailable
docstring), so a failed attempt always lands back on this function's
NULL, never a partial/fabricated result. The on-demand endpoint is
unchanged and still useful for regenerating after the fact.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

PaperWithDistance = tuple[uuid.UUID, float]


@dataclass
class OpportunitiesResult:
    opportunities: list[dict] | None
    evidence_ids: list[uuid.UUID]


_NOT_YET_IMPLEMENTED = OpportunitiesResult(opportunities=None, evidence_ids=[])


def assess_opportunities(session: Session, papers_by_distance: list[PaperWithDistance]) -> OpportunitiesResult:
    """Always returns nothing - see module docstring for why."""
    return _NOT_YET_IMPLEMENTED
