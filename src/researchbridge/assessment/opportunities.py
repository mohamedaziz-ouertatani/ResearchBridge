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
the other assess_* functions in this package) purely so that wiring in a
real implementation later is a localized change to this one function,
not a rework of build_assessment()'s call site.
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
