"""Evidence-grounded technical feasibility assessment (blueprint Sec 2A/20).

Deterministic and explainable, no LLM. Sec 20's feasibility dimensions
(deployment feasibility, compute requirements, data availability,
reproducibility, ...) don't map to a dedicated extraction field the way
research_gap/applications did, so this grounds the level in a narrower,
honest proxy instead: how much documented methodology/data a genuinely
CLOSE neighborhood provides, via the method/dataset claims those papers
already have (Sec 28/29). This never claims the idea WILL work or IS
feasible in an engineering sense - only that documented technical
grounding exists (or doesn't) in the retrieved literature. A real
feasibility judgment needs more than literature text; this is a
literature-grounded signal toward one, not a substitute for it.

Distance-gated tighter than novelty/gap/applications' RELEVANCE_DISTANCE
(0.65): checked live against the real corpus, that looser gate made
"high" fire for every idea tried, including a deliberately absurd one -
almost any paper has SOME method/dataset claim, so "2+ relevant papers
have one" was really just measuring "2+ topically-adjacent papers exist,"
not real feasibility grounding. Reusing novelty.py's tighter
NEAR_DISTANCE (0.35, its own "genuinely close match" threshold) instead
requires papers to be actually close, not merely topically adjacent,
before they can corroborate feasibility.

Level is corroboration-count based, same reasoning as the rest of this
package preferring multi-source agreement over a single data point:
- not_assessed: no genuinely close paper has a real (non-stub) method or
  dataset claim to ground a judgment in.
- medium: exactly one such paper does - single-source grounding.
- high: two or more DISTINCT such papers do - multiple independent
  sources document applicable methodology/data.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.assessment.novelty import NEAR_DISTANCE as RELEVANCE_DISTANCE
from researchbridge.db.models import Evidence, ExtractedClaim, Paper

PaperWithDistance = tuple[uuid.UUID, float]
FEASIBILITY_CLAIM_TYPES = ("method", "dataset")


@dataclass
class FeasibilityResult:
    level: str  # high | medium | not_assessed
    reasoning: str
    evidence_ids: list[uuid.UUID]


def assess_technical_feasibility(session: Session, papers_by_distance: list[PaperWithDistance]) -> FeasibilityResult:
    """papers_by_distance: every retrieved paper id paired with its cosine
    distance, nearest-first (the order search_by_text already returns)."""
    relevant_paper_ids = [paper_id for paper_id, distance in papers_by_distance if distance <= RELEVANCE_DISTANCE]
    if not relevant_paper_ids:
        return FeasibilityResult(
            level="not_assessed",
            reasoning=(
                "No retrieved paper was close enough in meaning to ground a feasibility "
                "judgment in, so technical feasibility cannot be assessed from the literature."
            ),
            evidence_ids=[],
        )

    rows = session.execute(
        select(ExtractedClaim.paper_id, ExtractedClaim.evidence_id, Paper.title)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .join(Paper, Paper.id == ExtractedClaim.paper_id)
        .where(
            ExtractedClaim.paper_id.in_(relevant_paper_ids),
            ExtractedClaim.claim_type.in_(FEASIBILITY_CLAIM_TYPES),
            Evidence.extraction_method != "stub",
        )
    ).all()
    if not rows:
        return FeasibilityResult(
            level="not_assessed",
            reasoning=(
                "No relevant retrieved paper had a documented method or dataset to ground "
                "a feasibility judgment in, so technical feasibility cannot be assessed from "
                "the literature alone."
            ),
            evidence_ids=[],
        )

    titles_by_paper: dict[uuid.UUID, str] = {}
    evidence_ids: list[uuid.UUID] = []
    for paper_id, evidence_id, title in rows:
        titles_by_paper[paper_id] = title
        evidence_ids.append(evidence_id)

    distinct_papers = len(titles_by_paper)
    if distinct_papers == 1:
        (title,) = titles_by_paper.values()
        level = "medium"
        reasoning = (
            f'One relevant retrieved paper, "{title}", documents a method or dataset '
            f"applicable to this idea - some technical grounding exists in the literature, "
            f"but from a single source only."
        )
    else:
        level = "high"
        reasoning = (
            f"{distinct_papers} relevant retrieved papers document methods or datasets "
            f"applicable to this idea, suggesting demonstrated technical grounding from "
            f"multiple independent sources in the literature. This reflects documented "
            f"prior work, not a guarantee of success for this specific idea."
        )

    return FeasibilityResult(level=level, reasoning=reasoning, evidence_ids=evidence_ids)
