"""Input-specific research-gap assessment (blueprint Sec 2A/32).

Scoped to one ResearchAssessment's own retrieved-paper neighborhood, not
the whole corpus - deliberately distinct from gaps/detect.py's per-seed-
paper corpus similarity search. This assessment already has its own
retrieved papers (with distances) from search_by_text(), so there's no
seed paper to expand a neighborhood from.

Relevance gate: a retrieved paper only participates if its distance is
within RELEVANCE_DISTANCE (reusing novelty.py's own calibrated
FAR_DISTANCE - the same "genuinely related vs. off-topic" boundary,
rather than inventing a second threshold). Checked live against the real
corpus: without this gate, an idea unrelated to anything in the corpus
(e.g. "cats playing chess with quantum computers") still surfaced an
explicit gap from whichever barely-related paper happened to rank in the
top-k - technically "evidence", but not actually relevant evidence. The
gate turns that into NULL/NOT_ASSESSED instead, per Sec 22.

Priority order, each labeled distinctly so a human reader never confuses
inference with a stated fact (Sec 34):

1. An already-reviewed ("approved") candidate_gaps row whose seed_paper_id
   is a relevant retrieved paper - reused outright. Still labeled as
   inference (candidate_gaps.gap_type is always "inference").
2. An explicit, author-stated research_gap claim on a relevant retrieved
   paper (Sec 32's "Explicit gaps" - the research_gap extraction field).
   The most direct source; never described as inferred. Checked
   nearest-first, not via an unordered SQL IN query - the DB does not
   preserve retrieval relevance order on its own.
3. A fresh cross-paper cluster over the relevant neighborhood's
   limitations/research_gap claims, reusing gaps/cluster.py's
   find_recurring_patterns directly (not gaps/detect.py's seed-paper
   machinery, since the neighborhood is already this assessment's own
   retrieval). Explicitly labeled as inference, matching gaps/detect.py's
   own phrasing.
4. Nothing found -> NULL/NOT_ASSESSED. Insufficient (or irrelevant)
   evidence stays insufficient; nothing here is ever fabricated.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.assessment.novelty import FAR_DISTANCE as RELEVANCE_DISTANCE
from researchbridge.db.models import CandidateGap, CandidateGapEvidence, Evidence, ExtractedClaim, Paper
from researchbridge.embedding.base import Embedder
from researchbridge.gaps.cluster import (
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    RELEVANT_CLAIM_TYPES,
    ClaimRecord,
    find_recurring_patterns,
)

PaperWithDistance = tuple[uuid.UUID, float]


@dataclass
class GapAssessmentResult:
    source: str | None  # "reused_candidate_gap" | "input_specific" | None
    text: str | None
    candidate_gap_id: uuid.UUID | None
    evidence_ids: list[uuid.UUID]


_NONE_RESULT = GapAssessmentResult(source=None, text=None, candidate_gap_id=None, evidence_ids=[])


def assess_research_gap(
    session: Session,
    papers_by_distance: list[PaperWithDistance],
    embedder: Embedder,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> GapAssessmentResult:
    """papers_by_distance: every retrieved paper id paired with its cosine
    distance, nearest-first (the order search_by_text already returns)."""
    relevant_paper_ids = [paper_id for paper_id, distance in papers_by_distance if distance <= RELEVANCE_DISTANCE]
    if not relevant_paper_ids:
        return _NONE_RESULT

    reused = _reuse_approved_candidate_gap(session, relevant_paper_ids)
    if reused is not None:
        return reused

    explicit = _explicit_research_gap_claim(session, relevant_paper_ids)
    if explicit is not None:
        return explicit

    return _inferred_cross_paper_gap(session, relevant_paper_ids, embedder, min_cluster_size, similarity_threshold)


def _reuse_approved_candidate_gap(
    session: Session, relevant_paper_ids: list[uuid.UUID]
) -> GapAssessmentResult | None:
    gap = session.execute(
        select(CandidateGap)
        .where(CandidateGap.seed_paper_id.in_(relevant_paper_ids), CandidateGap.status == "approved")
        .order_by(CandidateGap.contributing_paper_count.desc())
    ).scalars().first()
    if gap is None:
        return None

    evidence_ids = list(
        session.execute(
            select(CandidateGapEvidence.evidence_id).where(CandidateGapEvidence.candidate_gap_id == gap.id)
        ).scalars()
    )
    return GapAssessmentResult(
        source="reused_candidate_gap", text=gap.observation, candidate_gap_id=gap.id, evidence_ids=evidence_ids
    )


def _explicit_research_gap_claim(
    session: Session, relevant_paper_ids: list[uuid.UUID]
) -> GapAssessmentResult | None:
    """Checks papers in the given order (nearest-first, per the caller's own
    retrieval ranking) and returns the first explicit claim found - NOT an
    arbitrary one. An unordered `paper_id IN (...)` query would let the DB
    return any matching row first, regardless of how relevant that paper
    actually is to the input; that defeats the point of scoping this to
    the assessment's own neighborhood."""
    rows = session.execute(
        select(ExtractedClaim.paper_id, ExtractedClaim.text, ExtractedClaim.evidence_id, Paper.title)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .join(Paper, Paper.id == ExtractedClaim.paper_id)
        .where(
            ExtractedClaim.paper_id.in_(relevant_paper_ids),
            ExtractedClaim.claim_type == "research_gap",
            Evidence.extraction_method != "stub",
        )
    ).all()
    by_paper_id = {row.paper_id: (row.text, row.evidence_id, row.title) for row in rows}

    for paper_id in relevant_paper_ids:
        if paper_id in by_paper_id:
            text, evidence_id, paper_title = by_paper_id[paper_id]
            return GapAssessmentResult(
                source="input_specific",
                text=f'Explicitly stated in "{paper_title}": "{text}"',
                candidate_gap_id=None,
                evidence_ids=[evidence_id],
            )
    return None


def _inferred_cross_paper_gap(
    session: Session,
    relevant_paper_ids: list[uuid.UUID],
    embedder: Embedder,
    min_cluster_size: int,
    similarity_threshold: float,
) -> GapAssessmentResult:
    rows = session.execute(
        select(ExtractedClaim.paper_id, ExtractedClaim.evidence_id, ExtractedClaim.text)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .where(
            ExtractedClaim.paper_id.in_(relevant_paper_ids),
            ExtractedClaim.claim_type.in_(RELEVANT_CLAIM_TYPES),
            Evidence.extraction_method != "stub",
        )
    ).all()
    claims = [ClaimRecord(paper_id=paper_id, evidence_id=evidence_id, text=text) for paper_id, evidence_id, text in rows]

    clusters = find_recurring_patterns(claims, embedder, min_cluster_size, similarity_threshold)
    if not clusters:
        return _NONE_RESULT

    top = clusters[0]  # already sorted by contributing_paper_count, descending
    text = (
        f"Recurring pattern across {top.contributing_paper_count} related papers "
        f'(inference, not stated by any single author): "{top.representative_text}"'
    )
    evidence_ids = [member.evidence_id for member in top.members]
    return GapAssessmentResult(source="input_specific", text=text, candidate_gap_id=None, evidence_ids=evidence_ids)
