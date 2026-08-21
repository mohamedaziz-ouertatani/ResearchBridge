"""Turns Paper ORM rows into PaperSummary responses.

Authors and categories are read from the relational tables (authors /
paper_authors / paper_categories) rather than raw_metadata, so the API
stays correct for any future source whose connector doesn't happen to
stash the same keys in raw_metadata.

Lookups are batched per request (one query for all authors, one for all
categories) to avoid an N+1 across a page of papers.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.api.schemas import (
    AssessmentEvidenceOut,
    CandidateGapOut,
    ExtractedClaimOut,
    GapEvidenceOut,
    PaperSummary,
)
from researchbridge.db.models import (
    Author,
    CandidateGap,
    CandidateGapEvidence,
    Evidence,
    ExtractedClaim,
    Paper,
    PaperAuthor,
    PaperCategory,
    ResearchAssessmentEvidence,
)
from researchbridge.extraction.base import CLAIM_TYPE_ORDER


def to_summaries(session: Session, papers: Sequence[Paper]) -> list[PaperSummary]:
    if not papers:
        return []

    paper_ids = [p.id for p in papers]
    authors_by_paper = _authors_by_paper(session, paper_ids)
    categories_by_paper = _categories_by_paper(session, paper_ids)

    return [
        PaperSummary(
            id=paper.id,
            source=paper.source,
            source_id=paper.source_id,
            title=paper.title,
            abstract=paper.abstract,
            publication_date=paper.publication_date,
            url=paper.url,
            primary_category=(paper.raw_metadata or {}).get("primary_category"),
            categories=categories_by_paper.get(paper.id, []),
            authors=authors_by_paper.get(paper.id, []),
            excluded_at=paper.excluded_at,
        )
        for paper in papers
    ]


def to_summary(session: Session, paper: Paper) -> PaperSummary:
    return to_summaries(session, [paper])[0]


def to_claims(session: Session, paper_id: uuid.UUID) -> list[ExtractedClaimOut]:
    """Extracted claims for one paper, in Sec 25/28 field order.

    Excludes extraction_method="stub": those rows are synthetic
    placeholder data (see the Evidence docstring in db/models.py), never
    real content this API should hand back to a caller.
    """
    rows = session.execute(
        select(ExtractedClaim, Evidence)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .where(ExtractedClaim.paper_id == paper_id, Evidence.extraction_method != "stub")
    ).all()

    order = {claim_type: i for i, claim_type in enumerate(CLAIM_TYPE_ORDER)}
    rows.sort(key=lambda row: order.get(row[0].claim_type, len(order)))

    return [
        ExtractedClaimOut(
            claim_type=claim.claim_type,
            text=claim.text,
            confidence=claim.confidence,
            section=evidence.section,
            extraction_method=evidence.extraction_method,
        )
        for claim, evidence in rows
    ]


def to_gaps(session: Session, gaps: Sequence[CandidateGap]) -> list[CandidateGapOut]:
    """Candidate gaps with their seed paper title and contributing evidence.

    Evidence rows are joined through CandidateGapEvidence -> Evidence ->
    Paper so a reviewer can see exactly which papers/passages a gap is
    grounded in, batched to avoid an N+1 across a page of gaps.
    """
    if not gaps:
        return []

    gap_ids = [g.id for g in gaps]
    seed_ids = {g.seed_paper_id for g in gaps}

    titles_by_paper_id = _titles_by_paper(session, seed_ids)
    evidence_by_gap = _evidence_by_gap(session, gap_ids)

    return [
        CandidateGapOut(
            id=gap.id,
            seed_paper_id=gap.seed_paper_id,
            seed_paper_title=titles_by_paper_id.get(gap.seed_paper_id, ""),
            observation=gap.observation,
            gap_type=gap.gap_type,
            status=gap.status,
            contributing_paper_count=gap.contributing_paper_count,
            similarity_threshold=gap.similarity_threshold,
            detection_method=gap.detection_method,
            review_note=gap.review_note,
            evidence=evidence_by_gap.get(gap.id, []),
        )
        for gap in gaps
    ]


def to_assessment_evidence(session: Session, assessment_id: uuid.UUID) -> list[AssessmentEvidenceOut]:
    """The real quoted passages backing one assessment's report fields.

    Same shape and reasoning as _evidence_by_gap: an assessment field is a
    lightweight summary, and this is what makes it inspectable - which
    paper, which exact passage, backing which part of the report (Sec 15/17).
    Ordered by role so the report can group without re-sorting.
    """
    rows = session.execute(
        select(ResearchAssessmentEvidence.role, Evidence, Paper.title)
        .join(Evidence, Evidence.id == ResearchAssessmentEvidence.evidence_id)
        .join(Paper, Paper.id == Evidence.paper_id)
        .where(ResearchAssessmentEvidence.research_assessment_id == assessment_id)
        .order_by(ResearchAssessmentEvidence.role)
    ).all()

    return [
        AssessmentEvidenceOut(
            role=role, paper_id=evidence.paper_id, paper_title=title, text=evidence.text, section=evidence.section
        )
        for role, evidence, title in rows
    ]


def _titles_by_paper(session: Session, paper_ids: set[uuid.UUID]) -> dict[uuid.UUID, str]:
    rows = session.execute(select(Paper.id, Paper.title).where(Paper.id.in_(paper_ids))).all()
    return dict(rows)


def _evidence_by_gap(session: Session, gap_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[GapEvidenceOut]]:
    rows = session.execute(
        select(CandidateGapEvidence.candidate_gap_id, Evidence, Paper.title)
        .join(Evidence, Evidence.id == CandidateGapEvidence.evidence_id)
        .join(Paper, Paper.id == Evidence.paper_id)
        .where(CandidateGapEvidence.candidate_gap_id.in_(gap_ids))
    ).all()

    result: dict[uuid.UUID, list[GapEvidenceOut]] = defaultdict(list)
    for gap_id, evidence, paper_title in rows:
        result[gap_id].append(
            GapEvidenceOut(
                paper_id=evidence.paper_id,
                paper_title=paper_title,
                text=evidence.text,
                section=evidence.section,
            )
        )
    return result


def _authors_by_paper(session: Session, paper_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[str]]:
    rows = session.execute(
        select(PaperAuthor.paper_id, Author.name, PaperAuthor.author_order)
        .join(Author, Author.id == PaperAuthor.author_id)
        .where(PaperAuthor.paper_id.in_(paper_ids))
        .order_by(PaperAuthor.paper_id, PaperAuthor.author_order)
    ).all()

    result: dict[uuid.UUID, list[str]] = defaultdict(list)
    for paper_id, name, _order in rows:
        result[paper_id].append(name)
    return result


def _categories_by_paper(session: Session, paper_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[str]]:
    rows = session.execute(
        select(PaperCategory.paper_id, PaperCategory.category)
        .where(PaperCategory.paper_id.in_(paper_ids))
        .order_by(PaperCategory.paper_id, PaperCategory.category)
    ).all()

    result: dict[uuid.UUID, list[str]] = defaultdict(list)
    for paper_id, category in rows:
        result[paper_id].append(category)
    return result
