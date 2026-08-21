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

from researchbridge.api.schemas import ExtractedClaimOut, PaperSummary
from researchbridge.db.models import Author, Evidence, ExtractedClaim, Paper, PaperAuthor, PaperCategory
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
