"""Corpus explorer endpoints: browse, filter, and semantic search."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_embedder, get_session
from researchbridge.api.schemas import CorpusStats, ExtractedClaimOut, PaperPage, PaperSummary, SearchHit
from researchbridge.api.serializers import to_claims, to_summaries, to_summary
from researchbridge.db.models import Author, Embedding, Paper, PaperAuthor, PaperCategory
from researchbridge.embedding.base import Embedder
from researchbridge.embedding.pipeline import EMBEDDING_TYPE
from researchbridge.embedding.search import find_similar_to_paper, search_by_text

router = APIRouter(prefix="/api")

MAX_LIMIT = 100
MAX_TOP_K = 50


@router.get("/papers", response_model=PaperPage)
def list_papers(
    session: Session = Depends(get_session),
    limit: int = Query(20, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    year: int | None = Query(None, description="Filter by publication year"),
    category: str | None = Query(None, description="Filter by an arXiv category, e.g. cs.LG"),
    q: str | None = Query(None, description="Case-insensitive substring match on title"),
    include_excluded: bool = Query(False, description="Include papers excluded from curation"),
) -> PaperPage:
    """Browse the corpus. `q` is a plain title filter - for meaning-based search use /api/search."""
    query = select(Paper)
    count_query = select(func.count(Paper.id))

    if not include_excluded:
        query = query.where(Paper.excluded_at.is_(None))
        count_query = count_query.where(Paper.excluded_at.is_(None))

    for condition in _filters(year=year, category=category, q=q):
        query = query.where(condition)
        count_query = count_query.where(condition)

    total = session.execute(count_query).scalar_one()
    papers = list(
        session.execute(query.order_by(Paper.publication_date.desc().nullslast(), Paper.id).limit(limit).offset(offset))
        .scalars()
    )

    return PaperPage(items=to_summaries(session, papers), total=total, limit=limit, offset=offset)


def _filters(year: int | None, category: str | None, q: str | None) -> list:
    conditions = []
    if year is not None:
        conditions.append(func.extract("year", Paper.publication_date) == year)
    if category:
        conditions.append(
            Paper.id.in_(select(PaperCategory.paper_id).where(PaperCategory.category == category))
        )
    if q:
        conditions.append(Paper.title.ilike(f"%{q}%"))
    return conditions


@router.get("/papers/{paper_id}", response_model=PaperSummary)
def get_paper(paper_id: uuid.UUID, session: Session = Depends(get_session)) -> PaperSummary:
    paper = session.get(Paper, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"No paper with id {paper_id}")
    return to_summary(session, paper)


@router.get("/papers/{paper_id}/claims", response_model=list[ExtractedClaimOut])
def paper_claims(paper_id: uuid.UUID, session: Session = Depends(get_session)) -> list[ExtractedClaimOut]:
    """Automatically extracted claims for a paper - not human-verified.

    Empty is a normal response (no abstract, or extraction hasn't run for
    this paper yet), not an error - unlike a 404, which means the paper
    itself doesn't exist.
    """
    if session.get(Paper, paper_id) is None:
        raise HTTPException(status_code=404, detail=f"No paper with id {paper_id}")
    return to_claims(session, paper_id)


@router.get("/papers/{paper_id}/similar", response_model=list[SearchHit])
def similar_papers(
    paper_id: uuid.UUID,
    session: Session = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
    top_k: int = Query(10, ge=1, le=MAX_TOP_K),
) -> list[SearchHit]:
    if session.get(Paper, paper_id) is None:
        raise HTTPException(status_code=404, detail=f"No paper with id {paper_id}")

    try:
        results = find_similar_to_paper(session, paper_id, embedder.model_name, top_k)
    except ValueError as exc:
        # paper exists but was never embedded - a real state, not a server fault
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return _to_hits(session, results)


@router.get("/search", response_model=list[SearchHit])
def search(
    q: str = Query(..., min_length=1, description="Free-text query, matched by meaning"),
    session: Session = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
    top_k: int = Query(10, ge=1, le=MAX_TOP_K),
) -> list[SearchHit]:
    return _to_hits(session, search_by_text(session, q, embedder, top_k))


def _to_hits(session: Session, results: list[tuple[Paper, float]]) -> list[SearchHit]:
    summaries = to_summaries(session, [paper for paper, _ in results])
    return [SearchHit(paper=summary, distance=distance) for summary, (_, distance) in zip(summaries, results, strict=True)]


@router.get("/stats", response_model=CorpusStats)
def corpus_stats(session: Session = Depends(get_session)) -> CorpusStats:
    """Counts here always exclude curated-out papers (Paper.excluded_at set) -
    matching /api/papers' default so this endpoint's totals and the corpus
    page's default listing never disagree (see corpus-curation follow-up)."""
    not_excluded = Paper.excluded_at.is_(None)

    total_papers = session.execute(select(func.count(Paper.id)).where(not_excluded)).scalar_one()
    total_authors = session.execute(
        select(func.count(func.distinct(PaperAuthor.author_id)))
        .join(Paper, Paper.id == PaperAuthor.paper_id)
        .where(not_excluded)
    ).scalar_one()
    embedded = session.execute(
        select(func.count(func.distinct(Embedding.paper_id)))
        .join(Paper, Paper.id == Embedding.paper_id)
        .where(Embedding.embedding_type == EMBEDDING_TYPE, not_excluded)
    ).scalar_one()

    year_rows = session.execute(
        select(func.extract("year", Paper.publication_date), func.count(Paper.id))
        .where(Paper.publication_date.isnot(None), not_excluded)
        .group_by(func.extract("year", Paper.publication_date))
    ).all()

    category_rows = session.execute(
        select(PaperCategory.category, func.count(func.distinct(PaperCategory.paper_id)))
        .join(Paper, Paper.id == PaperCategory.paper_id)
        .where(not_excluded)
        .group_by(PaperCategory.category)
        .order_by(func.count(func.distinct(PaperCategory.paper_id)).desc())
        .limit(20)
    ).all()

    return CorpusStats(
        total_papers=total_papers,
        total_authors=total_authors,
        embedded_papers=embedded,
        papers_by_year={int(year): count for year, count in year_rows},
        papers_by_category={category: count for category, count in category_rows},
    )
