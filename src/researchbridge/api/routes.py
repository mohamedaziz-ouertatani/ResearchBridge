"""Corpus explorer endpoints: browse, filter, and semantic search."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_embedder, get_session
from researchbridge.api.schemas import (
    CitationEdgeOut,
    CitationGraphOut,
    CitationNodeOut,
    CorpusStats,
    ExtractedClaimOut,
    PaperPage,
    PaperSummary,
    SearchHit,
    TrendsOut,
)
from researchbridge.api.serializers import to_claims, to_summaries, to_summary
from researchbridge.db.models import (
    Author,
    Embedding,
    Evidence,
    ExtractedClaim,
    Paper,
    PaperAuthor,
    PaperCategory,
    PaperCitation,
)
from researchbridge.embedding.base import Embedder
from researchbridge.embedding.pipeline import EMBEDDING_TYPE
from researchbridge.embedding.search import find_similar_to_paper, search_by_text

router = APIRouter(prefix="/api")

MAX_LIMIT = 100
MAX_TOP_K = 50

SORT_ORDERINGS = {
    "date_desc": (Paper.publication_date.desc().nullslast(), Paper.id),
    "date_asc": (Paper.publication_date.asc().nullslast(), Paper.id),
    "title_asc": (func.lower(Paper.title).asc(), Paper.id),
}


@router.get("/papers", response_model=PaperPage)
def list_papers(
    session: Session = Depends(get_session),
    limit: int = Query(20, ge=1, le=MAX_LIMIT),
    offset: int = Query(0, ge=0),
    year: int | None = Query(None, description="Filter by publication year"),
    category: str | None = Query(None, description="Filter by an arXiv category, e.g. cs.LG"),
    source: str | None = Query(None, description="Filter by ingestion source, e.g. arxiv"),
    q: str | None = Query(None, description="Case-insensitive substring match on title"),
    include_excluded: bool = Query(False, description="Include papers excluded from curation"),
    sort: str = Query("date_desc", pattern="^(date_desc|date_asc|title_asc)$"),
) -> PaperPage:
    """Browse the corpus. `q` is a plain title filter - for meaning-based search use /api/search."""
    query = select(Paper)
    count_query = select(func.count(Paper.id))

    if not include_excluded:
        query = query.where(Paper.excluded_at.is_(None))
        count_query = count_query.where(Paper.excluded_at.is_(None))

    for condition in _filters(year=year, category=category, source=source, q=q):
        query = query.where(condition)
        count_query = count_query.where(condition)

    total = session.execute(count_query).scalar_one()
    papers = list(
        session.execute(query.order_by(*SORT_ORDERINGS[sort]).limit(limit).offset(offset)).scalars()
    )

    return PaperPage(items=to_summaries(session, papers), total=total, limit=limit, offset=offset)


def _filters(year: int | None, category: str | None, source: str | None, q: str | None) -> list:
    conditions = []
    if year is not None:
        conditions.append(func.extract("year", Paper.publication_date) == year)
    if category:
        conditions.append(
            Paper.id.in_(select(PaperCategory.paper_id).where(PaperCategory.category == category))
        )
    if source:
        conditions.append(Paper.source == source)
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


@router.get("/papers/{paper_id}/citations", response_model=CitationGraphOut)
def paper_citations(paper_id: uuid.UUID, session: Session = Depends(get_session)) -> CitationGraphOut:
    paper = session.get(Paper, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail=f"No paper with id {paper_id}")
    return _build_citation_graph(session, paper)


def _build_citation_graph(session: Session, paper: Paper) -> CitationGraphOut:
    cited_by_rows = session.execute(
        select(PaperCitation.citing_paper_id, Paper.title)
        .join(Paper, Paper.id == PaperCitation.citing_paper_id)
        .where(PaperCitation.cited_paper_id == paper.id)
    ).all()
    cites_rows = session.execute(
        select(PaperCitation.cited_paper_id, Paper.title)
        .join(Paper, Paper.id == PaperCitation.cited_paper_id)
        .where(PaperCitation.citing_paper_id == paper.id)
    ).all()

    nodes = [CitationNodeOut(id=str(paper.id), type="center", title=paper.title)]
    edges = []
    for citing_id, title in cited_by_rows:
        nodes.append(CitationNodeOut(id=str(citing_id), type="paper", title=title))
        edges.append(CitationEdgeOut(source=str(citing_id), target=str(paper.id), direction="cited_by"))
    for cited_id, title in cites_rows:
        nodes.append(CitationNodeOut(id=str(cited_id), type="paper", title=title))
        edges.append(CitationEdgeOut(source=str(paper.id), target=str(cited_id), direction="cites"))

    return CitationGraphOut(nodes=nodes, edges=edges)


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
def corpus_stats(
    session: Session = Depends(get_session),
    year: int | None = Query(None, description="Scope totals/categories to one publication year"),
) -> CorpusStats:
    """Counts here always exclude curated-out papers (Paper.excluded_at set) -
    matching /api/papers' default so this endpoint's totals and the corpus
    page's default listing never disagree (see corpus-curation follow-up).

    `year`, when given, scopes every field below except papers_by_year to
    that publication year - papers_by_year is always computed unfiltered so
    the YearStrip driving this filter keeps showing every year to navigate
    to, not just the one currently selected.
    """
    not_excluded = Paper.excluded_at.is_(None)
    year_scope = func.extract("year", Paper.publication_date) == year if year is not None else True

    total_papers = session.execute(
        select(func.count(Paper.id)).where(not_excluded, year_scope)
    ).scalar_one()
    total_authors = session.execute(
        select(func.count(func.distinct(PaperAuthor.author_id)))
        .join(Paper, Paper.id == PaperAuthor.paper_id)
        .where(not_excluded, year_scope)
    ).scalar_one()
    embedded = session.execute(
        select(func.count(func.distinct(Embedding.paper_id)))
        .join(Paper, Paper.id == Embedding.paper_id)
        .where(Embedding.embedding_type == EMBEDDING_TYPE, not_excluded, year_scope)
    ).scalar_one()
    with_claims = session.execute(
        select(func.count(func.distinct(ExtractedClaim.paper_id)))
        .join(Paper, Paper.id == ExtractedClaim.paper_id)
        .where(not_excluded, year_scope)
    ).scalar_one()

    year_rows = session.execute(
        select(func.extract("year", Paper.publication_date), func.count(Paper.id))
        .where(Paper.publication_date.isnot(None), not_excluded)
        .group_by(func.extract("year", Paper.publication_date))
    ).all()

    category_rows = session.execute(
        select(PaperCategory.category, func.count(func.distinct(PaperCategory.paper_id)))
        .join(Paper, Paper.id == PaperCategory.paper_id)
        .where(not_excluded, year_scope)
        .group_by(PaperCategory.category)
        .order_by(func.count(func.distinct(PaperCategory.paper_id)).desc())
        .limit(20)
    ).all()

    source_rows = session.execute(
        select(Paper.source, func.count(Paper.id)).where(not_excluded, year_scope).group_by(Paper.source)
    ).all()

    return CorpusStats(
        total_papers=total_papers,
        total_authors=total_authors,
        embedded_papers=embedded,
        papers_with_claims=with_claims,
        papers_by_year={int(year): count for year, count in year_rows},
        papers_by_category={category: count for category, count in category_rows},
        papers_by_source=dict(source_rows),
    )


@router.get("/trends", response_model=TrendsOut)
def trends(
    category: str = Query(..., description="An ingested category, e.g. cs.LG"),
    session: Session = Depends(get_session),
) -> TrendsOut:
    """Real counts of already-extracted claims per year within one category -
    not an inferred pattern. See TrendsOut for the response shape."""
    rows = session.execute(
        select(
            func.extract("year", Paper.publication_date),
            ExtractedClaim.claim_type,
            func.count(ExtractedClaim.id),
        )
        .join(PaperCategory, PaperCategory.paper_id == Paper.id)
        .join(ExtractedClaim, ExtractedClaim.paper_id == Paper.id)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .where(
            PaperCategory.category == category,
            Evidence.extraction_method != "stub",
            Paper.publication_date.isnot(None),
            Paper.excluded_at.is_(None),
        )
        .group_by(func.extract("year", Paper.publication_date), ExtractedClaim.claim_type)
    ).all()

    if not rows:
        return TrendsOut(category=category, years=[], series={})

    observed_years = sorted({int(year) for year, _, _ in rows})
    years = list(range(observed_years[0], observed_years[-1] + 1))
    year_index = {year: i for i, year in enumerate(years)}

    series: dict[str, list[int]] = {}
    for year, claim_type, count in rows:
        counts = series.setdefault(claim_type, [0] * len(years))
        counts[year_index[int(year)]] = count

    return TrendsOut(category=category, years=years, series=series)
