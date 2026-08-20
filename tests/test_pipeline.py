from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import select

from researchbridge.connectors.base import ConnectorFetchResult, NormalizedAuthor, NormalizedPaper
from researchbridge.db.models import Author, IngestionError, IngestionRun, Paper, PaperAuthor, PaperCategory
from researchbridge.ingestion.pipeline import IngestionPipeline


def _paper(
    source_id: str,
    title: str = "A Paper",
    authors: list[NormalizedAuthor] | None = None,
    categories: list[str] | None = None,
    doi: str | None = None,
    raw_metadata: dict | None = None,
) -> NormalizedPaper:
    return NormalizedPaper(
        source="fake",
        source_id=source_id,
        title=title,
        abstract="An abstract.",
        publication_date=date(2026, 1, 1),
        authors=authors or [],
        categories=categories or [],
        doi=doi,
        raw_metadata=raw_metadata or {},
    )


@dataclass
class FakeConnector:
    pages: list[ConnectorFetchResult]
    source_name: str = "fake"

    def fetch(self, resume_state: dict | None) -> ConnectorFetchResult:
        idx = (resume_state or {}).get("page", 0)
        return self.pages[idx]


def _two_page_connector() -> FakeConnector:
    paper_a = _paper("A")
    paper_b = _paper("B")
    paper_a_dup = _paper("A")  # same source_id as paper_a -> should dedupe
    paper_c = _paper("C")
    invalid = _paper("", title="Missing source id")

    page0 = ConnectorFetchResult(papers=[paper_a, paper_b], resume_state={"page": 1}, exhausted=False)
    page1 = ConnectorFetchResult(
        papers=[paper_a_dup, paper_c, invalid], resume_state={"page": 2}, exhausted=True
    )
    return FakeConnector(pages=[page0, page1])


def test_pipeline_runs_all_stages_to_completion(session_factory) -> None:
    connector = _two_page_connector()
    pipeline = IngestionPipeline(connector=connector, session_factory=session_factory)

    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(IngestionRun, run_id)
        assert run.status == "completed"
        assert run.records_fetched == 5
        assert run.records_inserted == 3
        assert run.records_duplicate == 1
        assert run.records_failed == 1
        assert run.resume_state == {"page": 2}

        papers = session.execute(select(Paper.source_id)).scalars().all()
        assert set(papers) == {"A", "B", "C"}

        errors = session.execute(select(IngestionError)).scalars().all()
        assert len(errors) == 1
        assert errors[0].error_type == "validation_error"
        assert errors[0].ingestion_run_id == run.id
        assert errors[0].raw_payload["title"] == "Missing source id"
    finally:
        session.close()


def test_pipeline_pauses_and_resumes_via_resume_state(session_factory) -> None:
    connector = _two_page_connector()
    pipeline = IngestionPipeline(connector=connector, session_factory=session_factory)

    first_run_id = pipeline.run(max_pages=1)

    session = session_factory()
    try:
        first_run = session.get(IngestionRun, first_run_id)
        assert first_run.status == "paused"
        assert first_run.resume_state == {"page": 1}
        assert first_run.records_inserted == 2  # A, B
    finally:
        session.close()

    second_run_id = pipeline.run(resume_state={"page": 1})

    session = session_factory()
    try:
        second_run = session.get(IngestionRun, second_run_id)
        assert second_run.status == "completed"
        assert second_run.records_inserted == 1  # only C; A is now a duplicate of the first run
        assert second_run.records_duplicate == 1

        papers = session.execute(select(Paper.source_id)).scalars().all()
        assert set(papers) == {"A", "B", "C"}
    finally:
        session.close()


def test_duplicate_within_same_page_is_deduplicated(session_factory) -> None:
    paper = _paper("X")
    same_paper_again = _paper("X")
    page = ConnectorFetchResult(papers=[paper, same_paper_again], resume_state={"page": 1}, exhausted=True)
    connector = FakeConnector(pages=[page])
    pipeline = IngestionPipeline(connector=connector, session_factory=session_factory)

    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(IngestionRun, run_id)
        assert run.records_inserted == 1
        assert run.records_duplicate == 1
        count = session.execute(select(Paper).where(Paper.source_id == "X")).scalars().all()
        assert len(count) == 1
    finally:
        session.close()


def test_pipeline_persists_authors_and_categories(session_factory) -> None:
    paper = _paper(
        "P1",
        authors=[NormalizedAuthor(name="Alice Example", order=0), NormalizedAuthor(name="Bob Sample", order=1)],
        categories=["cs.LG", "cs.AI"],
    )
    page = ConnectorFetchResult(papers=[paper], resume_state={"page": 1}, exhausted=True)
    pipeline = IngestionPipeline(connector=FakeConnector(pages=[page]), session_factory=session_factory)

    pipeline.run()

    session = session_factory()
    try:
        paper_row = session.execute(select(Paper).where(Paper.source_id == "P1")).scalar_one()

        links = (
            session.execute(select(PaperAuthor).where(PaperAuthor.paper_id == paper_row.id))
            .scalars()
            .all()
        )
        assert len(links) == 2
        names_by_order = {
            link.author_order: session.get(Author, link.author_id).name for link in links
        }
        assert names_by_order == {0: "Alice Example", 1: "Bob Sample"}

        categories = (
            session.execute(select(PaperCategory).where(PaperCategory.paper_id == paper_row.id))
            .scalars()
            .all()
        )
        assert {c.category for c in categories} == {"cs.LG", "cs.AI"}
        assert all(c.confidence == "high" and c.source == "fake" for c in categories)
    finally:
        session.close()


def test_pipeline_dedupes_authors_by_name_across_papers(session_factory) -> None:
    shared_author = NormalizedAuthor(name="Shared Author", order=0)
    paper1 = _paper("P1", authors=[shared_author])
    paper2 = _paper("P2", authors=[NormalizedAuthor(name="Shared Author", order=0)])
    page = ConnectorFetchResult(papers=[paper1, paper2], resume_state={"page": 1}, exhausted=True)
    pipeline = IngestionPipeline(connector=FakeConnector(pages=[page]), session_factory=session_factory)

    pipeline.run()

    session = session_factory()
    try:
        authors = session.execute(select(Author).where(Author.name == "Shared Author")).scalars().all()
        assert len(authors) == 1  # deduped by exact name match, not one row per paper

        links = session.execute(select(PaperAuthor).where(PaperAuthor.author_id == authors[0].id)).scalars().all()
        assert len(links) == 2  # linked to both papers
    finally:
        session.close()


def test_pipeline_normalizes_doi_but_keeps_original_in_raw_metadata(session_factory) -> None:
    paper = _paper(
        "P1",
        doi="https://doi.org/10.1145/ABC.DEF",
        raw_metadata={"doi": "https://doi.org/10.1145/ABC.DEF"},
    )
    page = ConnectorFetchResult(papers=[paper], resume_state={"page": 1}, exhausted=True)
    pipeline = IngestionPipeline(connector=FakeConnector(pages=[page]), session_factory=session_factory)

    pipeline.run()

    session = session_factory()
    try:
        paper_row = session.execute(select(Paper).where(Paper.source_id == "P1")).scalar_one()
        assert paper_row.doi == "10.1145/abc.def"  # normalized: prefix stripped, lowercased
        assert paper_row.raw_metadata["doi"] == "https://doi.org/10.1145/ABC.DEF"  # original preserved
    finally:
        session.close()
