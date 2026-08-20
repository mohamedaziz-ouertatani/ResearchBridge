from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import select

from researchbridge.connectors.base import ConnectorFetchResult, NormalizedPaper
from researchbridge.db.models import IngestionError, IngestionRun, Paper
from researchbridge.ingestion.pipeline import IngestionPipeline


def _paper(source_id: str, title: str = "A Paper") -> NormalizedPaper:
    return NormalizedPaper(
        source="fake",
        source_id=source_id,
        title=title,
        abstract="An abstract.",
        publication_date=date(2026, 1, 1),
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
