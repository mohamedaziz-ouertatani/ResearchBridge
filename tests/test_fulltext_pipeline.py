from __future__ import annotations

import uuid
from unittest.mock import Mock

import pytest
import requests

from researchbridge.db.models import FullTextFetchError, FullTextFetchRun, Paper, PaperFullText
from researchbridge.fulltext.parse import PdfParseError
from researchbridge.fulltext.pipeline import FullTextFetchPipeline


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch) -> None:
    # throttle() is a real time.sleep(3.0) for arxiv-sourced papers - the
    # default source in this file's _paper() helper - mocked everywhere so
    # the suite doesn't pay 3s per test for real
    import researchbridge.fulltext.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "throttle", lambda: None)


def _paper(session, source: str = "arxiv", source_id: str = "2401.00001", open_access: bool = True) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source=source, source_id=source_id, title="t", abstract="",
        url=f"https://arxiv.org/abs/{source_id}", open_access=open_access,
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.commit()
    return paper


def test_fetches_and_persists_fulltext_for_an_open_access_paper(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    paper = _paper(session)
    session.close()

    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda pdf_bytes: {"introduction": "hello"})
    monkeypatch.setattr(
        pipeline_module.requests, "get", Mock(return_value=Mock(content=b"pdf-bytes", raise_for_status=Mock()))
    )

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(FullTextFetchRun, run_id)
        assert run.status == "completed"
        assert run.papers_seen == 1
        assert run.papers_fetched == 1
        assert run.papers_failed == 0

        row = session.query(PaperFullText).filter_by(paper_id=paper.id).one()
        assert row.sections == {"introduction": "hello"}
        assert row.source_url == "https://arxiv.org/pdf/2401.00001"
    finally:
        session.close()


def test_non_open_access_papers_are_never_selected(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    _paper(session, open_access=False)
    session.close()

    mock_get = Mock()
    monkeypatch.setattr(pipeline_module.requests, "get", mock_get)

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    run = session.get(FullTextFetchRun, run_id)
    session.close()
    assert run.papers_seen == 0
    mock_get.assert_not_called()


def test_fetch_failure_is_logged_and_run_continues(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    paper = _paper(session)
    session.close()

    monkeypatch.setattr(
        pipeline_module.requests, "get", Mock(side_effect=requests.ConnectionError("connection refused"))
    )

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(FullTextFetchRun, run_id)
        assert run.status == "completed"
        assert run.papers_failed == 1
        assert run.papers_fetched == 0

        errors = session.query(FullTextFetchError).filter_by(paper_id=paper.id).all()
        assert len(errors) == 1
        assert errors[0].error_type == "fetch_error"
        assert session.query(PaperFullText).filter_by(paper_id=paper.id).one_or_none() is None
    finally:
        session.close()


def test_parse_failure_is_logged_and_run_continues(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    paper = _paper(session)
    session.close()

    monkeypatch.setattr(
        pipeline_module, "parse_pdf", Mock(side_effect=PdfParseError("not a valid PDF"))
    )
    monkeypatch.setattr(
        pipeline_module.requests, "get", Mock(return_value=Mock(content=b"junk", raise_for_status=Mock()))
    )

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(FullTextFetchRun, run_id)
        assert run.papers_failed == 1
        errors = session.query(FullTextFetchError).filter_by(paper_id=paper.id).all()
        assert errors[0].error_type == "parse_error"
    finally:
        session.close()


def test_idempotent_skips_already_fetched_paper(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    paper = _paper(session)
    session.close()

    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda pdf_bytes: {"body": "hello"})
    monkeypatch.setattr(
        pipeline_module.requests, "get", Mock(return_value=Mock(content=b"pdf-bytes", raise_for_status=Mock()))
    )

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    pipeline.run()
    second_run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(FullTextFetchRun, second_run_id)
        assert run.papers_seen == 0  # already has a paper_fulltext row
        assert len(session.query(PaperFullText).filter_by(paper_id=paper.id).all()) == 1  # not duplicated
    finally:
        session.close()


def test_force_refetches_an_already_fetched_paper(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    paper = _paper(session)
    session.close()

    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda pdf_bytes: {"body": "first"})
    monkeypatch.setattr(
        pipeline_module.requests, "get", Mock(return_value=Mock(content=b"pdf-bytes", raise_for_status=Mock()))
    )
    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    pipeline.run()

    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda pdf_bytes: {"body": "second"})
    second_run_id = pipeline.run(force=True)

    session = session_factory()
    try:
        run = session.get(FullTextFetchRun, second_run_id)
        assert run.papers_seen == 1
        assert run.force is True
        row = session.query(PaperFullText).filter_by(paper_id=paper.id).one()
        assert row.sections == {"body": "second"}  # updated in place, not duplicated
    finally:
        session.close()


def test_throttles_after_an_arxiv_fetch_whether_it_succeeds_or_fails(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    good = _paper(session, source_id="2401.00001")
    bad = _paper(session, source_id="2401.00002")
    session.close()

    mock_throttle = Mock()
    monkeypatch.setattr(pipeline_module, "throttle", mock_throttle)
    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda pdf_bytes: {"body": "hello"})

    def fake_get(url, timeout):
        if "2401.00002" in url:
            raise requests.ConnectionError("connection refused")
        return Mock(content=b"pdf-bytes", raise_for_status=Mock())

    monkeypatch.setattr(pipeline_module.requests, "get", fake_get)

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    pipeline.run()

    assert mock_throttle.call_count == 2  # once per arxiv paper, success and failure alike


def test_does_not_throttle_for_non_arxiv_sources(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    _paper(session, source="semantic_scholar", source_id="s2-1")
    session.close()

    mock_throttle = Mock()
    monkeypatch.setattr(pipeline_module, "throttle", mock_throttle)
    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda pdf_bytes: {"body": "hello"})
    monkeypatch.setattr(
        pipeline_module.requests, "get", Mock(return_value=Mock(content=b"pdf-bytes", raise_for_status=Mock()))
    )

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    pipeline.run()

    mock_throttle.assert_not_called()


def test_core_paper_fetches_via_the_api_not_a_pdf_url(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    paper = _paper(session, source="core", source_id="123")
    session.close()

    monkeypatch.setattr(pipeline_module, "fetch_core_fulltext", lambda source_id, api_key: "Introduction\nhello")
    mock_get = Mock()
    monkeypatch.setattr(pipeline_module.requests, "get", mock_get)

    pipeline = FullTextFetchPipeline(session_factory=session_factory, core_api_key="test-key")
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(FullTextFetchRun, run_id)
        assert run.papers_fetched == 1
        row = session.query(PaperFullText).filter_by(paper_id=paper.id).one()
        assert row.sections == {"introduction": "hello"}
        assert row.source_url == "https://api.core.ac.uk/v3/outputs/123"
        mock_get.assert_not_called()  # never falls through to the generic PDF path
    finally:
        session.close()


def test_core_paper_is_skipped_without_an_api_key(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    _paper(session, source="core", source_id="123")
    session.close()

    mock_fetch = Mock()
    monkeypatch.setattr(pipeline_module, "fetch_core_fulltext", mock_fetch)

    pipeline = FullTextFetchPipeline(session_factory=session_factory, core_api_key=None)
    run_id = pipeline.run()

    session = session_factory()
    run = session.get(FullTextFetchRun, run_id)
    session.close()
    assert run.papers_skipped_no_url == 1
    mock_fetch.assert_not_called()


def test_core_paper_is_skipped_when_fulltext_is_unavailable(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    _paper(session, source="core", source_id="123")
    session.close()

    monkeypatch.setattr(pipeline_module, "fetch_core_fulltext", lambda source_id, api_key: None)

    pipeline = FullTextFetchPipeline(session_factory=session_factory, core_api_key="test-key")
    run_id = pipeline.run()

    session = session_factory()
    run = session.get(FullTextFetchRun, run_id)
    session.close()
    assert run.papers_skipped_no_url == 1
    assert run.papers_fetched == 0


def test_core_paper_fetch_failure_is_logged_and_run_continues(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    paper = _paper(session, source="core", source_id="123")
    session.close()

    monkeypatch.setattr(
        pipeline_module, "fetch_core_fulltext", Mock(side_effect=requests.ConnectionError("connection refused"))
    )

    pipeline = FullTextFetchPipeline(session_factory=session_factory, core_api_key="test-key")
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(FullTextFetchRun, run_id)
        assert run.papers_failed == 1
        errors = session.query(FullTextFetchError).filter_by(paper_id=paper.id).all()
        assert errors[0].error_type == "fetch_error"
    finally:
        session.close()


def test_limit_caps_papers_processed(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    _paper(session, source_id="2401.00001")
    _paper(session, source_id="2401.00002")
    session.close()

    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda pdf_bytes: {"body": "hello"})
    monkeypatch.setattr(
        pipeline_module.requests, "get", Mock(return_value=Mock(content=b"pdf-bytes", raise_for_status=Mock()))
    )

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    run_id = pipeline.run(limit=1)

    session = session_factory()
    run = session.get(FullTextFetchRun, run_id)
    session.close()
    assert run.papers_seen == 1
