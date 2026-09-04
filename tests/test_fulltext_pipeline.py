from __future__ import annotations

import uuid
from unittest.mock import Mock

import pytest
import requests

from researchbridge.db.models import FullTextFetchError, FullTextFetchRun, Paper, PaperFullText
from researchbridge.fulltext.parse import PdfParseError
from researchbridge.fulltext.pipeline import FullTextFetchPipeline


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
