from __future__ import annotations

import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from researchbridge.db.models import FullTextFetchError, FullTextFetchRun, Paper, PaperFullText


def _paper(session, source_id: str = "p1") -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=source_id, title="A Paper", abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def test_paper_fulltext_persists_sections(session_factory) -> None:
    session = session_factory()
    paper = _paper(session)
    session.commit()

    row = PaperFullText(
        paper_id=paper.id, sections={"introduction": "text"}, source_url="https://example.com/p1.pdf",
    )
    session.add(row)
    session.commit()

    fetched = session.get(PaperFullText, row.id)
    session.close()
    assert fetched.sections == {"introduction": "text"}
    assert fetched.fetch_method == "pymupdf"


def test_paper_fulltext_enforces_one_row_per_paper(session_factory) -> None:
    session = session_factory()
    paper = _paper(session)
    session.add(PaperFullText(paper_id=paper.id, sections={}, source_url="https://example.com/p1.pdf"))
    session.commit()

    session.add(PaperFullText(paper_id=paper.id, sections={}, source_url="https://example.com/p1-again.pdf"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.close()


def test_fulltext_fetch_error_links_to_its_run(session_factory) -> None:
    session = session_factory()
    paper = _paper(session)
    run = FullTextFetchRun(status="running")
    session.add(run)
    session.flush()

    session.add(
        FullTextFetchError(
            fulltext_fetch_run_id=run.id, paper_id=paper.id, error_type="fetch_error", error_detail="boom",
        )
    )
    session.commit()

    errors = session.query(FullTextFetchError).filter_by(fulltext_fetch_run_id=run.id).all()
    session.close()
    assert len(errors) == 1
    assert errors[0].error_type == "fetch_error"
