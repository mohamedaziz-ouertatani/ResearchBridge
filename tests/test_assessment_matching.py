from __future__ import annotations

import uuid

from researchbridge.assessment.matching import match_uploaded_paper
from researchbridge.db.models import Paper


def _arxiv_paper(session, source_id: str, title: str = "a paper") -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def test_matches_by_arxiv_id_in_filename(session_factory) -> None:
    session = session_factory()
    paper = _arxiv_paper(session, "2501.00348")
    session.commit()

    matched = match_uploaded_paper(session, "2501.00348.pdf", "some extracted text")

    assert matched == paper.id
    session.close()


def test_matches_by_arxiv_id_in_filename_with_version_suffix(session_factory) -> None:
    session = session_factory()
    paper = _arxiv_paper(session, "2501.00348")
    session.commit()

    matched = match_uploaded_paper(session, "2501.00348v2.pdf", "some extracted text")

    assert matched == paper.id
    session.close()


def test_matches_by_arxiv_prefix_in_text_when_filename_has_no_id(session_factory) -> None:
    session = session_factory()
    paper = _arxiv_paper(session, "2501.00348")
    session.commit()

    matched = match_uploaded_paper(session, "my_paper.pdf", "Some header text\narXiv:2501.00348v1 [cs.LG]\nAbstract...")

    assert matched == paper.id
    session.close()


def test_returns_none_when_no_id_found(session_factory) -> None:
    session = session_factory()

    matched = match_uploaded_paper(session, "my_notes.pdf", "just some idea text with no identifiers")

    assert matched is None
    session.close()


def test_returns_none_when_id_found_but_no_matching_paper_in_corpus(session_factory) -> None:
    session = session_factory()

    matched = match_uploaded_paper(session, "2501.00348.pdf", "some extracted text")

    assert matched is None
    session.close()


def test_ignores_bare_digit_pattern_in_text_without_arxiv_prefix(session_factory) -> None:
    session = session_factory()
    _arxiv_paper(session, "2024.001")
    session.commit()

    # a coincidental digit.digit sequence in prose text should not trigger a match -
    # only an explicit "arXiv:" prefix in the text counts, unlike the filename check
    matched = match_uploaded_paper(session, None, "results improved by 2024.001 percent over baseline")

    assert matched is None
    session.close()


def test_returns_none_for_no_filename_and_plain_text(session_factory) -> None:
    session = session_factory()

    matched = match_uploaded_paper(session, None, "an idea with no arxiv reference at all")

    assert matched is None
    session.close()
