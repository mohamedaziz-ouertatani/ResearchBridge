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


def test_misses_an_arxiv_reference_beyond_the_text_search_window(session_factory) -> None:
    # _TEXT_SEARCH_WINDOW is a hard 2000-char cutoff: a genuine arXiv:
    # reference that appears later in the document (e.g. in a "related
    # work" section rather than the header) is silently never matched,
    # even though the text is otherwise identical to a matching case
    session = session_factory()
    _arxiv_paper(session, "2501.00348")
    session.commit()

    padding = "x" * 2000
    text = f"{padding}\narXiv:2501.00348v1 [cs.LG]\nAbstract..."

    matched = match_uploaded_paper(session, "my_paper.pdf", text)

    assert matched is None
    session.close()


def test_matches_an_arxiv_reference_just_inside_the_text_search_window(session_factory) -> None:
    session = session_factory()
    paper = _arxiv_paper(session, "2501.00348")
    session.commit()

    padding = "x" * 1000
    text = f"{padding}\narXiv:2501.00348v1 [cs.LG]\nAbstract..."

    matched = match_uploaded_paper(session, "my_paper.pdf", text)

    assert matched == paper.id
    session.close()


def test_filename_id_check_requires_a_plausible_arxiv_month(session_factory) -> None:
    # regression guard for a real false match found via testing: a filename
    # merely containing a date-like or version-like digit.digit substring
    # ("...2024.12345_final.pdf") must not match a corpus paper just because
    # the digits coincide - "24" isn't a valid arXiv YYMM month component
    # (01-12), so this should never have been treated as an arXiv id at all
    session = session_factory()
    _arxiv_paper(session, "2024.12345")
    session.commit()

    matched = match_uploaded_paper(session, "meeting_notes_2024.12345_final.pdf", "no arxiv reference in here")

    assert matched is None
    session.close()


def test_filename_id_check_still_matches_a_plausible_arxiv_month(session_factory) -> None:
    session = session_factory()
    paper = _arxiv_paper(session, "2412.00348")
    session.commit()

    matched = match_uploaded_paper(session, "2412.00348.pdf", "some extracted text")

    assert matched == paper.id
    session.close()
