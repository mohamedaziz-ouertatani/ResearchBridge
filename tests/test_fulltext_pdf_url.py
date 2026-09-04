from __future__ import annotations

import uuid

from researchbridge.db.models import Paper
from researchbridge.fulltext.pdf_url import resolve_pdf_url


def _paper(source: str, source_id: str, url: str | None, open_access: bool) -> Paper:
    return Paper(
        id=uuid.uuid4(), source=source, source_id=source_id, title="t", abstract="",
        url=url, open_access=open_access, raw_metadata={}, ingestion_metadata={},
    )


def test_returns_none_when_not_open_access() -> None:
    paper = _paper("arxiv", "2401.00001", "https://arxiv.org/abs/2401.00001", open_access=False)
    assert resolve_pdf_url(paper) is None


def test_arxiv_derives_pdf_url_from_source_id() -> None:
    paper = _paper("arxiv", "2401.00001", "https://arxiv.org/abs/2401.00001", open_access=True)
    assert resolve_pdf_url(paper) == "https://arxiv.org/pdf/2401.00001"


def test_core_uses_paper_url_as_is() -> None:
    paper = _paper("core", "123", "https://core.ac.uk/download/123.pdf", open_access=True)
    assert resolve_pdf_url(paper) == "https://core.ac.uk/download/123.pdf"


def test_semantic_scholar_uses_paper_url_as_is() -> None:
    paper = _paper("semantic_scholar", "abc", "https://example.com/openaccess.pdf", open_access=True)
    assert resolve_pdf_url(paper) == "https://example.com/openaccess.pdf"


def test_springer_uses_paper_url_as_is_even_though_its_an_html_page() -> None:
    paper = _paper("springer", "10.1007/xyz", "https://link.springer.com/article/10.1007/xyz", open_access=True)
    assert resolve_pdf_url(paper) == "https://link.springer.com/article/10.1007/xyz"


def test_returns_none_for_an_unknown_source_even_if_open_access() -> None:
    paper = _paper("some_future_source", "1", "https://example.com/x", open_access=True)
    assert resolve_pdf_url(paper) is None


def test_returns_none_when_url_is_missing_for_a_non_arxiv_source() -> None:
    paper = _paper("core", "123", None, open_access=True)
    assert resolve_pdf_url(paper) is None
