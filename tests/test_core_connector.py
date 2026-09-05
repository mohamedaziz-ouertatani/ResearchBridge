from pathlib import Path

import responses

from researchbridge.connectors.core import CORE_SEARCH_URL, CoreConnector

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@responses.activate
def test_fetch_first_page_returns_normalized_papers() -> None:
    responses.add(responses.GET, CORE_SEARCH_URL, body=_load("core_page1.json"), status=200)

    connector = CoreConnector(query="machine learning", api_key="fake-key")
    result = connector.fetch(resume_state=None)

    assert len(result.papers) == 2
    assert result.resume_state == {"offset": 2}
    assert result.exhausted is False


@responses.activate
def test_fetch_last_page_marks_exhausted() -> None:
    responses.add(responses.GET, CORE_SEARCH_URL, body=_load("core_page2.json"), status=200)

    connector = CoreConnector(query="machine learning", api_key="fake-key")
    result = connector.fetch(resume_state={"offset": 2})

    assert len(result.papers) == 1
    assert result.resume_state == {"offset": 3}
    assert result.exhausted is True


@responses.activate
def test_normalized_paper_fields() -> None:
    responses.add(responses.GET, CORE_SEARCH_URL, body=_load("core_page1.json"), status=200)

    connector = CoreConnector(query="machine learning", api_key="fake-key")
    result = connector.fetch(resume_state=None)
    paper = result.papers[0]

    assert paper.source == "core"
    assert paper.source_id == "123456789"
    assert paper.doi == "10.1234/mlstudy-2026"
    assert paper.title == "A Study of Machine Learning Techniques"
    assert paper.abstract == "This paper studies machine learning techniques for classification."
    assert [a.name for a in paper.authors] == ["Alice Example", "Bob Sample"]
    assert paper.authors[0].order == 0
    assert paper.authors[1].order == 1
    assert set(paper.categories) == {"Computer Science", "Artificial Intelligence"}
    assert paper.venue == "Example University Press"
    assert paper.publication_date.isoformat() == "2026-01-15"
    assert paper.document_type == "research"
    assert paper.language == "en"
    assert paper.url == "https://core.ac.uk/download/123456789.pdf"
    assert paper.open_access is True
    assert paper.raw_metadata["yearPublished"] == 2026


@responses.activate
def test_publication_date_falls_back_to_year_when_no_published_date() -> None:
    responses.add(responses.GET, CORE_SEARCH_URL, body=_load("core_page1.json"), status=200)

    connector = CoreConnector(query="machine learning", api_key="fake-key")
    result = connector.fetch(resume_state=None)
    paper = result.papers[1]  # publishedDate is null, yearPublished is 2025

    assert paper.publication_date.isoformat() == "2025-01-01"
    assert paper.doi is None


@responses.activate
def test_out_of_range_year_published_does_not_crash_the_fetch() -> None:
    """Live failure (2026-09-02/09-03): a real CORE record with
    yearPublished=10000 raised an uncaught ValueError that killed the
    whole ingestion run instead of just leaving this one record's
    publication_date unset - see core.py's _year_to_date docstring."""
    responses.add(responses.GET, CORE_SEARCH_URL, body=_load("core_bad_year.json"), status=200)

    connector = CoreConnector(query="machine learning", api_key="fake-key")
    result = connector.fetch(resume_state=None)
    paper = result.papers[0]

    assert paper.publication_date is None
    assert paper.raw_metadata["yearPublished"] == 10000


@responses.activate
def test_url_falls_back_to_source_fulltext_urls_when_no_download_url() -> None:
    responses.add(responses.GET, CORE_SEARCH_URL, body=_load("core_page1.json"), status=200)

    connector = CoreConnector(query="machine learning", api_key="fake-key")
    result = connector.fetch(resume_state=None)
    paper = result.papers[1]  # downloadUrl is null

    assert paper.url == "https://example.org/dl-chapter.pdf"


@responses.activate
def test_empty_page_is_exhausted() -> None:
    empty_response = b'{"totalHits": 0, "limit": 2, "offset": 0, "results": []}'
    responses.add(responses.GET, CORE_SEARCH_URL, body=empty_response, status=200)

    connector = CoreConnector(query="nonexistent query", api_key="fake-key")
    result = connector.fetch(resume_state=None)

    assert result.papers == []
    assert result.exhausted is True


@responses.activate
def test_sends_bearer_authorization_header() -> None:
    responses.add(responses.GET, CORE_SEARCH_URL, body=_load("core_page1.json"), status=200)

    connector = CoreConnector(query="machine learning", api_key="fake-key")
    connector.fetch(resume_state=None)

    assert responses.calls[0].request.headers["Authorization"] == "Bearer fake-key"


def test_requires_an_api_key() -> None:
    import pytest

    with pytest.raises(ValueError, match="api_key"):
        CoreConnector(query="machine learning", api_key=None)
