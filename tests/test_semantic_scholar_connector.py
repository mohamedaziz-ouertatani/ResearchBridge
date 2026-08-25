from pathlib import Path

import responses

from researchbridge.connectors.semantic_scholar import (
    SEMANTIC_SCHOLAR_BULK_SEARCH_URL,
    SemanticScholarConnector,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@responses.activate
def test_fetch_first_page_returns_normalized_papers() -> None:
    responses.add(responses.GET, SEMANTIC_SCHOLAR_BULK_SEARCH_URL, body=_load("semantic_scholar_page1.json"), status=200)

    connector = SemanticScholarConnector(query='"machine learning"')
    result = connector.fetch(resume_state=None)

    assert len(result.papers) == 2
    assert result.resume_state == {"token": "next-page-token-abc"}
    assert result.exhausted is False


@responses.activate
def test_fetch_last_page_marks_exhausted() -> None:
    responses.add(responses.GET, SEMANTIC_SCHOLAR_BULK_SEARCH_URL, body=_load("semantic_scholar_page2.json"), status=200)

    connector = SemanticScholarConnector(query='"machine learning"')
    result = connector.fetch(resume_state={"token": "next-page-token-abc"})

    assert len(result.papers) == 1
    assert result.resume_state == {"token": None}
    assert result.exhausted is True


@responses.activate
def test_normalized_paper_fields() -> None:
    responses.add(responses.GET, SEMANTIC_SCHOLAR_BULK_SEARCH_URL, body=_load("semantic_scholar_page1.json"), status=200)

    connector = SemanticScholarConnector(query='"machine learning"')
    result = connector.fetch(resume_state=None)
    paper = result.papers[0]

    assert paper.source == "semantic_scholar"
    assert paper.source_id == "a1b2c3d4e5f60000000000000000000000000001"
    assert paper.doi == "10.1007/s10994-026-00001-1"
    assert paper.title == "A Study of Machine Learning Techniques"
    assert paper.abstract == "This paper studies machine learning techniques for classification."
    assert [a.name for a in paper.authors] == ["Alice Example", "Bob Sample"]
    assert paper.authors[0].order == 0
    assert paper.authors[0].orcid is None  # bulk search's default author fields carry no ORCID
    assert paper.authors[1].order == 1
    assert set(paper.categories) == {"Computer Science", "Artificial Intelligence"}
    assert paper.venue == "Machine Learning"
    assert paper.publication_date.isoformat() == "2026-01-15"
    assert paper.document_type == "JournalArticle"
    assert paper.url == "https://example.org/pdf/00001.pdf"
    assert paper.open_access is True
    assert paper.raw_metadata["externalIds"] == {"DOI": "10.1007/s10994-026-00001-1", "ArXiv": "2601.00001"}


@responses.activate
def test_paper_without_doi_or_open_access_pdf() -> None:
    responses.add(responses.GET, SEMANTIC_SCHOLAR_BULK_SEARCH_URL, body=_load("semantic_scholar_page1.json"), status=200)

    connector = SemanticScholarConnector(query='"machine learning"')
    result = connector.fetch(resume_state=None)
    paper = result.papers[1]

    assert paper.doi is None
    # No open-access PDF, but "read online" still needs somewhere to go -
    # falls back to the Semantic Scholar landing page for this paperId.
    assert paper.url == "https://www.semanticscholar.org/paper/a1b2c3d4e5f60000000000000000000000000002"
    assert paper.open_access is False
    assert paper.publication_date is None  # no publicationDate and no year fallback needed for this fixture


@responses.activate
def test_empty_page_is_exhausted() -> None:
    empty_response = b'{"total": 0, "data": []}'
    responses.add(responses.GET, SEMANTIC_SCHOLAR_BULK_SEARCH_URL, body=empty_response, status=200)

    connector = SemanticScholarConnector(query='"nonexistent query"')
    result = connector.fetch(resume_state=None)

    assert result.papers == []
    assert result.exhausted is True


@responses.activate
def test_sends_api_key_header_when_provided() -> None:
    responses.add(responses.GET, SEMANTIC_SCHOLAR_BULK_SEARCH_URL, body=_load("semantic_scholar_page1.json"), status=200)

    connector = SemanticScholarConnector(query='"machine learning"', api_key="fake-key")
    connector.fetch(resume_state=None)

    assert responses.calls[0].request.headers["x-api-key"] == "fake-key"


@responses.activate
def test_no_api_key_header_when_not_provided() -> None:
    responses.add(responses.GET, SEMANTIC_SCHOLAR_BULK_SEARCH_URL, body=_load("semantic_scholar_page1.json"), status=200)

    connector = SemanticScholarConnector(query='"machine learning"')
    connector.fetch(resume_state=None)

    assert "x-api-key" not in responses.calls[0].request.headers
