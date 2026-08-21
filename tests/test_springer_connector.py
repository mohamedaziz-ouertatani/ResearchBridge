from pathlib import Path

import responses

from researchbridge.connectors.springer import SPRINGER_META_API_URL, SpringerConnector

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@responses.activate
def test_fetch_first_page_returns_normalized_papers() -> None:
    responses.add(responses.GET, SPRINGER_META_API_URL, body=_load("springer_page1.json"), status=200)

    connector = SpringerConnector(query='"machine learning"', api_key="fake-key")
    result = connector.fetch(resume_state=None)

    assert len(result.papers) == 2
    assert result.resume_state == {"start": 3}
    assert result.exhausted is False


@responses.activate
def test_fetch_last_page_marks_exhausted() -> None:
    responses.add(responses.GET, SPRINGER_META_API_URL, body=_load("springer_page2.json"), status=200)

    connector = SpringerConnector(query='"machine learning"', api_key="fake-key")
    result = connector.fetch(resume_state={"start": 3})

    assert len(result.papers) == 1
    assert result.resume_state == {"start": 4}
    assert result.exhausted is True


@responses.activate
def test_normalized_paper_fields() -> None:
    responses.add(responses.GET, SPRINGER_META_API_URL, body=_load("springer_page1.json"), status=200)

    connector = SpringerConnector(query='"machine learning"', api_key="fake-key")
    result = connector.fetch(resume_state=None)
    paper = result.papers[0]

    assert paper.source == "springer"
    assert paper.source_id == "10.1007/s10994-026-00001-1"  # "doi:" prefix stripped
    assert paper.doi == "10.1007/s10994-026-00001-1"
    assert paper.title == "A Study of Machine Learning Techniques"
    assert paper.abstract == "This paper studies machine learning techniques for classification."
    assert [a.name for a in paper.authors] == ["Alice Example", "Bob Sample"]
    assert paper.authors[0].order == 0
    assert paper.authors[0].orcid == "0000-0001-0000-0001"
    assert paper.authors[1].order == 1
    assert paper.authors[1].orcid is None
    assert set(paper.categories) == {"Computer Science", "Artificial Intelligence"}
    assert paper.venue == "Machine Learning"
    assert paper.publication_date.isoformat() == "2026-01-15"
    assert paper.document_type == "Journal"
    assert paper.language == "en"
    assert paper.url == "http://link.springer.com/openurl/fulltext?id=doi:10.1007/s10994-026-00001-1"
    assert paper.open_access is True
    assert paper.raw_metadata["keyword"] == ["Machine Learning", "Classification"]


@responses.activate
def test_second_record_open_access_false_and_no_orcid() -> None:
    responses.add(responses.GET, SPRINGER_META_API_URL, body=_load("springer_page1.json"), status=200)

    connector = SpringerConnector(query='"machine learning"', api_key="fake-key")
    result = connector.fetch(resume_state=None)
    paper = result.papers[1]

    assert paper.open_access is False
    assert paper.authors[0].orcid is None


@responses.activate
def test_empty_page_is_exhausted() -> None:
    empty_response = b'{"result": [{"total": "0", "start": "1", "pageLength": "0", "recordsDisplayed": "0"}], "records": []}'
    responses.add(responses.GET, SPRINGER_META_API_URL, body=empty_response, status=200)

    connector = SpringerConnector(query='"nonexistent query"', api_key="fake-key")
    result = connector.fetch(resume_state=None)

    assert result.papers == []
    assert result.exhausted is True


def test_requires_an_api_key() -> None:
    import pytest

    with pytest.raises(ValueError, match="api_key"):
        SpringerConnector(query='"machine learning"', api_key=None)
