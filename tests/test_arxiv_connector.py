from pathlib import Path

import responses

from researchbridge.connectors.arxiv import ARXIV_API_URL, ArxivConnector

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@responses.activate
def test_fetch_first_page_returns_normalized_papers() -> None:
    responses.add(responses.GET, ARXIV_API_URL, body=_load("arxiv_page1.xml"), status=200)

    connector = ArxivConnector(search_query="cat:cs.LG")
    result = connector.fetch(resume_state=None)

    assert len(result.papers) == 2
    assert result.resume_state == {"start_index": 2}
    assert result.exhausted is False


@responses.activate
def test_fetch_last_page_marks_exhausted() -> None:
    responses.add(responses.GET, ARXIV_API_URL, body=_load("arxiv_page2.xml"), status=200)

    connector = ArxivConnector(search_query="cat:cs.LG")
    result = connector.fetch(resume_state={"start_index": 2})

    assert len(result.papers) == 1
    assert result.resume_state == {"start_index": 3}
    assert result.exhausted is True


@responses.activate
def test_normalized_paper_fields() -> None:
    responses.add(responses.GET, ARXIV_API_URL, body=_load("arxiv_page1.xml"), status=200)

    connector = ArxivConnector(search_query="cat:cs.LG")
    result = connector.fetch(resume_state=None)
    paper = result.papers[0]

    assert paper.source == "arxiv"
    assert paper.source_id == "2408.01234"  # version stripped
    assert paper.title == "Graph Transformers for Real-Time Fraud Detection"
    assert paper.abstract is not None
    assert "offline evaluation" in paper.abstract
    assert "  " not in paper.abstract  # whitespace collapsed
    assert [a.name for a in paper.authors] == ["Alice Example", "Bob Sample"]
    assert paper.authors[0].order == 0
    assert paper.authors[1].order == 1
    assert set(paper.categories) == {"cs.LG", "cs.AI"}
    assert paper.raw_metadata["primary_category"] == "cs.LG"
    assert paper.url == "http://arxiv.org/abs/2408.01234v2"
    assert paper.open_access is True
    assert paper.document_type == "preprint"


@responses.activate
def test_empty_page_is_exhausted() -> None:
    empty_feed = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>0</opensearch:totalResults>
</feed>"""
    responses.add(responses.GET, ARXIV_API_URL, body=empty_feed, status=200)

    connector = ArxivConnector(search_query="cat:nonexistent")
    result = connector.fetch(resume_state=None)

    assert result.papers == []
    assert result.exhausted is True
