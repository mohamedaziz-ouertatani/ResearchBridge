from researchbridge.citations.fetch import (
    SEMANTIC_SCHOLAR_PAPER_DETAILS_URL,
    RawCitationsPayload,
    SemanticScholarCitationFetcher,
)

import responses


@responses.activate
def test_fetch_raw_returns_citing_and_cited_source_ids() -> None:
    url = SEMANTIC_SCHOLAR_PAPER_DETAILS_URL.format(paper_id="abc123")
    responses.add(
        responses.GET,
        url,
        json={
            "paperId": "abc123",
            "citations": [{"paperId": "cit1"}, {"paperId": "cit2"}],
            "references": [{"paperId": "ref1"}],
        },
        status=200,
    )

    fetcher = SemanticScholarCitationFetcher()
    result = fetcher.fetch_raw("abc123")

    assert result == RawCitationsPayload(citing_source_ids=["cit1", "cit2"], cited_source_ids=["ref1"])


@responses.activate
def test_fetch_raw_handles_missing_citations_and_references() -> None:
    url = SEMANTIC_SCHOLAR_PAPER_DETAILS_URL.format(paper_id="abc123")
    responses.add(responses.GET, url, json={"paperId": "abc123"}, status=200)

    fetcher = SemanticScholarCitationFetcher()
    result = fetcher.fetch_raw("abc123")

    assert result == RawCitationsPayload(citing_source_ids=[], cited_source_ids=[])


@responses.activate
def test_sends_api_key_header_when_provided() -> None:
    url = SEMANTIC_SCHOLAR_PAPER_DETAILS_URL.format(paper_id="abc123")
    responses.add(responses.GET, url, json={"paperId": "abc123"}, status=200)

    fetcher = SemanticScholarCitationFetcher(api_key="fake-key")
    fetcher.fetch_raw("abc123")

    assert responses.calls[0].request.headers["x-api-key"] == "fake-key"
