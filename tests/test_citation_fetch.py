import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from researchbridge.citations.fetch import (
    SEMANTIC_SCHOLAR_PAPER_DETAILS_URL,
    CitationResolution,
    RawCitationsPayload,
    SemanticScholarCitationFetcher,
    persist_citation_edges,
    resolve_citations,
)
from researchbridge.db.models import Paper, PaperCitation

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


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    s.close()


def _add_s2_paper(session, source_id: str, title: str = "A Paper") -> Paper:
    paper = Paper(id=uuid.uuid4(), source="semantic_scholar", source_id=source_id, title=title)
    session.add(paper)
    session.commit()
    return paper


def test_resolve_citations_matches_in_corpus_papers_only(session) -> None:
    target = _add_s2_paper(session, "target1")
    citing = _add_s2_paper(session, "citing1")
    cited = _add_s2_paper(session, "cited1")

    payload = RawCitationsPayload(citing_source_ids=["citing1", "not-in-corpus"], cited_source_ids=["cited1"])
    resolution = resolve_citations(session, target, payload)

    assert set(resolution.edges) == {(citing.id, target.id), (target.id, cited.id)}
    assert resolution.unresolved_citing == 1
    assert resolution.unresolved_cited == 0


def test_resolve_citations_with_no_ids_returns_empty_resolution(session) -> None:
    target = _add_s2_paper(session, "target1")

    resolution = resolve_citations(session, target, RawCitationsPayload())

    assert resolution == CitationResolution(seed_paper_id=target.id, edges=[], unresolved_citing=0, unresolved_cited=0)


def test_persist_citation_edges_creates_rows(session) -> None:
    target = _add_s2_paper(session, "target1")
    citing = _add_s2_paper(session, "citing1")

    resolution = CitationResolution(seed_paper_id=target.id, edges=[(citing.id, target.id)], unresolved_citing=0, unresolved_cited=0)
    created = persist_citation_edges(session, resolution)
    session.commit()

    assert created == 1
    row = session.query(PaperCitation).one()
    assert (row.citing_paper_id, row.cited_paper_id, row.source, row.confidence) == (citing.id, target.id, "semantic_scholar", "high")


def test_persist_citation_edges_skips_edges_that_already_exist(session) -> None:
    target = _add_s2_paper(session, "target1")
    citing = _add_s2_paper(session, "citing1")
    session.add(PaperCitation(citing_paper_id=citing.id, cited_paper_id=target.id, source="semantic_scholar", confidence="high"))
    session.commit()

    resolution = CitationResolution(seed_paper_id=target.id, edges=[(citing.id, target.id)], unresolved_citing=0, unresolved_cited=0)
    created = persist_citation_edges(session, resolution)
    session.commit()

    assert created == 0
    assert session.query(PaperCitation).count() == 1
