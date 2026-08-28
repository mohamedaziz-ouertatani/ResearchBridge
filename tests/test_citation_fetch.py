import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from researchbridge.citations.fetch import (
    CROSSREF_WORKS_URL,
    SEMANTIC_SCHOLAR_PAPER_DETAILS_URL,
    CitationResolution,
    CrossrefCitationFetcher,
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


@responses.activate
def test_crossref_fetch_raw_returns_reference_dois_only() -> None:
    url = CROSSREF_WORKS_URL.format(doi="10.1000/target")
    responses.add(
        responses.GET,
        url,
        json={
            "message": {
                "reference": [
                    {"DOI": "10.1000/ref1"},
                    {"DOI": "10.1000/REF2"},
                    {"unstructured": "a reference with no DOI"},
                ]
            }
        },
        status=200,
    )

    fetcher = CrossrefCitationFetcher()
    result = fetcher.fetch_raw("10.1000/target")

    # CrossRef only exposes a work's own reference list, never who cites it -
    # citing_source_ids always stays empty for this source.
    assert result == RawCitationsPayload(citing_source_ids=[], cited_source_ids=["10.1000/ref1", "10.1000/REF2"])


@responses.activate
def test_crossref_fetch_raw_handles_missing_reference_list() -> None:
    url = CROSSREF_WORKS_URL.format(doi="10.1000/target")
    responses.add(responses.GET, url, json={"message": {}}, status=200)

    fetcher = CrossrefCitationFetcher()
    result = fetcher.fetch_raw("10.1000/target")

    assert result == RawCitationsPayload(citing_source_ids=[], cited_source_ids=[])


@responses.activate
def test_crossref_sends_mailto_param_when_provided() -> None:
    url = CROSSREF_WORKS_URL.format(doi="10.1000/target")
    responses.add(responses.GET, url, json={"message": {}}, status=200)

    fetcher = CrossrefCitationFetcher(mailto="me@example.com")
    fetcher.fetch_raw("10.1000/target")

    assert responses.calls[0].request.params["mailto"] == "me@example.com"


def test_crossref_fetcher_identifies_papers_by_doi() -> None:
    paper = Paper(id=uuid.uuid4(), source="springer", source_id="s1", doi="10.1000/abc", title="A Paper")
    assert CrossrefCitationFetcher().identifier_for(paper) == "10.1000/abc"


def test_semantic_scholar_fetcher_identifies_papers_by_source_id() -> None:
    paper = Paper(id=uuid.uuid4(), source="semantic_scholar", source_id="s1", title="A Paper")
    assert SemanticScholarCitationFetcher().identifier_for(paper) == "s1"


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


def _add_doi_paper(session, doi: str, source: str = "springer", title: str = "A Paper") -> Paper:
    paper = Paper(id=uuid.uuid4(), source=source, source_id=doi, doi=doi, title=title)
    session.add(paper)
    session.commit()
    return paper


def test_resolve_citations_for_crossref_matches_by_doi_across_sources(session) -> None:
    target = _add_doi_paper(session, "10.1000/target", source="arxiv")
    cited = _add_doi_paper(session, "10.1000/cited", source="springer")

    payload = RawCitationsPayload(cited_source_ids=["10.1000/cited", "10.1000/not-in-corpus"])
    resolution = resolve_citations(session, target, payload, source="crossref")

    assert resolution.edges == [(target.id, cited.id)]
    assert resolution.unresolved_cited == 1


def test_resolve_citations_for_crossref_matches_case_insensitively(session) -> None:
    target = _add_doi_paper(session, "10.1000/target", source="arxiv")
    cited = _add_doi_paper(session, "10.1000/cited", source="springer")

    payload = RawCitationsPayload(cited_source_ids=["10.1000/CITED"])
    resolution = resolve_citations(session, target, payload, source="crossref")

    assert resolution.edges == [(target.id, cited.id)]


def test_resolve_citations_for_crossref_ignores_source_id_matches(session) -> None:
    """A crossref reference DOI must match Paper.doi, never Paper.source_id -
    a semantic_scholar paperId that happens to look like a DOI string must
    not accidentally resolve."""
    target = _add_doi_paper(session, "10.1000/target", source="arxiv")
    decoy = Paper(id=uuid.uuid4(), source="semantic_scholar", source_id="10.1000/cited", title="Decoy")
    session.add(decoy)
    session.commit()

    payload = RawCitationsPayload(cited_source_ids=["10.1000/cited"])
    resolution = resolve_citations(session, target, payload, source="crossref")

    assert resolution.edges == []
    assert resolution.unresolved_cited == 1


def test_persist_citation_edges_records_the_given_source(session) -> None:
    target = _add_doi_paper(session, "10.1000/target", source="arxiv")
    cited = _add_doi_paper(session, "10.1000/cited", source="springer")

    resolution = CitationResolution(seed_paper_id=target.id, edges=[(target.id, cited.id)], unresolved_citing=0, unresolved_cited=0)
    created = persist_citation_edges(session, resolution, source="crossref")
    session.commit()

    assert created == 1
    row = session.query(PaperCitation).one()
    assert row.source == "crossref"
