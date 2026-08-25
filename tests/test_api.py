from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import pytest
from fastapi.testclient import TestClient

from researchbridge.api.app import create_app
from researchbridge.api.deps import get_embedder, get_session
from researchbridge.db.models import EMBEDDING_DIM, Author, Embedding, Evidence, ExtractedClaim, Paper, PaperAuthor, PaperCategory
from researchbridge.embedding.pipeline import EMBEDDING_TYPE


@dataclass
class FakeEmbedder:
    """Deterministic hash-based embedder - avoids loading the real model in API tests."""

    model_name: str = "fake-embedder-v1"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_hash_to_unit_vector(t) for t in texts]


def _hash_to_unit_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def client(session_factory, embedder):
    app = create_app()

    def _session_override():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = _session_override
    app.dependency_overrides[get_embedder] = lambda: embedder
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    s.close()  # an open session would block the fixture's TRUNCATE teardown


def _add_paper(
    session,
    source_id: str,
    title: str = "A Paper",
    year: int = 2024,
    categories: tuple[str, ...] = ("cs.LG",),
    authors: tuple[str, ...] = (),
    embed: FakeEmbedder | None = None,
    source: str = "arxiv",
    claim: bool = False,
) -> Paper:
    paper = Paper(
        id=uuid.uuid4(),
        source=source,
        source_id=source_id,
        title=title,
        abstract=f"Abstract for {title}.",
        publication_date=date(year, 6, 1),
        url=f"https://arxiv.org/abs/{source_id}",
        raw_metadata={"primary_category": categories[0] if categories else None},
        ingestion_metadata={},
    )
    session.add(paper)
    session.flush()

    for i, name in enumerate(authors):
        author = session.query(Author).filter_by(name=name).one_or_none()
        if author is None:
            author = Author(id=uuid.uuid4(), name=name, author_metadata={})
            session.add(author)
            session.flush()
        session.add(PaperAuthor(paper_id=paper.id, author_id=author.id, author_order=i))

    for category in categories:
        session.add(PaperCategory(paper_id=paper.id, category=category, confidence="high", source="arxiv"))

    if embed is not None:
        [vector] = embed.embed_texts([f"{title}\n\n{paper.abstract}"])
        session.add(
            Embedding(
                paper_id=paper.id,
                embedding_type=EMBEDDING_TYPE,
                model_name=embed.model_name,
                vector=vector,
            )
        )

    if claim:
        evidence = Evidence(
            paper_id=paper.id, evidence_type="method", section=None, text=paper.abstract,
            extraction_method="hybrid", model_version="v1", confidence="medium",
        )
        session.add(evidence)
        session.flush()
        session.add(
            ExtractedClaim(
                paper_id=paper.id, claim_type="method", text=paper.abstract,
                evidence_id=evidence.id, confidence="medium",
            )
        )

    session.commit()
    return paper


def test_health(client) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_list_papers_returns_page_with_total(client, session) -> None:
    for i in range(3):
        _add_paper(session, f"p{i}", title=f"Paper {i}")

    body = client.get("/api/papers", params={"limit": 2}).json()

    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["limit"] == 2
    assert body["offset"] == 0


def test_list_papers_includes_authors_and_categories(client, session) -> None:
    _add_paper(session, "p1", authors=("Alice Example", "Bob Sample"), categories=("cs.LG", "cs.AI"))

    item = client.get("/api/papers").json()["items"][0]

    assert item["authors"] == ["Alice Example", "Bob Sample"]  # author_order preserved
    assert set(item["categories"]) == {"cs.LG", "cs.AI"}
    assert item["primary_category"] == "cs.LG"


def test_list_papers_falls_back_to_first_category_when_raw_metadata_lacks_primary_category(
    client, session
) -> None:
    # A connector (e.g. Springer) that never stashes "primary_category" in
    # raw_metadata shouldn't leave primary_category permanently null - fall
    # back to a real category from paper_categories instead.
    paper = Paper(
        id=uuid.uuid4(), source="springer", source_id="10.1007/abc", title="A Paper",
        abstract="", raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    session.add(PaperCategory(paper_id=paper.id, category="Computer Science", confidence="high", source="springer"))
    session.add(PaperCategory(paper_id=paper.id, category="Artificial Intelligence", confidence="high", source="springer"))
    session.commit()

    item = client.get("/api/papers").json()["items"][0]

    assert item["primary_category"] == "Artificial Intelligence"  # alphabetically first


def test_list_papers_filters_by_year(client, session) -> None:
    _add_paper(session, "old", year=2019)
    _add_paper(session, "new", year=2025)

    body = client.get("/api/papers", params={"year": 2019}).json()

    assert body["total"] == 1
    assert body["items"][0]["source_id"] == "old"


def test_list_papers_filters_by_category(client, session) -> None:
    _add_paper(session, "ml", categories=("cs.LG",))
    _add_paper(session, "nlp", categories=("cs.CL",))

    body = client.get("/api/papers", params={"category": "cs.CL"}).json()

    assert body["total"] == 1
    assert body["items"][0]["source_id"] == "nlp"


def test_list_papers_title_filter_is_case_insensitive(client, session) -> None:
    _add_paper(session, "p1", title="Graph Transformers for Fraud")
    _add_paper(session, "p2", title="Unrelated Work")

    body = client.get("/api/papers", params={"q": "graph transformers"}).json()

    assert body["total"] == 1
    assert body["items"][0]["source_id"] == "p1"


def test_list_papers_rejects_oversized_limit(client) -> None:
    assert client.get("/api/papers", params={"limit": 5000}).status_code == 422


def test_list_papers_hides_excluded_papers_by_default(client, session) -> None:
    included = _add_paper(session, "included")
    excluded = _add_paper(session, "excluded")
    excluded.excluded_at = datetime.now(timezone.utc)
    session.commit()

    body = client.get("/api/papers").json()

    ids = {item["id"] for item in body["items"]}
    assert str(included.id) in ids
    assert str(excluded.id) not in ids
    assert body["total"] == 1


def test_list_papers_include_excluded_shows_them(client, session) -> None:
    excluded = _add_paper(session, "excluded")
    excluded.excluded_at = datetime.now(timezone.utc)
    session.commit()

    body = client.get("/api/papers?include_excluded=true").json()

    ids = {item["id"] for item in body["items"]}
    assert str(excluded.id) in ids


def test_get_paper_returns_detail(client, session) -> None:
    paper = _add_paper(session, "p1", title="Specific Paper")

    body = client.get(f"/api/papers/{paper.id}").json()

    assert body["title"] == "Specific Paper"
    assert body["url"] == "https://arxiv.org/abs/p1"


def test_paper_summary_includes_excluded_at(client, session) -> None:
    paper = _add_paper(session, "p1")

    body = client.get(f"/api/papers/{paper.id}").json()

    assert body["excluded_at"] is None


def test_get_paper_404s_for_unknown_id(client) -> None:
    assert client.get(f"/api/papers/{uuid.uuid4()}").status_code == 404


def _add_claim(
    session,
    paper: Paper,
    claim_type: str,
    text: str,
    confidence: str = "medium",
    section: str | None = None,
    extraction_method: str = "hybrid",
) -> None:
    evidence = Evidence(
        paper_id=paper.id, evidence_type=claim_type, section=section, text=text,
        extraction_method=extraction_method, model_version="test-v1", confidence=confidence,
    )
    session.add(evidence)
    session.flush()
    session.add(ExtractedClaim(paper_id=paper.id, claim_type=claim_type, text=text, evidence_id=evidence.id, confidence=confidence))
    session.commit()


def test_claims_returns_extracted_claims_in_field_order(client, session) -> None:
    paper = _add_paper(session, "p1")
    _add_claim(session, paper, "results", "We achieve 90% accuracy.")
    _add_claim(session, paper, "problem", "Latency is too high under load.")

    body = client.get(f"/api/papers/{paper.id}/claims").json()

    assert [c["claim_type"] for c in body] == ["problem", "results"]  # Sec 25/28 order, not insertion order


def test_claims_includes_confidence_and_section(client, session) -> None:
    paper = _add_paper(session, "p1")
    _add_claim(session, paper, "problem", "Latency is too high.", confidence="low", section="Abstract")

    body = client.get(f"/api/papers/{paper.id}/claims").json()

    assert body[0]["confidence"] == "low"
    assert body[0]["section"] == "Abstract"
    assert body[0]["extraction_method"] == "hybrid"


def test_claims_excludes_stub_rows(client, session) -> None:
    paper = _add_paper(session, "p1")
    _add_claim(session, paper, "problem", "Synthetic placeholder text.", extraction_method="stub")

    body = client.get(f"/api/papers/{paper.id}/claims").json()

    assert body == []


def test_claims_empty_list_for_paper_with_no_extraction(client, session) -> None:
    paper = _add_paper(session, "p1")

    assert client.get(f"/api/papers/{paper.id}/claims").json() == []


def test_claims_404s_for_unknown_paper(client) -> None:
    assert client.get(f"/api/papers/{uuid.uuid4()}/claims").status_code == 404


def test_search_returns_hits_ordered_by_distance(client, session, embedder) -> None:
    _add_paper(session, "p1", title="Neural networks for vision", embed=embedder)
    _add_paper(session, "p2", title="Distributed consensus protocols", embed=embedder)

    hits = client.get("/api/search", params={"q": "Neural networks for vision"}).json()

    assert len(hits) == 2
    assert hits[0]["paper"]["source_id"] == "p1"  # exact text match embeds identically
    assert hits[0]["distance"] < hits[1]["distance"]


def test_search_requires_a_query(client) -> None:
    assert client.get("/api/search", params={"q": ""}).status_code == 422


def test_similar_papers_excludes_the_source_paper(client, session, embedder) -> None:
    target = _add_paper(session, "p1", title="Neural networks", embed=embedder)
    _add_paper(session, "p2", title="Something else", embed=embedder)

    hits = client.get(f"/api/papers/{target.id}/similar").json()

    assert [h["paper"]["source_id"] for h in hits] == ["p2"]


def test_similar_papers_404s_for_unknown_paper(client) -> None:
    assert client.get(f"/api/papers/{uuid.uuid4()}/similar").status_code == 404


def test_similar_papers_409s_when_paper_has_no_embedding(client, session) -> None:
    paper = _add_paper(session, "p1")  # embed=None

    response = client.get(f"/api/papers/{paper.id}/similar")

    assert response.status_code == 409
    assert "No embedding" in response.json()["detail"]


def test_stats_reports_corpus_shape(client, session, embedder) -> None:
    _add_paper(session, "p1", year=2019, categories=("cs.LG",), authors=("Alice",), embed=embedder, source="arxiv", claim=True)
    _add_paper(session, "p2", year=2019, categories=("cs.LG",), authors=("Bob",), source="springer")
    _add_paper(session, "p3", year=2024, categories=("cs.CL",), authors=("Alice",), source="arxiv")

    body = client.get("/api/stats").json()

    assert body["total_papers"] == 3
    assert body["total_authors"] == 2  # Alice deduped across two papers
    assert body["embedded_papers"] == 1
    assert body["papers_with_claims"] == 1
    assert body["papers_by_year"] == {"2019": 2, "2024": 1}
    assert body["papers_by_category"]["cs.LG"] == 2
    assert body["papers_by_source"] == {"arxiv": 2, "springer": 1}


def test_stats_year_param_scopes_every_field_but_papers_by_year(client, session, embedder) -> None:
    _add_paper(session, "p1", year=2019, categories=("cs.LG",), authors=("Alice",), embed=embedder, source="arxiv", claim=True)
    _add_paper(session, "p2", year=2019, categories=("cs.LG",), authors=("Bob",), source="springer")
    _add_paper(session, "p3", year=2024, categories=("cs.CL",), authors=("Alice",), embed=embedder, source="arxiv", claim=True)

    body = client.get("/api/stats", params={"year": 2019}).json()

    assert body["total_papers"] == 2
    assert body["total_authors"] == 2
    assert body["embedded_papers"] == 1
    assert body["papers_with_claims"] == 1
    assert body["papers_by_category"] == {"cs.LG": 2}
    assert body["papers_by_source"] == {"arxiv": 1, "springer": 1}
    # papers_by_year stays unfiltered - it's what drives navigating to a
    # different year, not just the currently selected one.
    assert body["papers_by_year"] == {"2019": 2, "2024": 1}


def test_stats_excludes_excluded_papers_by_default(client, session, embedder) -> None:
    _add_paper(session, "p1", year=2019, categories=("cs.LG",), authors=("Alice",), embed=embedder)
    excluded = _add_paper(session, "p2", year=2019, categories=("cs.LG",), authors=("Bob",), embed=embedder)
    excluded.excluded_at = datetime.now(timezone.utc)
    session.commit()

    body = client.get("/api/stats").json()

    assert body["total_papers"] == 1
    assert body["total_authors"] == 1
    assert body["embedded_papers"] == 1
    assert body["papers_by_year"] == {"2019": 1}
    assert body["papers_by_category"]["cs.LG"] == 1
