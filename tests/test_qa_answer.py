from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

from researchbridge.db.models import EMBEDDING_DIM, Embedding, Evidence, ExtractedClaim, Paper
from researchbridge.embedding.pipeline import EMBEDDING_TYPE
from researchbridge.qa.answer import answer_question


def _hash_to_unit_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


@dataclass
class FakeEmbedder:
    model_name: str = "fake-embedder-v1"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_hash_to_unit_vector(t) for t in texts]


def _add_paper(session, embedder, source_id: str, title: str, source: str = "arxiv") -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source=source, source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    [vector] = embedder.embed_texts([title])
    session.add(
        Embedding(paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name, vector=vector)
    )
    return paper


def _add_claim(
    session, paper: Paper, claim_type: str, text: str, extraction_method: str = "hybrid", section: str | None = None
) -> None:
    evidence = Evidence(
        paper_id=paper.id, evidence_type=claim_type, section=section, text=text,
        extraction_method=extraction_method, model_version="v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    session.add(
        ExtractedClaim(paper_id=paper.id, claim_type=claim_type, text=text, evidence_id=evidence.id, confidence="medium")
    )


def test_ranks_the_quote_matching_the_question_first(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper_a = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    paper_b = _add_paper(session, embedder, "p2", "graph transformers for fraud detection variant")
    _add_claim(session, paper_a, "limitations", "evaluated only on offline datasets")
    _add_claim(session, paper_b, "limitations", "the exact question text")
    session.commit()

    # The fake embedder hashes exact text, so asking the exact text of one
    # quote guarantees it scores highest - same trick test_assessment_api.py
    # already uses for "same title as the query text -> distance 0.0".
    hits = answer_question(session, embedder, "the exact question text")

    assert hits[0].text == "the exact question text"
    assert hits[0].paper_id == paper_b.id
    session.close()


def test_filters_out_stub_evidence(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "synthetic placeholder claim", extraction_method="stub")
    session.commit()

    hits = answer_question(session, embedder, "graph transformers for fraud detection")

    assert hits == []
    session.close()


def test_excludes_curated_out_papers(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    from datetime import datetime, timezone

    paper.excluded_at = datetime.now(timezone.utc)
    session.commit()

    hits = answer_question(session, embedder, "graph transformers for fraud detection")

    assert hits == []
    session.close()


def test_returns_empty_list_for_empty_corpus(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()

    hits = answer_question(session, embedder, "anything")

    assert hits == []
    session.close()


def test_returns_empty_list_when_candidates_have_no_real_evidence(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    session.commit()

    hits = answer_question(session, embedder, "graph transformers for fraud detection")

    assert hits == []
    session.close()


def test_respects_top_k_quotes_limit(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    for i in range(5):
        _add_claim(session, paper, "limitations", f"limitation number {i}")
    session.commit()

    hits = answer_question(session, embedder, "graph transformers for fraud detection", top_k_quotes=3)

    assert len(hits) == 3
    session.close()


def test_hit_carries_paper_and_section_metadata(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection", source="springer")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets", section="Discussion")
    session.commit()

    hits = answer_question(session, embedder, "graph transformers for fraud detection")

    assert hits[0].paper_title == "graph transformers for fraud detection"
    assert hits[0].paper_source == "springer"
    assert hits[0].claim_type == "limitations"
    assert hits[0].section == "Discussion"
    assert hits[0].confidence == "medium"
    session.close()
