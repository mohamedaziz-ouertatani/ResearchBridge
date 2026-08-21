from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

import pytest

from researchbridge.assessment.build import build_assessment
from researchbridge.db.models import (
    EMBEDDING_DIM,
    Embedding,
    Evidence,
    ExtractedClaim,
    Paper,
    ResearchAssessment,
    ResearchAssessmentEvidence,
    ResearchInput,
)
from researchbridge.embedding.pipeline import EMBEDDING_TYPE


def _hash_to_unit_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


@dataclass
class FakeEmbedder:
    """Deterministic hash-based embedder - fixed EMBEDDING_DIM regardless of
    batch size, unlike a word-overlap fake whose vector length depends on
    the vocabulary of whatever's in the current call. Identical text always
    embeds identically, so an exact-text-match retrieval test is reliable."""

    model_name: str = "fake-embedder-v1"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_hash_to_unit_vector(t) for t in texts]


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def _paper(session, embedder, source_id: str, title: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    [vector] = embedder.embed_texts([title])
    session.add(
        Embedding(
            paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name, vector=vector,
        )
    )
    return paper


def _claim(session, paper: Paper, claim_type: str, text: str, extraction_method: str = "hybrid") -> uuid.UUID:
    evidence = Evidence(
        paper_id=paper.id, evidence_type=claim_type, section=None, text=text,
        extraction_method=extraction_method, model_version="v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    session.add(
        ExtractedClaim(paper_id=paper.id, claim_type=claim_type, text=text, evidence_id=evidence.id, confidence="medium")
    )
    return evidence.id


def _research_input(session, raw_text: str) -> ResearchInput:
    ri = ResearchInput(id=uuid.uuid4(), input_type="idea", raw_text=raw_text)
    session.add(ri)
    session.flush()
    return ri


def test_retrieves_related_papers_by_embedding_the_input_text(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert str(paper.id) in assessment.retrieved_paper_ids


def test_comparison_summary_is_grounded_in_real_claim_text(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "limitations", "evaluated only in offline settings")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.comparison_summary is not None
    assert "evaluated only in offline settings" in assessment.comparison_summary


def test_comparison_evidence_is_linked_for_every_claim_used(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    evidence_id = _claim(session, paper, "limitations", "evaluated only in offline settings")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    links = (
        session.query(ResearchAssessmentEvidence)
        .filter_by(research_assessment_id=assessment.id, role="comparison")
        .all()
    )
    session.close()
    assert {link.evidence_id for link in links} == {evidence_id}


def test_excludes_stub_claims_from_comparison(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "limitations", "synthetic placeholder text", extraction_method="stub")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.comparison_summary is None


def test_assessment_defaults_unassessed_fields_rather_than_fabricating(session_factory, embedder) -> None:
    session = session_factory()
    _paper(session, embedder, "p1", "graph transformers for fraud detection")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.status == "completed"
    assert assessment.novelty_level == "not_assessed"
    assert assessment.technical_feasibility_level == "not_assessed"
    assert assessment.research_gap_text is None
    assert assessment.recommendation is None
    assert assessment.completed_at is not None


def test_raises_for_unknown_research_input(session_factory, embedder) -> None:
    session = session_factory()
    with pytest.raises(ValueError):
        build_assessment(session, uuid.uuid4(), embedder)
    session.close()


def test_novelty_is_set_when_a_close_match_is_retrieved(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "limitations", "evaluated only in offline settings")
    # identical text to the paper's title -> distance 0.0 -> "low" novelty
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.novelty_level == "low"
    assert paper.title in assessment.novelty_reasoning


def test_novelty_evidence_is_linked_with_role_novelty(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    evidence_id = _claim(session, paper, "limitations", "evaluated only in offline settings")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    novelty_links = (
        session.query(ResearchAssessmentEvidence)
        .filter_by(research_assessment_id=assessment.id, role="novelty")
        .all()
    )
    session.close()
    assert {link.evidence_id for link in novelty_links} == {evidence_id}


def test_research_gap_is_set_from_an_explicit_claim_in_the_neighborhood(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "research_gap", "no real-time evaluation exists")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.research_gap_source == "input_specific"
    assert "no real-time evaluation exists" in assessment.research_gap_text
    assert assessment.candidate_gap_id is None


def test_research_gap_evidence_is_linked_with_role_research_gap(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    evidence_id = _claim(session, paper, "research_gap", "no real-time evaluation exists")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    gap_links = (
        session.query(ResearchAssessmentEvidence)
        .filter_by(research_assessment_id=assessment.id, role="research_gap")
        .all()
    )
    session.close()
    assert {link.evidence_id for link in gap_links} == {evidence_id}


def test_research_gap_stays_null_when_no_gap_evidence_exists(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "limitations", "some unrelated limitation")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.research_gap_text is None
    assert assessment.research_gap_source is None


def test_potential_applications_is_set_from_the_neighborhood(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "applications", "real-time payment fraud screening")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.potential_applications is not None
    assert assessment.potential_applications[0]["application"] == "real-time payment fraud screening"
    assert assessment.potential_applications[0]["source_paper"] == "graph transformers for fraud detection"


def test_applications_evidence_is_linked_with_role_application(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    evidence_id = _claim(session, paper, "applications", "real-time payment fraud screening")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    app_links = (
        session.query(ResearchAssessmentEvidence)
        .filter_by(research_assessment_id=assessment.id, role="application")
        .all()
    )
    session.close()
    assert {link.evidence_id for link in app_links} == {evidence_id}


def test_potential_applications_stays_null_without_any_applications_claim(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "limitations", "some unrelated limitation")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.potential_applications is None


def test_no_novelty_evidence_linked_when_nothing_could_be_assessed(session_factory, embedder) -> None:
    session = session_factory()
    _paper(session, embedder, "p1", "graph transformers for fraud detection")  # no claims
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    novelty_links = (
        session.query(ResearchAssessmentEvidence)
        .filter_by(research_assessment_id=assessment.id, role="novelty")
        .all()
    )
    session.close()
    assert novelty_links == []
    assert assessment.novelty_level == "not_assessed"
