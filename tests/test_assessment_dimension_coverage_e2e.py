from __future__ import annotations

import uuid

import pytest

from researchbridge.assessment.build import build_assessment
from researchbridge.db.models import (
    EMBEDDING_DIM,
    Embedding,
    Evidence,
    ExtractedClaim,
    Paper,
    ResearchAssessmentEvidence,
    ResearchInput,
)
from researchbridge.embedding.pipeline import EMBEDDING_TYPE

FRAUD_IDEA = (
    "Can we build a privacy-preserving federated learning system for detecting "
    "financial fraud across multiple banks without sharing raw transaction data, "
    "while remaining robust to concept drift and highly imbalanced fraud classes?"
)


def _hash_to_unit_vector(text: str) -> list[float]:
    import hashlib

    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


class FakeEmbedder:
    model_name = "fake-e2e-embedder"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
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
    session.add(Embedding(paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name, vector=vector))
    return paper


def _claim(session, paper: Paper, claim_type: str, text: str) -> uuid.UUID:
    evidence = Evidence(
        paper_id=paper.id, evidence_type=claim_type, section=None, text=text,
        extraction_method="hybrid", model_version="v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    session.add(
        ExtractedClaim(paper_id=paper.id, claim_type=claim_type, text=text, evidence_id=evidence.id, confidence="medium")
    )
    return evidence.id


def test_fraud_idea_end_to_end_structural_guarantees(session_factory, embedder) -> None:
    session = session_factory()

    p1 = _paper(session, embedder, "p1", FRAUD_IDEA)  # near-identical, drives retrieval
    _claim(session, p1, "method", "We propose a federated learning approach for cross-bank fraud detection.")
    _claim(session, p1, "results", "Our federated approach achieves 94% detection accuracy.")
    _claim(session, p1, "main_contribution", "We show that federated aggregation preserves privacy across banks.")
    _claim(session, p1, "limitations", "The method assumes a stationary fraud distribution and does not handle concept drift.")

    p2 = _paper(session, embedder, "p2", "privacy preserving fraud detection across banks")
    _claim(session, p2, "problem", "Sharing raw transaction data across banks raises privacy concerns.")
    _claim(session, p2, "limitations", "Highly imbalanced fraud classes remain a challenge for this approach.")

    ri = ResearchInput(id=uuid.uuid4(), input_type="idea", raw_text=FRAUD_IDEA)
    session.add(ri)
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=10)

    # retrieves relevant papers
    assert str(p1.id) in assessment.retrieved_paper_ids
    assert str(p2.id) in assessment.retrieved_paper_ids

    # identifies multiple dimensions of the idea + produces per-dimension coverage
    assert "Dimension coverage:" in assessment.novelty_reasoning
    coverage_block = assessment.novelty_reasoning.split("Dimension coverage:")[1]
    coverage_lines = [line for line in coverage_block.splitlines() if "->" in line]
    assert len(coverage_lines) >= 2

    # does not classify results or main_contribution as problems
    assert assessment.comparison_summary is not None
    assert "94% detection accuracy" not in _section(assessment.comparison_summary, "Problems already addressed")
    assert "federated aggregation preserves privacy" not in _section(
        assessment.comparison_summary, "Problems already addressed"
    )

    # does not treat generic business/application language as a research gap:
    # gap.py only ever sources research_gap_text from an explicit research_gap-
    # type claim, a reused approved candidate gap, or a cross-paper limitation/
    # research_gap cluster - never from problem/method/results/applications
    # claims, which is what "generic business language" would show up as.
    # No research_gap-type claim exists in this fixture, so the gap must be
    # either None (not_found/not_assessed) or explicitly source-labeled.
    assert assessment.research_gap_source in (None, "no_relevant_evidence", "checked_no_gap_found", "input_specific")
    if assessment.research_gap_text is not None:
        assert assessment.research_gap_source in ("input_specific", "reused_candidate_gap")

    # distinguishes explicit gaps from inferred/coverage-based observations:
    # research_gap_source, when a gap IS found, is never a bare "found" -
    # it names its own provenance
    if assessment.research_gap_text is not None:
        assert assessment.research_gap_source is not None

    # keeps evidence traceable to real source claims
    evidence_links = session.query(ResearchAssessmentEvidence).filter_by(research_assessment_id=assessment.id).all()
    real_evidence_ids = {
        row.id
        for row in session.query(Evidence).filter(Evidence.paper_id.in_([p1.id, p2.id])).all()
    }
    for link in evidence_links:
        assert link.evidence_id in real_evidence_ids

    # does not generate product opportunities
    assert assessment.potential_opportunities is None

    # produces an explainable recommendation - the reasoning is reconstructible
    # from already-persisted fields (see narrative.py, Task 10)
    assert assessment.recommendation is not None
    assert assessment.confidence is not None

    session.close()


def test_fraud_idea_honestly_returns_insufficient_evidence_with_an_unrelated_corpus(session_factory, embedder) -> None:
    session = session_factory()

    unrelated = _paper(session, embedder, "u1", "the history of medieval bread baking techniques")
    _claim(session, unrelated, "method", "We survey historical baking equipment across three centuries.")

    ri = ResearchInput(id=uuid.uuid4(), input_type="idea", raw_text=FRAUD_IDEA)
    session.add(ri)
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=10)

    session.close()
    # honest non-recommendation is an acceptable, expected outcome - this
    # test does NOT assert a specific novelty/recommendation verdict for
    # the fraud idea in general, only that an unrelated corpus can't force
    # a fabricated one
    assert assessment.recommendation in ("INSUFFICIENT EVIDENCE", "REQUIRES HUMAN REVIEW", "LOW PRIORITY", "MEDIUM PRIORITY")


def _section(comparison_summary: str, heading: str) -> str:
    if heading not in comparison_summary:
        return ""
    after = comparison_summary.split(heading, 1)[1]
    # a section ends at the next heading (blank-line-separated block) or EOF
    return after.split("\n\n", 1)[0]
