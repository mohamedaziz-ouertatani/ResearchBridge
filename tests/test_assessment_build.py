from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest

from sqlalchemy import select

from researchbridge.assessment.build import build_assessment
from researchbridge.db.models import (
    EMBEDDING_DIM,
    AnalysisClaim,
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
    # a relevant paper WAS retrieved (near-exact title match) - applications.py
    # gates relevance on distance alone, not on whether the paper has claims,
    # so a claim-less paper still yields "no_evidence" ([]), not "not_assessed"
    # (None) - see test_potential_applications_is_empty_list_not_null_when_
    # relevant_papers_have_no_application for the same distinction targeted
    # directly, and novelty.py's assess_novelty for the DIFFERENT, claims-
    # gated criterion that makes novelty_level "not_assessed" here instead.
    assert assessment.potential_applications == []
    assert assessment.potential_applications_status == "no_evidence"
    assert assessment.recommendation == "INSUFFICIENT EVIDENCE"  # honest non-recommendation, not silently None
    assert assessment.confidence == "low"
    assert assessment.completed_at is not None


def test_potential_applications_is_empty_list_not_null_when_relevant_papers_have_no_application(
    session_factory, embedder
) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "method", "an unrelated method claim")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    # relevant paper WAS retrieved (near-exact title match) but stated no
    # application claim - this must be distinguishable from "nothing relevant
    # was retrieved at all", which stays None (see the pre-existing
    # test_assessment_defaults_unassessed_fields_rather_than_fabricating)
    assert assessment.potential_applications == []
    assert assessment.potential_applications_status == "no_evidence"


def test_comparison_creates_a_fact_analysis_claim(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "limitations", "evaluated only in offline settings")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    claim = session.execute(
        select(AnalysisClaim).where(
            AnalysisClaim.source_table == "research_assessments", AnalysisClaim.source_id == assessment.id
        )
    ).scalars().all()
    session.close()
    comparison_claims = [c for c in claim if c.claim_type == "fact"]
    assert len(comparison_claims) == 1
    assert "evaluated only in offline settings" in comparison_claims[0].claim_text


def test_applications_creates_an_opportunity_analysis_claim(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "applications", "real-time payment fraud screening")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    claims = session.execute(
        select(AnalysisClaim).where(
            AnalysisClaim.source_table == "research_assessments", AnalysisClaim.source_id == assessment.id
        )
    ).scalars().all()
    session.close()
    application_claims = [c for c in claims if "real-time payment fraud screening" in c.claim_text]
    assert len(application_claims) == 1
    assert application_claims[0].claim_type == "opportunity"


def test_research_gap_claim_is_skipped_when_the_gap_is_reused_from_a_candidate_gap(
    session_factory, embedder, monkeypatch
) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "limitations", "evaluated only in offline settings")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    import researchbridge.assessment.build as build_module
    from researchbridge.assessment.gap import GapAssessmentResult

    reused_gap = GapAssessmentResult(
        source="reused_candidate_gap", text="a reused gap", candidate_gap_id=None,
        evidence_ids=[], is_closely_grounded=True, status="found",
    )
    monkeypatch.setattr(build_module, "assess_research_gap", lambda *a, **k: reused_gap)

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    claims = session.execute(
        select(AnalysisClaim).where(
            AnalysisClaim.source_table == "research_assessments", AnalysisClaim.source_id == assessment.id
        )
    ).scalars().all()
    session.close()
    assert all(c.claim_text != "a reused gap" for c in claims)


def test_document_input_uses_extracted_representation_as_the_query(session_factory, embedder, monkeypatch) -> None:
    session = session_factory()
    _paper(session, embedder, "p1", "graph transformers for fraud detection")
    ri = ResearchInput(
        id=uuid.uuid4(), input_type="document", raw_text="the raw uploaded document dump",
        source_filename="paper.pdf",
    )
    session.add(ri)
    session.flush()
    session.commit()

    import researchbridge.assessment.build as build_module

    monkeypatch.setattr(
        build_module, "build_research_representation", lambda raw_text, embedder: "the extracted representation"
    )

    build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    # the query embedding call should have used the representation, not the raw dump
    assert ["the extracted representation"] in embedder.calls
    assert ["the raw uploaded document dump"] not in embedder.calls


def test_idea_input_still_uses_raw_text_directly(session_factory, embedder) -> None:
    session = session_factory()
    _paper(session, embedder, "p1", "graph transformers for fraud detection")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert ["graph transformers for fraud detection"] in embedder.calls


def test_raises_for_unknown_research_input(session_factory, embedder) -> None:
    session = session_factory()
    with pytest.raises(ValueError):
        build_assessment(session, uuid.uuid4(), embedder)
    session.close()


def test_novelty_is_low_when_dimension_coverage_is_well_established(session_factory, embedder) -> None:
    # dimension coverage is now computed whenever dimensions can be extracted
    # from the query text, and drives novelty.level via the aggregate rule
    # (see assessment/novelty.py) rather than nearest_distance alone. Under
    # that rule a dimension can only reach "established" with 2+ distinct
    # corroborating papers (by title - the same identity coverage.py and
    # novelty.py both key on), mirroring feasibility.py's own multi-source
    # requirement - so this fixture needs two DIFFERENTLY-TITLED papers
    # whose claims jointly cover both RAKE-extracted dimensions ("graph
    # transformers", "fraud detection"). The FakeEmbedder here is a pure
    # text hash, so a second paper can only land at the same retrieval
    # distance as the query if its embedding vector is set directly rather
    # than derived from its own (necessarily different) title.
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "method", "graph transformers")
    _claim(session, paper, "results", "fraud detection")

    paper2 = Paper(
        id=uuid.uuid4(), source="fake", source_id="p2", title="A Second Related Paper", abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper2)
    session.flush()
    [query_vector] = embedder.embed_texts(["graph transformers for fraud detection"])
    session.add(
        Embedding(paper_id=paper2.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name, vector=query_vector)
    )
    _claim(session, paper2, "dataset", "graph transformers")
    _claim(session, paper2, "applications", "fraud detection")

    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.novelty_level == "low"
    assert "Dimension coverage:" in assessment.novelty_reasoning
    assert "graph transformers -> established" in assessment.novelty_reasoning
    assert "fraud detection -> established" in assessment.novelty_reasoning


def test_novelty_is_high_when_a_single_paper_cannot_corroborate_dimension_coverage(session_factory, embedder) -> None:
    # this is the behavior change the redesign targets: a single retrieved
    # paper at distance 0.0 (an exact title match) used to guarantee "low"
    # novelty on its own. Now, with dimensions extracted, a single paper's
    # claims can only ever reach "weak_evidence" per dimension (coverage.py
    # requires 2+ distinct corroborating papers for "established") - so
    # novelty correctly reads "high" here even though the nearest paper is
    # an exact match, because the evidence backing that match's own
    # dimensions is thin, not because closeness itself was ignored.
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "limitations", "evaluated only in offline settings")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.novelty_level == "high"
    assert "Dimension coverage:" in assessment.novelty_reasoning


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
    # a relevant paper WAS retrieved and checked - it just had no gap-signaling
    # claim, distinct from "no relevant papers to check at all" - see
    # assessment/gap.py's status field
    assert assessment.research_gap_source == "checked_no_gap_found"


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
    assert assessment.potential_applications_status == "found"


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


def test_potential_applications_is_empty_list_without_any_applications_claim(session_factory, embedder) -> None:
    # a relevant paper WAS retrieved, it just stated no application claim -
    # distinct from no relevant papers at all, which stays None (see
    # test_potential_applications_is_empty_list_not_null_when_relevant_papers_have_no_application
    # and the not_assessed case covered elsewhere in this file)
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "limitations", "some unrelated limitation")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.potential_applications == []


def test_technical_feasibility_is_set_from_the_neighborhood(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "method", "a graph attention mechanism")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.technical_feasibility_level == "medium"
    assert paper.title in assessment.technical_feasibility_reasoning


def test_feasibility_evidence_is_linked_with_role_feasibility(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    evidence_id = _claim(session, paper, "method", "a graph attention mechanism")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    links = (
        session.query(ResearchAssessmentEvidence)
        .filter_by(research_assessment_id=assessment.id, role="feasibility")
        .all()
    )
    session.close()
    assert {link.evidence_id for link in links} == {evidence_id}


def test_potential_opportunities_stays_null_by_design(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "applications", "real-time payment fraud screening")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    # deliberately NULL even with a strong application present - see assessment/opportunities.py
    assert assessment.potential_opportunities is None


def test_risks_and_limitations_is_set_from_the_neighborhood(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "limitations", "evaluated only on offline datasets")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert "evaluated only on offline datasets" in assessment.risks_and_limitations


def test_risks_evidence_is_linked_with_role_risk(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    evidence_id = _claim(session, paper, "limitations", "evaluated only on offline datasets")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    links = (
        session.query(ResearchAssessmentEvidence)
        .filter_by(research_assessment_id=assessment.id, role="risk")
        .all()
    )
    session.close()
    assert {link.evidence_id for link in links} == {evidence_id}


def test_recommendation_and_confidence_are_set_from_the_other_signals(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "research_gap", "no real-time evaluation exists")
    _claim(session, paper, "method", "a graph attention mechanism")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.recommendation is not None
    assert assessment.confidence is not None


def test_recommendation_not_boosted_by_a_gap_that_is_not_closely_grounded(session_factory, embedder, monkeypatch) -> None:
    # a gap found only via the looser relevance band (is_closely_grounded=False)
    # must not count as a strong "gap found" signal for the recommendation,
    # even though the report field still surfaces the text
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "method", "a graph attention mechanism")
    _claim(session, paper, "method", "a second graph attention mechanism")  # 2 distinct method claims -> high feasibility needs 2 papers, keep simple: not required here
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    import researchbridge.assessment.build as build_module
    from researchbridge.assessment.gap import GapAssessmentResult
    from researchbridge.assessment.novelty import NoveltyResult

    weakly_grounded_gap = GapAssessmentResult(
        source="input_specific", text="a weakly grounded gap", candidate_gap_id=None,
        evidence_ids=[], is_closely_grounded=False,
    )
    monkeypatch.setattr(build_module, "assess_research_gap", lambda *a, **k: weakly_grounded_gap)
    monkeypatch.setattr(
        build_module,
        "assess_novelty",
        lambda *a, **k: NoveltyResult(level="high", reasoning="no close match found", evidence_ids=[]),
    )

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.novelty_level == "high"
    assert assessment.research_gap_text == "a weakly grounded gap"  # report field still shows it
    assert assessment.recommendation != "HIGH PRIORITY"  # but it doesn't count toward the top-line call


def test_recommendation_is_insufficient_evidence_with_no_signal_at_all(session_factory, embedder) -> None:
    session = session_factory()
    ri = _research_input(session, "an idea with no related papers in the corpus")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.recommendation == "INSUFFICIENT EVIDENCE"
    assert assessment.confidence == "low"
    # nothing at all was retrieved here (no paper in the corpus is even
    # distantly related) - the one case where potential_applications is
    # genuinely None/"not_assessed", not "no_evidence" ([])
    assert assessment.potential_applications is None
    assert assessment.potential_applications_status == "not_assessed"


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


def test_novelty_reasoning_includes_dimension_coverage_when_dimensions_are_extracted(
    session_factory, embedder
) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1", "federated learning for fraud detection across banks")
    _claim(session, paper, "method", "a federated learning method for fraud detection")
    ri = _research_input(
        session,
        "A federated learning system for detecting financial fraud across multiple banks "
        "while remaining robust to concept drift and class imbalance.",
    )
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert "Dimension coverage:" in assessment.novelty_reasoning


def test_recommendation_reasoning_is_persisted_and_derivable_at_read_time(session_factory, embedder) -> None:
    # recommendation reasoning is NOT a new column - it's derived at read
    # time from novelty_level/novelty_reasoning/research_gap_text/
    # technical_feasibility_level/recommendation/confidence, all of which
    # ARE already persisted. This test confirms build_assessment still
    # populates every one of those source fields (the narrative builder
    # itself is exercised in the export/API tests).
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "limitations", "evaluated only in offline settings")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.novelty_level is not None
    assert assessment.technical_feasibility_level is not None
    assert assessment.recommendation is not None
    assert assessment.confidence is not None


# --- enable_llm_stages: inline application relevance filtering + opportunity
# synthesis (2026-09-04), both OFF unless explicitly requested - every test
# above this point implicitly covers enable_llm_stages=False (the default)
# by never passing it at all. These mock both Ollama call sites directly
# rather than running a real model, same as test_application_relevance.py/
# test_opportunity_synthesis.py.


def _mock_ollama(monkeypatch: pytest.MonkeyPatch, relevance_content: str, opportunity_content: str) -> None:
    relevance_response = Mock()
    relevance_response.json.return_value = {"message": {"content": relevance_content}}
    relevance_response.raise_for_status = Mock()
    monkeypatch.setattr(
        "researchbridge.assessment.application_relevance.requests.post", Mock(return_value=relevance_response)
    )

    opportunity_response = Mock()
    opportunity_response.json.return_value = {"message": {"content": opportunity_content}}
    opportunity_response.raise_for_status = Mock()
    monkeypatch.setattr(
        "researchbridge.assessment.opportunity_synthesis.requests.post", Mock(return_value=opportunity_response)
    )


def test_llm_stages_disabled_by_default_even_with_ollama_env_enabled(
    session_factory, embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # OLLAMA_ENABLED=true alone must NOT be enough to trigger either LLM
    # stage - only an explicit enable_llm_stages=True does, see build.py's
    # own docstring for why (a leaked dev .env value must never silently
    # turn a plain build_assessment() call into a network-dependent one)
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    mock_post = Mock(side_effect=AssertionError("Ollama must not be called when enable_llm_stages is unset"))
    monkeypatch.setattr("researchbridge.assessment.application_relevance.requests.post", mock_post)
    monkeypatch.setattr("researchbridge.assessment.opportunity_synthesis.requests.post", mock_post)
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "applications", "real-time payment fraud screening")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5)

    session.close()
    assert assessment.potential_applications[0]["application"] == "real-time payment fraud screening"
    assert assessment.potential_opportunities is None


def test_llm_stages_filters_applications_and_synthesizes_opportunities_when_enabled(
    session_factory, embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    _mock_ollama(
        monkeypatch,
        relevance_content="1: relevant",
        opportunity_content=(
            "Direct: real-time fraud screening API for banks [1]\n"
            "Adjacent: a broader fraud-risk monitoring platform [1]\n"
            "Speculative: an industry-wide fraud intelligence network [1]"
        ),
    )
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "applications", "real-time payment fraud screening")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5, enable_llm_stages=True)

    session.close()
    assert assessment.potential_applications[0]["application"] == "real-time payment fraud screening"
    assert assessment.potential_opportunities is not None
    assert [o["tier"] for o in assessment.potential_opportunities] == ["direct", "adjacent", "speculative"]


def test_llm_stages_application_filter_drops_irrelevant_candidates_when_enabled(
    session_factory, embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    relevance_response = Mock()
    relevance_response.json.return_value = {"message": {"content": "1: irrelevant"}}
    relevance_response.raise_for_status = Mock()
    monkeypatch.setattr(
        "researchbridge.assessment.application_relevance.requests.post", Mock(return_value=relevance_response)
    )
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "applications", "real-time payment fraud screening")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5, enable_llm_stages=True)

    session.close()
    # filtered down to nothing -> no_evidence status -> empty list, and
    # opportunity synthesis is never attempted with zero applications
    assert assessment.potential_applications == []
    assert assessment.potential_opportunities is None


def test_llm_stages_fail_open_on_application_filter_when_ollama_unreachable(
    session_factory, embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    import requests

    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    monkeypatch.setattr(
        "researchbridge.assessment.application_relevance.requests.post",
        Mock(side_effect=requests.ConnectionError("connection refused")),
    )
    monkeypatch.setattr(
        "researchbridge.assessment.opportunity_synthesis.requests.post",
        Mock(side_effect=requests.ConnectionError("connection refused")),
    )
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "applications", "real-time payment fraud screening")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5, enable_llm_stages=True)

    session.close()
    # relevance filtering failed -> keeps the deterministic, unfiltered
    # application; opportunity synthesis also failed -> falls back to NULL
    assert assessment.potential_applications[0]["application"] == "real-time payment fraud screening"
    assert assessment.potential_opportunities is None


def test_llm_stages_opportunity_evidence_is_linked_with_role_opportunity(
    session_factory, embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    _mock_ollama(
        monkeypatch,
        relevance_content="1: relevant",
        opportunity_content=(
            "Direct: real-time fraud screening API for banks [1]\n"
            "Adjacent: a broader fraud-risk monitoring platform [1]\n"
            "Speculative: an industry-wide fraud intelligence network [1]"
        ),
    )
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    evidence_id = _claim(session, paper, "applications", "real-time payment fraud screening")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5, enable_llm_stages=True)

    opportunity_links = (
        session.query(ResearchAssessmentEvidence)
        .filter_by(research_assessment_id=assessment.id, role="opportunity")
        .all()
    )
    session.close()
    assert {link.evidence_id for link in opportunity_links} == {evidence_id}


def test_llm_stages_splits_speculative_opportunities_into_a_separate_claim(
    session_factory, embedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", "true")
    _mock_ollama(
        monkeypatch,
        relevance_content="1: relevant",
        opportunity_content=(
            "Direct: real-time fraud screening API for banks [1]\n"
            "Adjacent: a broader fraud-risk monitoring platform [1]\n"
            "Speculative: an industry-wide fraud intelligence network [1]"
        ),
    )
    session = session_factory()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "applications", "real-time payment fraud screening")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=5, enable_llm_stages=True)

    claims = session.execute(
        select(AnalysisClaim).where(
            AnalysisClaim.source_table == "research_assessments", AnalysisClaim.source_id == assessment.id
        )
    ).scalars().all()
    session.close()

    speculation_claims = [c for c in claims if c.claim_type == "speculation"]
    non_application_opportunity_claims = [
        c for c in claims if c.claim_type == "opportunity" and "direct" in c.claim_text.lower()
    ]
    assert len(speculation_claims) == 1
    assert "industry-wide fraud intelligence network" in speculation_claims[0].claim_text
    assert len(non_application_opportunity_claims) == 1
    opportunity_claim = non_application_opportunity_claims[0]
    assert "industry-wide" not in opportunity_claim.claim_text
    assert "real-time fraud screening API" in opportunity_claim.claim_text
    assert "broader fraud-risk monitoring platform" in opportunity_claim.claim_text
