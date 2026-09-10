"""Tests for build_claim_evidence_graph (src/researchbridge/assessment/claim_graph.py)."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from researchbridge.assessment.claim_graph import build_claim_evidence_graph
from researchbridge.db.models import (
    AnalysisClaim,
    CandidateGap,
    ClaimEvidence,
    Evidence,
    Paper,
    PaperCategory,
    ResearchAssessment,
    ResearchInput,
)


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    # analysis_claims/claim_evidence/candidate_gaps/candidate_gap_evidence and
    # research_assessments/research_inputs aren't in conftest's TRUNCATE list
    s.execute(
        text(
            "TRUNCATE TABLE claim_evidence, analysis_claims, candidate_gap_evidence, candidate_gaps, "
            "research_assessments, research_inputs CASCADE"
        )
    )
    s.commit()
    s.close()


def _add_paper(session, title: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=str(uuid.uuid4()), title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def _add_evidence(session, paper: Paper, text_: str) -> Evidence:
    evidence = Evidence(
        paper_id=paper.id, evidence_type="results", section="results", text=text_,
        extraction_method="hybrid", model_version="v1", confidence="high",
    )
    session.add(evidence)
    session.flush()
    return evidence


def _add_assessment(session, candidate_gap_id: uuid.UUID | None = None) -> ResearchAssessment:
    research_input = ResearchInput(id=uuid.uuid4(), input_type="idea", raw_text="an idea")
    session.add(research_input)
    session.flush()
    assessment = ResearchAssessment(id=uuid.uuid4(), research_input_id=research_input.id, candidate_gap_id=candidate_gap_id)
    session.add(assessment)
    session.flush()
    return assessment


def _add_assessment_claim(session, assessment: ResearchAssessment, evidence: Evidence, relationship: str = "supports") -> AnalysisClaim:
    claim = AnalysisClaim(
        id=uuid.uuid4(), claim_type="fact", claim_text="the comparison summary claim", confidence="high",
        status="pending", source_table="research_assessments", source_id=assessment.id,
    )
    session.add(claim)
    session.flush()
    session.add(ClaimEvidence(claim_id=claim.id, evidence_id=evidence.id, relationship=relationship))
    session.commit()
    return claim


def _add_gap_with_claim(session, seed_paper: Paper, evidence: Evidence) -> tuple[CandidateGap, AnalysisClaim]:
    gap = CandidateGap(
        id=uuid.uuid4(), seed_paper_id=seed_paper.id, observation="a recurring unaddressed limitation",
        gap_type="inference", status="pending", contributing_paper_count=3, similarity_threshold=0.35,
        detection_method="cluster-v1",
    )
    session.add(gap)
    session.flush()
    gap_claim = AnalysisClaim(
        id=uuid.uuid4(), claim_type="inference", claim_text=gap.observation, confidence="medium",
        status=gap.status, source_table="candidate_gaps", source_id=gap.id,
    )
    session.add(gap_claim)
    session.flush()
    session.add(ClaimEvidence(claim_id=gap_claim.id, evidence_id=evidence.id, relationship="supports"))
    session.commit()
    return gap, gap_claim


def test_returns_assessment_claim_node_with_its_evidence_node_and_edge(session) -> None:
    paper = _add_paper(session, "a paper")
    evidence = _add_evidence(session, paper, "quoted passage")
    assessment = _add_assessment(session)
    claim = _add_assessment_claim(session, assessment, evidence)

    graph = build_claim_evidence_graph(session, assessment)

    claim_nodes = [n for n in graph.nodes if n.kind == "claim"]
    evidence_nodes = [n for n in graph.nodes if n.kind == "evidence"]
    assert [n.id for n in claim_nodes] == [str(claim.id)]
    assert claim_nodes[0].claim_type == "fact"
    assert claim_nodes[0].confidence == "high"
    assert [n.id for n in evidence_nodes] == [str(evidence.id)]
    assert evidence_nodes[0].label == "quoted passage"
    assert evidence_nodes[0].paper_title == "a paper"
    assert len(graph.edges) == 1
    assert graph.edges[0].source == str(claim.id)
    assert graph.edges[0].target == str(evidence.id)
    assert graph.edges[0].relationship == "supports"


def test_preserves_contradicts_and_contextualizes_relationships(session) -> None:
    paper = _add_paper(session, "a paper")
    evidence = _add_evidence(session, paper, "a conflicting passage")
    assessment = _add_assessment(session)
    _add_assessment_claim(session, assessment, evidence, relationship="contradicts")

    graph = build_claim_evidence_graph(session, assessment)

    assert graph.edges[0].relationship == "contradicts"


def test_includes_gap_node_with_categories_and_addresses_gap_edges(session) -> None:
    seed_paper = _add_paper(session, "seed paper")
    session.add(PaperCategory(paper_id=seed_paper.id, category="Machine Learning", confidence="high", source="arxiv"))
    session.commit()
    gap_evidence = _add_evidence(session, seed_paper, "gap-supporting passage")
    gap, gap_claim = _add_gap_with_claim(session, seed_paper, gap_evidence)

    other_paper = _add_paper(session, "other paper")
    assessment_evidence = _add_evidence(session, other_paper, "assessment evidence passage")
    assessment = _add_assessment(session, candidate_gap_id=gap.id)
    assessment_claim = _add_assessment_claim(session, assessment, assessment_evidence)

    graph = build_claim_evidence_graph(session, assessment)

    gap_nodes = [n for n in graph.nodes if n.kind == "gap"]
    assert len(gap_nodes) == 1
    assert gap_nodes[0].id == str(gap.id)
    assert gap_nodes[0].gap_status == gap.gap_status
    assert gap_nodes[0].categories == ["Machine Learning"]

    claim_nodes = {n.id for n in graph.nodes if n.kind == "claim"}
    assert claim_nodes == {str(assessment_claim.id), str(gap_claim.id)}

    addresses_edges = [e for e in graph.edges if e.relationship == "addresses_gap"]
    assert [e.source for e in addresses_edges] == [str(assessment_claim.id)]
    assert addresses_edges[0].target == str(gap.id)


def test_no_gap_node_when_assessment_has_no_candidate_gap(session) -> None:
    paper = _add_paper(session, "a paper")
    evidence = _add_evidence(session, paper, "quoted passage")
    assessment = _add_assessment(session)
    _add_assessment_claim(session, assessment, evidence)

    graph = build_claim_evidence_graph(session, assessment)

    assert not any(n.kind == "gap" for n in graph.nodes)
    assert not any(e.relationship == "addresses_gap" for e in graph.edges)


def test_empty_graph_for_assessment_with_no_claims_and_no_gap(session) -> None:
    assessment = _add_assessment(session)

    graph = build_claim_evidence_graph(session, assessment)

    assert graph.nodes == []
    assert graph.edges == []
