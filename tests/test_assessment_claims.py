from __future__ import annotations

import uuid

from sqlalchemy import select

from researchbridge.assessment.claims import save_claims_for_assessment
from researchbridge.db.models import AnalysisClaim, ClaimEvidence, Evidence, Paper, ResearchAssessment, ResearchInput


def _research_input(session) -> ResearchInput:
    ri = ResearchInput(id=uuid.uuid4(), input_type="idea", raw_text="an idea")
    session.add(ri)
    session.flush()
    return ri


def _paper(session) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id="p1", title="A Paper", abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def _evidence(session, paper: Paper) -> Evidence:
    evidence = Evidence(
        paper_id=paper.id, evidence_type="limitations", section=None, text="a limitation",
        extraction_method="fake", model_version="fake-v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    return evidence


def _assessment(session, ri: ResearchInput, confidence: str = "medium") -> ResearchAssessment:
    assessment = ResearchAssessment(research_input_id=ri.id, status="completed", confidence=confidence)
    session.add(assessment)
    session.flush()
    return assessment


def test_creates_a_fact_claim_for_comparison_and_inference_for_the_rest(session_factory) -> None:
    session = session_factory()
    ri = _research_input(session)
    paper = _paper(session)
    ev = _evidence(session, paper)
    session.commit()
    assessment = _assessment(session, ri)

    saved = save_claims_for_assessment(
        session,
        assessment,
        texts_by_role={"comparison": "existing solutions text", "novelty": "novelty reasoning"},
        evidence_by_role={"comparison": [ev.id], "novelty": [ev.id]},
    )
    session.commit()

    by_role_type = {c.claim_text: c.claim_type for c in saved}
    session.close()
    assert by_role_type == {"existing solutions text": "fact", "novelty reasoning": "inference"}


def test_skips_roles_with_no_text_or_no_evidence(session_factory) -> None:
    session = session_factory()
    ri = _research_input(session)
    paper = _paper(session)
    ev = _evidence(session, paper)
    session.commit()
    assessment = _assessment(session, ri)

    saved = save_claims_for_assessment(
        session,
        assessment,
        texts_by_role={"comparison": None, "novelty": "has text but no evidence", "feasibility": "has both"},
        evidence_by_role={"novelty": [], "feasibility": [ev.id]},
    )
    session.commit()

    roles = {c.claim_text for c in saved}
    session.close()
    assert roles == {"has both"}


def test_confidence_is_the_assessments_overall_confidence(session_factory) -> None:
    session = session_factory()
    ri = _research_input(session)
    paper = _paper(session)
    ev = _evidence(session, paper)
    session.commit()
    assessment = _assessment(session, ri, confidence="high")

    saved = save_claims_for_assessment(
        session, assessment, texts_by_role={"risk": "a risk"}, evidence_by_role={"risk": [ev.id]}
    )
    session.commit()

    session.close()
    assert saved[0].confidence == "high"


def test_status_is_always_pending(session_factory) -> None:
    session = session_factory()
    ri = _research_input(session)
    paper = _paper(session)
    ev = _evidence(session, paper)
    session.commit()
    assessment = _assessment(session, ri)

    saved = save_claims_for_assessment(
        session, assessment, texts_by_role={"risk": "a risk"}, evidence_by_role={"risk": [ev.id]}
    )
    session.commit()

    session.close()
    assert saved[0].status == "pending"


def test_source_table_and_id_point_back_to_the_assessment(session_factory) -> None:
    session = session_factory()
    ri = _research_input(session)
    paper = _paper(session)
    ev = _evidence(session, paper)
    session.commit()
    assessment = _assessment(session, ri)

    saved = save_claims_for_assessment(
        session, assessment, texts_by_role={"risk": "a risk"}, evidence_by_role={"risk": [ev.id]}
    )
    session.commit()

    session.close()
    assert saved[0].source_table == "research_assessments"
    assert saved[0].source_id == assessment.id


def test_creates_one_claim_evidence_row_per_evidence_id(session_factory) -> None:
    session = session_factory()
    ri = _research_input(session)
    paper = _paper(session)
    ev1, ev2 = _evidence(session, paper), _evidence(session, paper)
    session.commit()
    assessment = _assessment(session, ri)

    saved = save_claims_for_assessment(
        session, assessment, texts_by_role={"risk": "a risk"}, evidence_by_role={"risk": [ev1.id, ev2.id]}
    )
    session.commit()

    links = session.execute(select(ClaimEvidence).where(ClaimEvidence.claim_id == saved[0].id)).scalars().all()
    session.close()
    assert {link.evidence_id for link in links} == {ev1.id, ev2.id}
    assert all(link.relationship == "supports" for link in links)


def test_no_roles_populated_creates_nothing(session_factory) -> None:
    session = session_factory()
    ri = _research_input(session)
    session.commit()
    assessment = _assessment(session, ri)

    saved = save_claims_for_assessment(session, assessment, texts_by_role={}, evidence_by_role={})
    session.commit()

    session.close()
    assert saved == []
