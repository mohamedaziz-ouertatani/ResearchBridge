from __future__ import annotations

import uuid

from sqlalchemy import select

from researchbridge.db.models import AnalysisClaim, CandidateGap, ClaimEvidence, Evidence, Paper
from researchbridge.gaps.claims import save_claim_for_gap, sync_claim_status
from researchbridge.gaps.detect import CandidateGapDraft
from researchbridge.gaps.persistence import save_candidate_gaps
from researchbridge.gaps.signals import ClaimClassification


def _seed_paper(session) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id="seed", title="Seed Paper", abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def _evidence(session, paper: Paper) -> Evidence:
    evidence = Evidence(
        paper_id=paper.id, evidence_type="limitations", section=None, text="tested only offline",
        extraction_method="fake", model_version="fake-v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    return evidence


def _draft(seed_paper_id: uuid.UUID, evidence_ids: list[uuid.UUID]) -> CandidateGapDraft:
    return CandidateGapDraft(
        seed_paper_id=seed_paper_id,
        observation="Recurring pattern across related papers (inference): tested only offline",
        contributing_paper_count=len(evidence_ids),
        evidence_ids=evidence_ids,
        gap_status="known_limitation",
        resolution_note=None,
        evidence_roles={
            eid: ClaimClassification(role="supporting", self_resolution=False, field_scope=False, own_contribution_overlap=0.0)
            for eid in evidence_ids
        },
    )


def test_saving_a_candidate_gap_also_creates_a_linked_analysis_claim(session_factory) -> None:
    session = session_factory()
    seed = _seed_paper(session)
    ev1, ev2 = _evidence(session, seed), _evidence(session, seed)
    session.commit()

    draft = _draft(seed.id, [ev1.id, ev2.id])
    saved = save_candidate_gaps(session, [draft], similarity_threshold=0.5)
    gap = saved[0]

    claim = session.execute(
        select(AnalysisClaim).where(AnalysisClaim.source_table == "candidate_gaps", AnalysisClaim.source_id == gap.id)
    ).scalar_one()

    assert claim.claim_type == "inference"
    assert claim.claim_text == gap.observation
    assert claim.confidence == "medium"
    assert claim.status == "pending"

    links = session.execute(select(ClaimEvidence).where(ClaimEvidence.claim_id == claim.id)).scalars().all()
    session.close()
    assert {link.evidence_id for link in links} == {ev1.id, ev2.id}
    assert all(link.relationship == "supports" for link in links)


def test_no_drafts_creates_no_claims(session_factory) -> None:
    session = session_factory()
    saved = save_candidate_gaps(session, [], similarity_threshold=0.5)
    session.close()
    assert saved == []


def test_sync_claim_status_propagates_gap_status_onto_linked_claim(session_factory) -> None:
    session = session_factory()
    seed = _seed_paper(session)
    ev1 = _evidence(session, seed)
    session.commit()

    draft = _draft(seed.id, [ev1.id])
    gap = save_candidate_gaps(session, [draft], similarity_threshold=0.5)[0]

    gap.status = "approved"
    sync_claim_status(session, gap)
    session.commit()

    claim = session.execute(
        select(AnalysisClaim).where(AnalysisClaim.source_table == "candidate_gaps", AnalysisClaim.source_id == gap.id)
    ).scalar_one()
    session.close()
    assert claim.status == "approved"


def test_sync_claim_status_is_a_no_op_when_no_claim_is_linked(session_factory) -> None:
    session = session_factory()
    seed = _seed_paper(session)
    session.commit()

    gap = CandidateGap(
        seed_paper_id=seed.id, observation="obs", gap_type="inference", status="pending",
        contributing_paper_count=1, similarity_threshold=0.5, detection_method="cluster-v2",
    )
    session.add(gap)
    session.flush()

    gap.status = "rejected"
    sync_claim_status(session, gap)  # must not raise
    session.commit()
    session.close()


def test_save_claim_for_gap_does_not_commit(session_factory) -> None:
    session = session_factory()
    seed = _seed_paper(session)
    ev1 = _evidence(session, seed)
    session.commit()

    gap = CandidateGap(
        seed_paper_id=seed.id, observation="obs", gap_type="inference", status="pending",
        contributing_paper_count=1, similarity_threshold=0.5, detection_method="cluster-v2",
    )
    session.add(gap)
    session.flush()

    draft = _draft(seed.id, [ev1.id])
    save_claim_for_gap(session, gap, draft)

    session.rollback()
    claim = session.execute(
        select(AnalysisClaim).where(AnalysisClaim.source_table == "candidate_gaps", AnalysisClaim.source_id == gap.id)
    ).scalar_one_or_none()
    session.close()
    assert claim is None
