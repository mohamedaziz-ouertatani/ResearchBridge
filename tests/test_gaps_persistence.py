from __future__ import annotations

import uuid

from sqlalchemy import select

from researchbridge.db.models import CandidateGap, CandidateGapEvidence, Evidence, Paper
from researchbridge.gaps.detect import CandidateGapDraft
from researchbridge.gaps.persistence import save_candidate_gaps


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


def test_saves_candidate_gap_as_pending_inference(session_factory) -> None:
    session = session_factory()
    seed = _seed_paper(session)
    ev1, ev2, ev3 = _evidence(session, seed), _evidence(session, seed), _evidence(session, seed)
    session.commit()

    draft = CandidateGapDraft(
        seed_paper_id=seed.id,
        observation='Recurring pattern across 3 related papers (inference): "tested only offline"',
        contributing_paper_count=3,
        evidence_ids=[ev1.id, ev2.id, ev3.id],
    )

    saved = save_candidate_gaps(session, [draft], similarity_threshold=0.5)

    assert len(saved) == 1
    gap = session.get(CandidateGap, saved[0].id)
    assert gap.status == "pending"
    assert gap.gap_type == "inference"
    assert gap.contributing_paper_count == 3
    assert gap.similarity_threshold == 0.5
    session.close()


def test_links_every_contributing_evidence_row(session_factory) -> None:
    session = session_factory()
    seed = _seed_paper(session)
    ev1, ev2 = _evidence(session, seed), _evidence(session, seed)
    session.commit()

    draft = CandidateGapDraft(
        seed_paper_id=seed.id, observation="obs", contributing_paper_count=2, evidence_ids=[ev1.id, ev2.id]
    )

    saved = save_candidate_gaps(session, [draft], similarity_threshold=0.4)

    links = session.execute(
        select(CandidateGapEvidence).where(CandidateGapEvidence.candidate_gap_id == saved[0].id)
    ).scalars().all()
    session.close()
    assert {link.evidence_id for link in links} == {ev1.id, ev2.id}


def test_saves_multiple_drafts_independently(session_factory) -> None:
    session = session_factory()
    seed = _seed_paper(session)
    ev1, ev2 = _evidence(session, seed), _evidence(session, seed)
    session.commit()

    drafts = [
        CandidateGapDraft(seed_paper_id=seed.id, observation="first", contributing_paper_count=1, evidence_ids=[ev1.id]),
        CandidateGapDraft(seed_paper_id=seed.id, observation="second", contributing_paper_count=1, evidence_ids=[ev2.id]),
    ]

    saved = save_candidate_gaps(session, drafts, similarity_threshold=0.5)

    session.close()
    assert {g.observation for g in saved} == {"first", "second"}


def test_no_drafts_saves_nothing(session_factory) -> None:
    session = session_factory()
    saved = save_candidate_gaps(session, [], similarity_threshold=0.5)
    session.close()
    assert saved == []
