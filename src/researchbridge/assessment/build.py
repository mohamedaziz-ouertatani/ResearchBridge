"""First vertical slice of ResearchInput -> ResearchAssessment (blueprint Sec 2A).

ResearchInput -> Retrieve Related Papers -> Compare/Extract -> Basic
Evidence-Grounded Assessment Report.

Deliberately minimal, per the roadmap's build-order guidance (Sec 45):
retrieval reuses search_by_text() unchanged (no input-side extraction yet -
that's the uploaded-paper path, not this idea-only slice), and comparison
reuses each retrieved paper's already-extracted claims rather than running
any new extraction. Every field this doesn't compute (novelty, research
gap, feasibility, recommendation, ...) stays NULL/"not_assessed" - NULL is
preferable to fabricated certainty (Sec 22), and those are later
enrichment passes, not this slice's job.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.assessment.novelty import assess_novelty
from researchbridge.db.models import Evidence, ExtractedClaim, ResearchAssessment, ResearchAssessmentEvidence, ResearchInput
from researchbridge.embedding.base import Embedder
from researchbridge.embedding.search import search_by_text


def build_assessment(
    session: Session,
    research_input_id: uuid.UUID,
    embedder: Embedder,
    top_k: int = 10,
) -> ResearchAssessment:
    research_input = session.get(ResearchInput, research_input_id)
    if research_input is None:
        raise ValueError(f"no research input with id {research_input_id}")

    results = search_by_text(session, research_input.raw_text, embedder, top_k)
    papers_with_claims = [(paper, distance, _claims_for_paper(session, paper.id)) for paper, distance in results]

    summary_parts: list[str] = []
    comparison_evidence_ids: list[uuid.UUID] = []
    for paper, _distance, claims in papers_with_claims:
        if not claims:
            continue
        lines = [f"- {claim_type}: {text}" for claim_type, text, _evidence_id in claims]
        summary_parts.append(f"{paper.title}\n" + "\n".join(lines))
        comparison_evidence_ids.extend(evidence_id for _, _, evidence_id in claims)

    novelty = assess_novelty([(paper.title, distance, claims) for paper, distance, claims in papers_with_claims])

    assessment = ResearchAssessment(
        research_input_id=research_input.id,
        status="completed",
        retrieved_paper_ids=[str(paper.id) for paper, _distance, _claims in papers_with_claims],
        comparison_summary="\n\n".join(summary_parts) if summary_parts else None,
        novelty_level=novelty.level,
        novelty_reasoning=novelty.reasoning,
        completed_at=datetime.now(UTC),
    )
    session.add(assessment)
    session.flush()

    for evidence_id in comparison_evidence_ids:
        session.add(
            ResearchAssessmentEvidence(
                research_assessment_id=assessment.id, evidence_id=evidence_id, role="comparison"
            )
        )
    for evidence_id in novelty.evidence_ids:
        session.add(
            ResearchAssessmentEvidence(research_assessment_id=assessment.id, evidence_id=evidence_id, role="novelty")
        )

    session.commit()
    return assessment


def _claims_for_paper(session: Session, paper_id: uuid.UUID) -> list[tuple[str, str, uuid.UUID]]:
    """(claim_type, text, evidence_id) for one paper's real (non-stub) claims."""
    rows = session.execute(
        select(ExtractedClaim.claim_type, ExtractedClaim.text, ExtractedClaim.evidence_id)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .where(ExtractedClaim.paper_id == paper_id, Evidence.extraction_method != "stub")
    ).all()
    return [(claim_type, text, evidence_id) for claim_type, text, evidence_id in rows]
