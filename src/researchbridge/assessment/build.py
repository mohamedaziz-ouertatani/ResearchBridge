"""ResearchInput -> ResearchAssessment orchestration (blueprint Sec 2A).

ResearchInput -> Retrieve Related Papers -> Compare/Extract -> Novelty ->
Research Gap -> Applications -> Feasibility -> Opportunities (always NULL
by design, see assessment/opportunities.py) -> Risks -> External
Validation -> Recommendation/Confidence.

Idea-only input, per the roadmap's build-order guidance (Sec 45):
retrieval reuses search_by_text() unchanged (no input-side extraction yet -
that's the uploaded-paper path, not yet built), and every downstream
assess_* function reuses each retrieved paper's already-extracted claims
rather than running any new extraction or calling an LLM. Any field a
sub-assessment can't ground in real evidence stays NULL/"not_assessed" -
NULL is preferable to fabricated certainty (Sec 22).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.assessment.applications import assess_applications
from researchbridge.assessment.coverage import compute_dimension_coverage
from researchbridge.assessment.dimensions import extract_dimensions
from researchbridge.assessment.existing_solutions import build_existing_solutions
from researchbridge.assessment.external_validation import assess_external_validation
from researchbridge.assessment.feasibility import assess_technical_feasibility
from researchbridge.assessment.gap import assess_research_gap
from researchbridge.assessment.novelty import assess_novelty
from researchbridge.assessment.opportunities import assess_opportunities
from researchbridge.assessment.recommendation import assess_recommendation
from researchbridge.assessment.representation import build_research_representation
from researchbridge.assessment.risks import assess_risks
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

    query_text = (
        build_research_representation(research_input.raw_text, embedder)
        if research_input.input_type == "document"
        else research_input.raw_text
    )
    results = search_by_text(session, query_text, embedder, top_k)
    papers_with_claims = [(paper, distance, _claims_for_paper(session, paper.id)) for paper, distance in results]

    dimensions = extract_dimensions(query_text)
    dimension_coverages = compute_dimension_coverage(
        dimensions,
        [(paper.title, distance, claims) for paper, distance, claims in papers_with_claims],
        embedder,
    )

    existing_solutions = build_existing_solutions(
        [(paper.title, claims) for paper, _distance, claims in papers_with_claims if claims]
    )

    novelty = assess_novelty(
        [(paper.title, distance, claims) for paper, distance, claims in papers_with_claims],
        dimension_coverages=dimension_coverages,
    )

    retrieved_paper_uuids = [paper.id for paper, _distance, _claims in papers_with_claims]
    papers_by_distance = [(paper.id, distance) for paper, distance, _claims in papers_with_claims]
    gap = assess_research_gap(session, papers_by_distance, embedder)
    applications = assess_applications(
        [(paper.id, paper.title, distance, claims) for paper, distance, claims in papers_with_claims], embedder
    )
    feasibility = assess_technical_feasibility(session, papers_by_distance)
    opportunities = assess_opportunities(session, papers_by_distance)
    risks = assess_risks(session, papers_by_distance, dimension_coverages=dimension_coverages)
    external_validation_needed = assess_external_validation(has_applications=bool(applications.applications))
    recommendation = assess_recommendation(
        novelty_level=novelty.level,
        # only count the gap as "found" for recommendation purposes if it's
        # closely grounded - the report field still shows gap.text regardless
        research_gap_text=(gap.text if gap.is_closely_grounded else None),
        # a distinct, stricter signal for CONFIDENCE specifically - see
        # gap.py's own docstring on is_strongly_stated for why "found" and
        # "strongly stated" need to stay separate checks
        research_gap_is_strong=gap.is_closely_grounded and gap.is_strongly_stated,
        technical_feasibility_level=feasibility.level,
    )

    assessment = ResearchAssessment(
        research_input_id=research_input.id,
        status="completed",
        retrieved_paper_ids=[str(paper_id) for paper_id in retrieved_paper_uuids],
        comparison_summary=existing_solutions.text,
        novelty_level=novelty.level,
        novelty_reasoning=novelty.reasoning,
        research_gap_text=gap.text,
        research_gap_source=(
            gap.source
            if gap.status == "found"
            else ("no_relevant_evidence" if gap.status == "not_assessed" else "checked_no_gap_found")
        ),
        candidate_gap_id=gap.candidate_gap_id,
        potential_applications=(
            [
                {
                    "application": app.application,
                    "source_paper": app.source_paper,
                    "paper_id": str(app.paper_id),
                    # carried so a later, on-demand opportunity synthesis
                    # (assessment/opportunity_synthesis.py) can link its own
                    # ResearchAssessmentEvidence rows back to the same real
                    # evidence this application already traces to, instead
                    # of inventing new evidence for a field this project
                    # never fabricates grounding for.
                    "evidence_id": str(app.evidence_id),
                }
                for app in applications.applications
            ]
            if applications.status == "found"
            else ([] if applications.status == "no_evidence" else None)
        ),
        technical_feasibility_level=feasibility.level,
        technical_feasibility_reasoning=feasibility.reasoning,
        potential_opportunities=opportunities.opportunities,
        risks_and_limitations=risks.text,
        external_validation_needed=external_validation_needed,
        recommendation=recommendation.recommendation,
        confidence=recommendation.confidence,
        completed_at=datetime.now(UTC),
    )
    session.add(assessment)
    session.flush()

    for evidence_id in existing_solutions.evidence_ids:
        session.add(
            ResearchAssessmentEvidence(
                research_assessment_id=assessment.id, evidence_id=evidence_id, role="comparison"
            )
        )
    for evidence_id in novelty.evidence_ids:
        session.add(
            ResearchAssessmentEvidence(research_assessment_id=assessment.id, evidence_id=evidence_id, role="novelty")
        )
    for evidence_id in gap.evidence_ids:
        session.add(
            ResearchAssessmentEvidence(
                research_assessment_id=assessment.id, evidence_id=evidence_id, role="research_gap"
            )
        )
    for evidence_id in applications.evidence_ids:
        session.add(
            ResearchAssessmentEvidence(
                research_assessment_id=assessment.id, evidence_id=evidence_id, role="application"
            )
        )
    for evidence_id in feasibility.evidence_ids:
        session.add(
            ResearchAssessmentEvidence(
                research_assessment_id=assessment.id, evidence_id=evidence_id, role="feasibility"
            )
        )
    for evidence_id in opportunities.evidence_ids:
        session.add(
            ResearchAssessmentEvidence(
                research_assessment_id=assessment.id, evidence_id=evidence_id, role="opportunity"
            )
        )
    for evidence_id in risks.evidence_ids:
        session.add(
            ResearchAssessmentEvidence(research_assessment_id=assessment.id, evidence_id=evidence_id, role="risk")
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
