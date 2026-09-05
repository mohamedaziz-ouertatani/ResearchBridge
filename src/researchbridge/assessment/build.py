"""ResearchInput -> ResearchAssessment orchestration (blueprint Sec 2A).

ResearchInput -> Retrieve Related Papers -> Compare/Extract -> Novelty ->
Research Gap -> Applications -> Feasibility -> Opportunities -> Risks ->
External Validation -> Recommendation/Confidence.

Idea-only input, per the roadmap's build-order guidance (Sec 45):
retrieval reuses search_by_text() unchanged (no input-side extraction yet -
that's the uploaded-paper path, not yet built), and every deterministic
assess_* function reuses each retrieved paper's already-extracted claims
rather than running any new extraction. Any field a sub-assessment can't
ground in real evidence stays NULL/"not_assessed" - NULL is preferable to
fabricated certainty (Sec 22).

enable_llm_stages (2026-09-04): two optional local-LLM stages, both OFF
by default (only the API route layer passes True, from ollama_enabled() -
see api/assessment_routes.py) so this function stays synchronous/
deterministic/no-external-dependency for every existing and future direct
caller (tests, scripts, benchmarks) unless they explicitly opt in - the
previous invariant ("no new model calls") is preserved as an explicit
default, not silently broken by an environment variable leaking into a
process that never asked for it (see opportunity_synthesis.py's docstring
for why an unconditional env-var gate here was rejected: OLLAMA_ENABLED is
commonly true in local dev .env files, which would otherwise turn every
test/script calling this function into a slow, network-dependent one).
When True:
- application_relevance.py's local-LLM judge filters assess_applications()'s
  deterministic candidates down to those genuinely relevant to THIS idea,
  not just topically adjacent (see that module's docstring for the real
  false positives - a flower/irrigation "application" surfaced for a
  sourdough-baking idea - that motivated this; embedding similarity alone
  could not separate genuine matches from these). Fails OPEN: unavailable/
  invalid output keeps assess_applications()'s unfiltered result, per that
  module's own fail-open reasoning.
- opportunity_synthesis.py's local-LLM call synthesizes Direct/Adjacent/
  Speculative product opportunities from the (now-filtered) applications,
  same as the on-demand POST /{id}/opportunities endpoint always did -
  this just runs it automatically instead of requiring a separate manual
  trigger. Fails CLOSED (falls back to assess_opportunities()'s
  deterministic always-NULL default) on unavailability/invalid output,
  matching that module's own reasoning: opportunities IS the entire field
  being generated, so there's no safe partial result to keep.

corpus_idf (2026-09-05): the corpus-wide IDF table feasibility.py's widened
band (see that module's docstring) uses to admit a 0.35-0.40 paper via
claim-level lexical overlap. Optional, defaulting to None -> treated as {}
by assess_technical_feasibility(), meaning the widened band admits nothing
and behavior is identical to before this existed - same "safe until
explicitly supplied" shape as enable_llm_stages above. The API route layer
supplies a real one via api/deps.py's get_corpus_idf() (process-wide
cached, same pattern as get_embedder()); any other caller (tests, scripts)
that doesn't pass one simply gets the pre-existing 0.35-only behavior.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.assessment.application_relevance import ApplicationRelevanceUnavailable, filter_relevant_applications
from researchbridge.assessment.applications import ApplicationsResult, assess_applications
from researchbridge.assessment.claims import (
    render_applications_text,
    render_opportunities_text,
    save_claims_for_assessment,
)
from researchbridge.assessment.coverage import compute_dimension_coverage
from researchbridge.assessment.dimensions import extract_dimensions
from researchbridge.assessment.existing_solutions import build_existing_solutions
from researchbridge.assessment.external_validation import assess_external_validation
from researchbridge.assessment.feasibility import assess_technical_feasibility
from researchbridge.assessment.gap import assess_research_gap
from researchbridge.assessment.language import is_likely_non_latin_script
from researchbridge.assessment.novelty import assess_novelty
from researchbridge.assessment.opportunities import assess_opportunities
from researchbridge.assessment.opportunity_synthesis import (
    OpportunitySynthesisUnavailable,
    SourceApplication,
    synthesize_opportunities,
    to_persisted_opportunities,
)
from researchbridge.assessment.recommendation import assess_recommendation
from researchbridge.assessment.representation import build_research_representation
from researchbridge.assessment.risks import assess_risks
from researchbridge.connectors.epo_patents import EPOPatentConnector
from researchbridge.connectors.world_bank import WorldBankConnector
from researchbridge.db.models import Evidence, ExtractedClaim, ResearchAssessment, ResearchAssessmentEvidence, ResearchInput
from researchbridge.embedding.base import Embedder
from researchbridge.embedding.search import search_by_text


def _build_patent_connector() -> EPOPatentConnector | None:
    consumer_key = os.environ.get("EPO_OPS_CONSUMER_KEY")
    consumer_secret = os.environ.get("EPO_OPS_CONSUMER_SECRET")
    if not consumer_key or not consumer_secret:
        return None
    return EPOPatentConnector(consumer_key=consumer_key, consumer_secret=consumer_secret)


def build_assessment(
    session: Session,
    research_input_id: uuid.UUID,
    embedder: Embedder,
    top_k: int = 10,
    enable_llm_stages: bool = False,
    corpus_idf: dict[str, float] | None = None,
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
        [(paper.title, distance, claims) for paper, distance, claims in papers_with_claims if claims]
    )

    novelty = assess_novelty(
        [(paper.title, distance, claims) for paper, distance, claims in papers_with_claims],
        dimension_coverages=dimension_coverages,
    )
    if is_likely_non_latin_script(research_input.raw_text):
        novelty = replace(
            novelty,
            reasoning=(
                "Note: this idea appears to be written in a non-English/non-Latin-"
                "script language. This corpus and its embedding model are English-"
                "optimized, so retrieval quality (and therefore this novelty signal) "
                "is less reliable here than for English input.\n\n" + novelty.reasoning
            ),
        )

    retrieved_paper_uuids = [paper.id for paper, _distance, _claims in papers_with_claims]
    papers_by_distance = [(paper.id, distance) for paper, distance, _claims in papers_with_claims]
    gap = assess_research_gap(session, papers_by_distance, embedder)
    applications = assess_applications(
        [(paper.id, paper.title, distance, claims) for paper, distance, claims in papers_with_claims], embedder
    )
    if enable_llm_stages and applications.status == "found":
        try:
            filtered = filter_relevant_applications(query_text, applications.applications)
            applications = ApplicationsResult(
                applications=filtered,
                evidence_ids=[app.evidence_id for app in filtered],
                status="found" if filtered else "no_evidence",
            )
        except ApplicationRelevanceUnavailable:
            pass  # fail open - keep the deterministic, unfiltered result

    feasibility = assess_technical_feasibility(session, papers_by_distance, query_text, corpus_idf or {})
    opportunities = assess_opportunities(session, papers_by_distance)

    opportunities_json: list[dict] | None = None
    opportunity_evidence_ids: set[str] = set()
    non_speculative_evidence_ids: set[str] = set()
    speculative_evidence_ids: set[str] = set()
    """Evidence cited by each tier of synthesized opportunity, tracked
    separately (not derived by subtracting one from the combined pool) -
    the same evidence can legitimately support both a modest and a
    speculative reading of the same application, so the two sets can and
    do overlap. Used by assessment/claims.py to route claim_type="opportunity"
    vs "speculation" (see that module's docstring)."""
    if enable_llm_stages and applications.status == "found":
        source_applications = [
            SourceApplication(
                application=app.application,
                source_paper=app.source_paper,
                paper_id=str(app.paper_id),
                evidence_id=str(app.evidence_id),
            )
            for app in applications.applications
        ]
        try:
            synthesis_result = synthesize_opportunities(query_text, source_applications)
            opportunities_json, opportunity_evidence_ids = to_persisted_opportunities(
                source_applications, synthesis_result
            )
            for opp in synthesis_result.opportunities:
                target = speculative_evidence_ids if opp.tier == "speculative" else non_speculative_evidence_ids
                for i in opp.source_application_indices:
                    evidence_id_str = source_applications[i - 1].evidence_id
                    if evidence_id_str is not None:
                        target.add(evidence_id_str)
        except OpportunitySynthesisUnavailable:
            pass  # fail closed - keep assess_opportunities()'s deterministic NULL default

    risks = assess_risks(session, papers_by_distance, dimension_coverages=dimension_coverages)
    external_validation = assess_external_validation(
        title=research_input.title,
        raw_text=research_input.raw_text,
        has_applications=bool(applications.applications),
        patent_connector=_build_patent_connector(),
        market_connector=WorldBankConnector(),
    )
    external_validation_needed = external_validation.text
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

    research_gap_source_value = (
        gap.source
        if gap.status == "found"
        else ("no_relevant_evidence" if gap.status == "not_assessed" else "checked_no_gap_found")
    )

    assessment = ResearchAssessment(
        research_input_id=research_input.id,
        status="completed",
        retrieved_paper_ids=[str(paper_id) for paper_id in retrieved_paper_uuids],
        comparison_summary=existing_solutions.text,
        novelty_level=novelty.level,
        novelty_reasoning=novelty.reasoning,
        research_gap_text=gap.text,
        research_gap_source=research_gap_source_value,
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
        potential_opportunities=(opportunities_json if opportunities_json is not None else opportunities.opportunities),
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
    for evidence_id_str in opportunity_evidence_ids:
        session.add(
            ResearchAssessmentEvidence(
                research_assessment_id=assessment.id, evidence_id=uuid.UUID(evidence_id_str), role="opportunity"
            )
        )
    for evidence_id in risks.evidence_ids:
        session.add(
            ResearchAssessmentEvidence(research_assessment_id=assessment.id, evidence_id=evidence_id, role="risk")
        )

    # Split by tier for the claims layer only (ResearchAssessmentEvidence
    # above stays role="opportunity" for every tier - this split is just
    # for claim_type, see assessment/claims.py's docstring).
    non_speculative_opportunities = [
        o for o in (assessment.potential_opportunities or []) if o["tier"] != "speculative"
    ]
    speculative_opportunities = [o for o in (assessment.potential_opportunities or []) if o["tier"] == "speculative"]
    non_speculative_opportunity_evidence_ids = [uuid.UUID(evidence_id_str) for evidence_id_str in non_speculative_evidence_ids]
    speculative_opportunity_evidence_ids = [uuid.UUID(evidence_id_str) for evidence_id_str in speculative_evidence_ids]

    save_claims_for_assessment(
        session,
        assessment,
        texts_by_role={
            "comparison": existing_solutions.text,
            "novelty": novelty.reasoning,
            # only mirror the gap text when it's this assessment's own
            # finding - a reused candidate gap already has its own claim,
            # see assessment/claims.py's docstring
            "research_gap": (gap.text if research_gap_source_value == "input_specific" else None),
            "application": (
                render_applications_text(assessment.potential_applications)
                if assessment.potential_applications
                else None
            ),
            "opportunity": (
                render_opportunities_text(non_speculative_opportunities) if non_speculative_opportunities else None
            ),
            "speculation": (
                render_opportunities_text(speculative_opportunities) if speculative_opportunities else None
            ),
            "feasibility": feasibility.reasoning,
            "risk": risks.text,
        },
        evidence_by_role={
            "comparison": existing_solutions.evidence_ids,
            "novelty": novelty.evidence_ids,
            "research_gap": gap.evidence_ids,
            "application": applications.evidence_ids,
            "opportunity": list(opportunities.evidence_ids) + non_speculative_opportunity_evidence_ids,
            "speculation": speculative_opportunity_evidence_ids,
            "feasibility": feasibility.evidence_ids,
            "risk": risks.evidence_ids,
        },
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
