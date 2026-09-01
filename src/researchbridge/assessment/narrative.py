"""Read-time recommendation narrative.

The recommendation chain must be inspectable, without adding a new
persisted column - see docs/superpowers/plans/2026-08-29-dimension-aware-
assessment-coverage.md, "Migration implications".

Purely derived from fields ResearchAssessment already persists. The
per-dimension coverage table lives inside novelty_reasoning's own text
(assessment/novelty.py appends it there); this function doesn't recompute
it, just stitches the already-persisted pieces into one readable block.
"""

from __future__ import annotations

from researchbridge.api.schemas import ResearchAssessmentOut

_GAP_SOURCE_LABELS = {
    "reused_candidate_gap": "a reviewed candidate gap was reused",
    "input_specific": "found for this input",
    "no_relevant_evidence": "not assessed - insufficient relevant evidence was retrieved to investigate",
    "checked_no_gap_found": "none found - relevant literature was checked, no gap surfaced",
}


def build_recommendation_narrative(assessment: ResearchAssessmentOut) -> str:
    novelty_line = f"Novelty signal: {assessment.novelty_level}"
    if assessment.novelty_reasoning:
        novelty_line += f"\n{assessment.novelty_reasoning}"

    if assessment.research_gap_text:
        gap_line = f"Research gap evidence: {assessment.research_gap_text}"
    else:
        gap_line = f"Research gap evidence: {_GAP_SOURCE_LABELS.get(assessment.research_gap_source, 'not assessed')}"

    feasibility_line = f"Technical grounding: {assessment.technical_feasibility_level}"
    if assessment.technical_feasibility_reasoning:
        feasibility_line += f"\n{assessment.technical_feasibility_reasoning}"

    recommendation_line = f"Recommendation: {assessment.recommendation}\nConfidence: {assessment.confidence}"

    return "\n\n".join([novelty_line, gap_line, feasibility_line, recommendation_line])
