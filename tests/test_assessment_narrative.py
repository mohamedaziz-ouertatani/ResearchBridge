from __future__ import annotations

import uuid
from datetime import datetime, timezone

from researchbridge.api.schemas import ResearchAssessmentOut, ResearchInputOut
from researchbridge.assessment.narrative import build_recommendation_narrative


def _assessment(**overrides) -> ResearchAssessmentOut:
    defaults = dict(
        id=uuid.uuid4(),
        created_at=datetime.now(timezone.utc),
        research_input=ResearchInputOut(
            id=uuid.uuid4(),
            input_type="idea",
            raw_text="an idea",
            title=None,
            matched_paper_id=None,
        ),
        status="completed",
        retrieved_paper_ids=[],
        comparison_summary=None,
        novelty_level="medium",
        novelty_reasoning="The closest retrieved paper is moderately related.\n\nDimension coverage:\n  concept drift -> not_found",
        research_gap_text="a stated gap",
        research_gap_source="input_specific",
        candidate_gap_id=None,
        potential_applications=None,
        technical_feasibility_level="high",
        technical_feasibility_reasoning="Two relevant papers document applicable methods.",
        potential_opportunities=None,
        risks_and_limitations=None,
        external_validation_needed="Always required.",
        recommendation="HIGH PRIORITY",
        confidence="high",
        human_reviewed=False,
        evidence=[],
        claims=[],
    )
    defaults.update(overrides)
    return ResearchAssessmentOut(**defaults)


def test_narrative_includes_all_four_signal_lines() -> None:
    narrative = build_recommendation_narrative(_assessment())

    assert "Novelty signal:" in narrative
    assert "Research gap evidence:" in narrative
    assert "Technical grounding:" in narrative
    assert "Recommendation:" in narrative


def test_narrative_carries_the_dimension_coverage_block_through_from_novelty_reasoning() -> None:
    narrative = build_recommendation_narrative(_assessment())
    assert "concept drift -> not_found" in narrative


def test_narrative_distinguishes_not_assessed_from_checked_no_gap_found() -> None:
    not_assessed = build_recommendation_narrative(
        _assessment(research_gap_text=None, research_gap_source="no_relevant_evidence")
    )
    not_found = build_recommendation_narrative(
        _assessment(research_gap_text=None, research_gap_source="checked_no_gap_found")
    )

    assert not_assessed != not_found
    assert "insufficient" in not_assessed.lower() or "not enough" in not_assessed.lower()
    assert "no gap" in not_found.lower() or "none found" in not_found.lower()


def test_narrative_reports_final_recommendation_and_confidence() -> None:
    narrative = build_recommendation_narrative(_assessment(recommendation="LOW PRIORITY", confidence="low"))
    assert "LOW PRIORITY" in narrative
    assert "low" in narrative.lower()
