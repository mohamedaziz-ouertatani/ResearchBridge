from __future__ import annotations

from researchbridge.assessment.recommendation import assess_recommendation


def test_insufficient_evidence_when_nothing_assessed() -> None:
    result = assess_recommendation(
        novelty_level="not_assessed", research_gap_text=None, technical_feasibility_level="not_assessed"
    )

    assert result.recommendation == "INSUFFICIENT EVIDENCE"
    assert result.confidence == "low"


def test_high_priority_when_novel_gap_found_and_feasible() -> None:
    result = assess_recommendation(
        novelty_level="high", research_gap_text="a real gap", technical_feasibility_level="medium"
    )

    assert result.recommendation == "HIGH PRIORITY"
    assert result.confidence == "high"


def test_high_priority_with_medium_novelty_too() -> None:
    result = assess_recommendation(
        novelty_level="medium", research_gap_text="a real gap", technical_feasibility_level="high"
    )

    assert result.recommendation == "HIGH PRIORITY"


def test_low_priority_when_idea_already_closely_covered() -> None:
    result = assess_recommendation(
        novelty_level="low", research_gap_text="a real gap", technical_feasibility_level="high"
    )

    assert result.recommendation == "LOW PRIORITY"


def test_requires_human_review_when_only_one_signal_assessed() -> None:
    result = assess_recommendation(
        novelty_level="not_assessed", research_gap_text=None, technical_feasibility_level="medium"
    )

    assert result.recommendation == "REQUIRES HUMAN REVIEW"
    assert result.confidence == "low"


def test_medium_priority_for_a_partial_but_incomplete_match(session_factory=None) -> None:
    result = assess_recommendation(
        novelty_level="medium", research_gap_text=None, technical_feasibility_level="medium"
    )

    assert result.recommendation == "MEDIUM PRIORITY"
    assert result.confidence == "medium"


def test_confidence_is_high_only_when_all_three_signals_assessed() -> None:
    result = assess_recommendation(
        novelty_level="medium", research_gap_text="a real gap", technical_feasibility_level="medium"
    )

    assert result.confidence == "high"


def test_confidence_is_medium_for_two_of_three_signals() -> None:
    result = assess_recommendation(
        novelty_level="medium", research_gap_text="a real gap", technical_feasibility_level="not_assessed"
    )

    assert result.confidence == "medium"


def test_confidence_is_low_for_zero_or_one_signal() -> None:
    result = assess_recommendation(
        novelty_level="not_assessed", research_gap_text="a real gap", technical_feasibility_level="not_assessed"
    )

    assert result.confidence == "low"


def test_never_forces_a_recommendation_when_evidence_is_thin() -> None:
    # Sec 42: a trustworthy system does not always force a recommendation
    result = assess_recommendation(
        novelty_level="not_assessed", research_gap_text=None, technical_feasibility_level="not_assessed"
    )

    assert result.recommendation == "INSUFFICIENT EVIDENCE"
