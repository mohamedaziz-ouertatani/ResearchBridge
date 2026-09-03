from __future__ import annotations

from researchbridge.assessment.recommendation import assess_recommendation


def test_insufficient_evidence_when_nothing_assessed() -> None:
    result = assess_recommendation(
        novelty_level="not_assessed",
        research_gap_text=None,
        research_gap_is_strong=False,
        technical_feasibility_level="not_assessed",
    )

    assert result.recommendation == "INSUFFICIENT EVIDENCE"
    assert result.confidence == "low"


def test_high_priority_when_novel_gap_found_and_feasible() -> None:
    # recommendation CATEGORY only needs feasibility medium/high and any
    # found gap - unchanged by the confidence-strength fix below
    result = assess_recommendation(
        novelty_level="high",
        research_gap_text="a real gap",
        research_gap_is_strong=True,
        technical_feasibility_level="medium",
    )

    assert result.recommendation == "HIGH PRIORITY"


def test_high_priority_with_medium_novelty_too() -> None:
    result = assess_recommendation(
        novelty_level="medium",
        research_gap_text="a real gap",
        research_gap_is_strong=True,
        technical_feasibility_level="high",
    )

    assert result.recommendation == "HIGH PRIORITY"


def test_low_priority_when_idea_already_closely_covered() -> None:
    result = assess_recommendation(
        novelty_level="low",
        research_gap_text="a real gap",
        research_gap_is_strong=True,
        technical_feasibility_level="high",
    )

    assert result.recommendation == "LOW PRIORITY"


def test_requires_human_review_when_only_one_signal_assessed() -> None:
    result = assess_recommendation(
        novelty_level="not_assessed",
        research_gap_text=None,
        research_gap_is_strong=False,
        technical_feasibility_level="medium",
    )

    assert result.recommendation == "REQUIRES HUMAN REVIEW"
    assert result.confidence == "low"


def test_medium_priority_for_a_partial_but_incomplete_match() -> None:
    result = assess_recommendation(
        novelty_level="medium",
        research_gap_text=None,
        research_gap_is_strong=False,
        technical_feasibility_level="medium",
    )

    assert result.recommendation == "MEDIUM PRIORITY"


# --- confidence: strength, not just presence (2026-09-04 fix) ---------------
# Found live-testing real ideas: a fraud-detection assessment reported "HIGH
# PRIORITY / high confidence" backed by single-source ("medium") feasibility
# and a pure "future research" boilerplate gap sentence (extraction/
# validation.py's weak tier) - every signal had SOME value, so the old
# assessed_count-based formula called it "high confidence" even though none
# of the three signals was actually strong.


def test_confidence_is_high_only_when_all_three_signals_are_strong() -> None:
    result = assess_recommendation(
        novelty_level="medium",
        research_gap_text="a real gap",
        research_gap_is_strong=True,
        technical_feasibility_level="high",
    )

    assert result.confidence == "high"


def test_confidence_drops_to_medium_when_feasibility_is_only_medium() -> None:
    # single-source ("medium") feasibility no longer counts as a strong
    # signal for confidence, even though it still counts for the
    # recommendation category (see test_high_priority_when_novel_gap_found_
    # and_feasible above, same inputs, unaffected recommendation)
    result = assess_recommendation(
        novelty_level="high",
        research_gap_text="a real gap",
        research_gap_is_strong=True,
        technical_feasibility_level="medium",
    )

    assert result.confidence == "medium"


def test_confidence_drops_to_medium_when_gap_is_found_but_not_strong() -> None:
    # a gap was found (still counts for the recommendation category) but
    # research_gap_is_strong=False (e.g. a weak-tier boilerplate "future
    # research" sentence, or a source paper that's only loosely related) -
    # must not count as a strong signal for confidence
    result = assess_recommendation(
        novelty_level="medium",
        research_gap_text="Extensive experiments suggest promising avenues for future research.",
        research_gap_is_strong=False,
        technical_feasibility_level="high",
    )

    assert result.recommendation == "HIGH PRIORITY"  # category unaffected
    assert result.confidence == "medium"  # but confidence reflects the thin gap


def test_confidence_is_low_when_only_one_signal_is_strong() -> None:
    # real production shape: medium feasibility (single source) + a found-
    # but-not-strong gap + assessed novelty - only novelty is actually
    # strong, so confidence must not read "high" just because all three
    # fields happen to have SOME value
    result = assess_recommendation(
        novelty_level="medium",
        research_gap_text="Extensive experiments suggest promising avenues for future research.",
        research_gap_is_strong=False,
        technical_feasibility_level="medium",
    )

    assert result.confidence == "low"


def test_confidence_is_medium_for_two_of_three_strong_signals() -> None:
    result = assess_recommendation(
        novelty_level="medium",
        research_gap_text="a real gap",
        research_gap_is_strong=True,
        technical_feasibility_level="not_assessed",
    )

    assert result.confidence == "medium"


def test_confidence_is_low_for_zero_or_one_strong_signal() -> None:
    result = assess_recommendation(
        novelty_level="not_assessed",
        research_gap_text="a real gap",
        research_gap_is_strong=True,
        technical_feasibility_level="not_assessed",
    )

    assert result.confidence == "low"


def test_never_forces_a_recommendation_when_evidence_is_thin() -> None:
    # Sec 42: a trustworthy system does not always force a recommendation
    result = assess_recommendation(
        novelty_level="not_assessed",
        research_gap_text=None,
        research_gap_is_strong=False,
        technical_feasibility_level="not_assessed",
    )

    assert result.recommendation == "INSUFFICIENT EVIDENCE"


def test_reasoning_names_all_three_signals() -> None:
    result = assess_recommendation(
        novelty_level="medium",
        research_gap_text="a real gap",
        research_gap_is_strong=True,
        technical_feasibility_level="high",
    )
    assert "novelty" in result.reasoning.lower()
    assert "gap" in result.reasoning.lower()
    assert "feasibility" in result.reasoning.lower()


def test_reasoning_flags_a_found_but_not_strong_gap() -> None:
    result = assess_recommendation(
        novelty_level="medium",
        research_gap_text="a generic gap sentence",
        research_gap_is_strong=False,
        technical_feasibility_level="high",
    )
    lowered = result.reasoning.lower()
    assert "gap was found" in lowered
    assert "not strong" in lowered or "loosely related" in lowered or "generic" in lowered


def test_reasoning_says_when_a_signal_was_not_assessed() -> None:
    result = assess_recommendation(
        novelty_level="not_assessed",
        research_gap_text=None,
        research_gap_is_strong=False,
        technical_feasibility_level="medium",
    )
    lowered = result.reasoning.lower()
    assert "novelty" in lowered
    assert "not assessed" in lowered or "not_assessed" in lowered


def test_reasoning_mentions_recommendation_and_confidence() -> None:
    result = assess_recommendation(
        novelty_level="low",
        research_gap_text="a real gap",
        research_gap_is_strong=True,
        technical_feasibility_level="high",
    )
    assert result.recommendation.lower() in result.reasoning.lower()
    assert result.confidence.lower() in result.reasoning.lower()
