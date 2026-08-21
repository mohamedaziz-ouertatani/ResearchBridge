from __future__ import annotations

import uuid

from researchbridge.assessment.novelty import assess_novelty


def _claims(*texts: str) -> list[tuple[str, str, uuid.UUID]]:
    return [("limitations", text, uuid.uuid4()) for text in texts]


def test_not_assessed_when_no_retrieved_paper_has_claims() -> None:
    result = assess_novelty([("Paper A", 0.5, [])])

    assert result.level == "not_assessed"
    assert result.evidence_ids == []


def test_not_assessed_when_no_papers_retrieved_at_all() -> None:
    result = assess_novelty([])

    assert result.level == "not_assessed"


def test_low_novelty_when_nearest_evidenced_paper_is_a_close_match() -> None:
    result = assess_novelty([("Existing Solution Paper", 0.1, _claims("tested only offline"))])

    assert result.level == "low"
    assert "Existing Solution Paper" in result.reasoning
    assert len(result.evidence_ids) == 1


def test_medium_novelty_when_nearest_evidenced_paper_is_moderately_related() -> None:
    result = assess_novelty([("Somewhat Related Paper", 0.5, _claims("a moderately related claim"))])

    assert result.level == "medium"
    assert "Somewhat Related Paper" in result.reasoning


def test_high_novelty_when_nearest_evidenced_paper_is_distant() -> None:
    result = assess_novelty([("Distant Paper", 0.9, _claims("a distantly related claim"))])

    assert result.level == "high"
    assert "Distant Paper" in result.reasoning
    assert "not confirmed" in result.reasoning.lower() or "not confirmed novelty" in result.reasoning.lower()


def test_high_novelty_reasoning_never_claims_absolute_novelty() -> None:
    result = assess_novelty([("Distant Paper", 0.9, _claims("a distantly related claim"))])

    lowered = result.reasoning.lower()
    assert "this idea is novel" not in lowered
    assert "genuinely novel" not in lowered
    assert "not confirmed" in lowered  # the actual hedge this reasoning must contain


def test_papers_without_claims_are_skipped_when_choosing_the_nearest(session_factory=None) -> None:
    # nearest paper (distance 0.05) has no claims, so the second one should drive the result
    result = assess_novelty(
        [
            ("No Evidence Paper", 0.05, []),
            ("Has Evidence Paper", 0.5, _claims("a moderately related claim")),
        ]
    )

    assert result.level == "medium"
    assert "Has Evidence Paper" in result.reasoning
    assert "No Evidence Paper" not in result.reasoning


def test_evidence_ids_include_up_to_top_n_evidenced_papers() -> None:
    result = assess_novelty(
        [
            ("Paper 1", 0.1, _claims("claim one")),
            ("Paper 2", 0.2, _claims("claim two")),
            ("Paper 3", 0.3, _claims("claim three")),
            ("Paper 4", 0.4, _claims("claim four")),
        ]
    )

    assert len(result.evidence_ids) == 3  # top 3 only, per TOP_N_FOR_EVIDENCE
