from __future__ import annotations

from researchbridge.assessment.external_validation import assess_external_validation


def test_states_not_assessed_for_market_and_economic_impact() -> None:
    text = assess_external_validation(has_applications=False)

    assert "not assessed" in text.lower() or "NOT ASSESSED" in text
    assert "market" in text.lower()


def test_names_the_required_validation() -> None:
    text = assess_external_validation(has_applications=False)

    assert "market research" in text.lower() or "industry" in text.lower()


def test_never_asserts_a_market_potential_number_or_level() -> None:
    text = assess_external_validation(has_applications=False)

    lowered = text.lower()
    assert "high market potential" not in lowered
    assert "strong demand" not in lowered


def test_references_applications_when_present() -> None:
    text = assess_external_validation(has_applications=True)

    assert "application" in text.lower()


def test_is_deterministic() -> None:
    assert assess_external_validation(has_applications=False) == assess_external_validation(has_applications=False)
    assert assess_external_validation(has_applications=True) == assess_external_validation(has_applications=True)
