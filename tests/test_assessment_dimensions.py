from __future__ import annotations

from researchbridge.assessment.dimensions import extract_dimensions

FRAUD_IDEA = (
    "Can we build a privacy-preserving federated learning system for detecting "
    "financial fraud across multiple banks without sharing raw transaction data, "
    "while remaining robust to concept drift and highly imbalanced fraud classes?"
)


def test_empty_text_yields_no_dimensions() -> None:
    assert extract_dimensions("") == []
    assert extract_dimensions("   ") == []


def test_extracts_multiple_dimensions_from_the_fraud_idea() -> None:
    dimensions = extract_dimensions(FRAUD_IDEA)

    labels = [d.label.lower() for d in dimensions]
    assert len(dimensions) >= 3
    assert any("concept drift" in label for label in labels)
    assert any("fraud" in label for label in labels)


def test_dimension_labels_are_verbatim_substrings_of_the_input() -> None:
    dimensions = extract_dimensions(FRAUD_IDEA)

    lowered_source = FRAUD_IDEA.lower()
    for dimension in dimensions:
        assert dimension.label.lower() in lowered_source


def test_respects_max_dimensions_cap() -> None:
    dimensions = extract_dimensions(FRAUD_IDEA, max_dimensions=2)
    assert len(dimensions) <= 2


def test_no_duplicate_or_substring_overlapping_dimensions() -> None:
    dimensions = extract_dimensions(FRAUD_IDEA)
    labels = [d.label.lower() for d in dimensions]

    assert len(labels) == len(set(labels))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j:
                assert a not in b, f"{a!r} is a substring of {b!r}; should have been deduped"


def test_single_short_sentence_still_yields_something() -> None:
    dimensions = extract_dimensions("A lock-free queue for concurrent systems.")
    assert len(dimensions) >= 1
