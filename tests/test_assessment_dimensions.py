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


def test_citation_marker_does_not_fuse_unrelated_words_into_one_phrase() -> None:
    # Real corpus bug: _TOKEN_RE never captures punctuation, so without a
    # punctuation boundary "(see [1])" simply vanishes from the token
    # stream and "method"/"see" - on either side of the citation, with no
    # stopword between them - silently fused into the nonsense candidate
    # "method see". The module's own docstring already claims RAKE splits
    # at "stopword/punctuation boundaries" - this is that other half.
    text = (
        "We propose a method (see [1]) that, unlike prior work [2][3], achieves "
        "state-of-the-art results by combining transformers with reinforcement learning."
    )
    dimensions = extract_dimensions(text, max_dimensions=20)
    labels = [d.label.lower() for d in dimensions]

    assert "method see" not in labels
    assert "method" in labels


def test_comma_separated_list_splits_into_separate_dimensions() -> None:
    # Without a punctuation boundary, a comma-separated list with no
    # intervening stopword ("attendance, study habits, previous grades")
    # merges into one blended candidate phrase instead of three precise
    # ones, diluting its RAKE score and blurring the embedding used for
    # downstream retrieval matching.
    text = "A system that predicts outcomes with attendance, study habits, previous grades as input."
    dimensions = extract_dimensions(text, max_dimensions=20)
    labels = [d.label.lower() for d in dimensions]

    assert "attendance" in labels
    assert "study habits" in labels
    assert "previous grades" in labels
    assert not any("attendance study habits" in label for label in labels)


def test_them_is_treated_as_a_stopword() -> None:
    # "them" leaked through as a standalone dimension in a real idea
    # ("...flagging unsupported claims before showing them to the
    # clinician") - the stopword list already has it/its/we/our/they/
    # their/he/she/his/her but was missing the object pronoun "them".
    text = "The system flags unsupported claims before showing them to the clinician for review."
    dimensions = extract_dimensions(text, max_dimensions=20)
    labels = [d.label.lower() for d in dimensions]

    assert "them" not in labels
