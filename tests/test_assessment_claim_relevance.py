from __future__ import annotations

from researchbridge.assessment.claim_relevance import claim_overlap


def test_no_shared_tokens_scores_zero() -> None:
    idf = {"federated": 2.0, "learning": 1.5, "sourdough": 3.0, "starter": 3.0}

    score = claim_overlap("federated learning", "sourdough starter", idf)

    assert score == 0.0


def test_shared_specific_terms_score_higher_than_generic_terms() -> None:
    # "federated"/"privacy" are rare (high idf); "system"/"using" are filtered
    # as stopwords, so they must not be able to drive the score up on their own.
    idf = {"federated": 3.0, "privacy": 3.0, "sepsis": 3.0, "icu": 3.0}

    specific_overlap = claim_overlap(
        "federated learning for sepsis prediction in icu patients",
        "we propose a federated learning approach with privacy guarantees",
        idf,
    )
    no_overlap = claim_overlap(
        "federated learning for sepsis prediction in icu patients",
        "an unrelated approach to image compression",
        idf,
    )

    assert specific_overlap > no_overlap
    assert no_overlap == 0.0


def test_stopwords_are_excluded_from_scoring() -> None:
    idf = {"the": 0.01, "and": 0.01, "a": 0.01, "of": 0.01}

    score = claim_overlap("the system and a model of the data", "the and a of", idf)

    assert score == 0.0


def test_unknown_tokens_default_to_zero_idf_weight() -> None:
    idf = {"federated": 3.0}

    score = claim_overlap("federated zzzznovel", "federated zzzznovel", idf)

    assert 0.0 < score <= 1.0


def test_negative_idf_is_clamped_to_zero_not_subtracted() -> None:
    # rank_bm25's idf can go negative for extremely common terms - a negative
    # weight must never make the score negative or invert the ranking.
    idf = {"common": -0.5, "rare": 3.0}

    score = claim_overlap("common rare term", "common rare term", idf)

    assert score >= 0.0


def test_is_deterministic() -> None:
    idf = {"federated": 3.0, "learning": 1.0}
    args = ("federated learning system", "a federated learning approach", idf)

    assert claim_overlap(*args) == claim_overlap(*args)


def test_empty_claim_text_scores_zero() -> None:
    idf = {"federated": 3.0}

    score = claim_overlap("federated learning", "", idf)

    assert score == 0.0


def test_full_overlap_scores_one() -> None:
    idf = {"federated": 3.0, "learning": 2.0}

    score = claim_overlap("federated learning", "federated learning", idf)

    assert score == 1.0
