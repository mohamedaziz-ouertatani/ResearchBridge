from __future__ import annotations

import uuid

from researchbridge.gaps.signals import (
    OWN_CONTRIBUTION_OVERLAP_THRESHOLD,
    ClaimClassification,
    classify_claim,
    cosine_similarity,
)


def test_strong_gap_with_field_scope_and_no_negative_signal_is_anchor() -> None:
    text = "Despite recent progress, robustness to adversarial perturbations remains unresolved for existing methods."
    result = classify_claim(text, "research_gap", gap_tier="strong", own_contribution_overlap=0.0)

    assert result.role == "anchor"
    assert result.self_resolution is False
    assert result.field_scope is True


def test_strong_gap_without_field_scope_language_is_only_supporting() -> None:
    # unambiguous unresolved-language, but no explicit third-person/field framing -
    # still real signal, just not enough alone to anchor a strong_gap
    text = "This remains an open problem for future work."
    result = classify_claim(text, "research_gap", gap_tier="strong", own_contribution_overlap=0.0)

    assert result.role == "supporting"


def test_self_resolution_language_downgrades_a_strong_anchor_to_supporting() -> None:
    text = "This paper investigates whether large language models can help bridge this gap."
    result = classify_claim(text, "research_gap", gap_tier="strong", own_contribution_overlap=0.0)

    assert result.self_resolution is True
    assert result.role != "anchor"


def test_high_own_contribution_overlap_alone_downgrades_but_does_not_exclude() -> None:
    # the CentaurBench-adjacent case for a paper that only PARTIALLY overlaps -
    # e.g. a benchmark/evaluation paper: overlap is real negative evidence,
    # but on its own (no self-resolution language) it must not fully exclude
    # the claim, per the "benchmarking is not solving" requirement
    text = "No existing benchmark evaluates robustness under realistic table perturbations."
    result = classify_claim(
        text, "research_gap", gap_tier="strong",
        own_contribution_overlap=OWN_CONTRIBUTION_OVERLAP_THRESHOLD + 0.1,
    )

    assert result.self_resolution is False
    assert result.role == "supporting"  # demoted from anchor, not excluded


def test_self_resolution_and_high_overlap_together_exclude_as_motivation() -> None:
    # both signals firing together is the "prior work has X; we address X"
    # pattern the design calls out explicitly - this is the one case that
    # should fully exclude the claim from clustering
    text = "This paper investigates whether large language models can help bridge this gap."
    result = classify_claim(
        text, "research_gap", gap_tier="strong",
        own_contribution_overlap=OWN_CONTRIBUTION_OVERLAP_THRESHOLD + 0.1,
    )

    assert result.role == "motivation"


def test_weak_gap_tier_can_only_ever_be_supporting_or_motivation_never_anchor() -> None:
    text = "This finding suggests a promising direction for future research in the area."
    result = classify_claim(text, "research_gap", gap_tier="weak", own_contribution_overlap=0.0)

    assert result.role == "supporting"


def test_limitations_claim_type_is_always_treated_as_weak_tier() -> None:
    # limitations has no strong/weak split of its own (validation.py never
    # sets a tier for it) - by design it can corroborate but never single-
    # handedly anchor a strong_gap
    text = "However, the approach does not scale to graphs with more than a million nodes despite existing methods."
    result = classify_claim(text, "limitations", gap_tier=None, own_contribution_overlap=0.0)

    assert result.role == "supporting"


def test_limitations_claim_with_self_resolution_and_overlap_is_motivation() -> None:
    text = "We propose a graph-based method to address this limitation."
    result = classify_claim(
        text, "limitations", gap_tier=None,
        own_contribution_overlap=OWN_CONTRIBUTION_OVERLAP_THRESHOLD + 0.1,
    )

    assert result.role == "motivation"


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


# Task 3: classify_cluster tests

from researchbridge.gaps.signals import ClassifiedMember, classify_cluster


def _member(paper_id: uuid.UUID | None, role: str, overlap: float = 0.0) -> ClassifiedMember:
    return ClassifiedMember(
        paper_id=paper_id or uuid.uuid4(),
        evidence_id=uuid.uuid4(),
        text="claim text",
        classification=ClaimClassification(
            role=role, self_resolution=False, field_scope=(role == "anchor"), own_contribution_overlap=overlap
        ),
    )


def test_two_anchor_papers_is_strong_gap() -> None:
    members = [_member(None, "anchor"), _member(None, "anchor"), _member(None, "supporting")]

    assert classify_cluster(members, min_cluster_size=3) == "strong_gap"


def test_one_anchor_with_enough_independent_support_is_potential_gap() -> None:
    members = [
        _member(None, "anchor"),
        _member(None, "supporting", overlap=0.1),
        _member(None, "supporting", overlap=0.1),
    ]

    assert classify_cluster(members, min_cluster_size=3) == "potential_gap"


def test_one_anchor_with_only_overlap_flagged_support_is_insufficient() -> None:
    # the anchor paper alone can't clear min_cluster_size, and the
    # "corroborating" papers are all explainable by their own contribution -
    # not independent corroboration of anything unresolved
    members = [
        _member(None, "anchor"),
        _member(None, "supporting", overlap=0.9),
        _member(None, "supporting", overlap=0.9),
    ]

    assert classify_cluster(members, min_cluster_size=3) is None


def test_three_independent_supporting_papers_with_no_anchor_is_known_limitation() -> None:
    members = [
        _member(None, "supporting", overlap=0.1),
        _member(None, "supporting", overlap=0.1),
        _member(None, "supporting", overlap=0.1),
    ]

    assert classify_cluster(members, min_cluster_size=3) == "known_limitation"


def test_all_overlap_flagged_supporting_papers_is_insufficient_evidence() -> None:
    # the FinRCA-Bench pattern: three benchmark papers each independently
    # motivating their own contribution with near-identical "existing
    # benchmarks lack X" phrasing, each demoted for overlapping with their
    # OWN contribution - recurring topic, not corroborated unresolvedness
    members = [
        _member(None, "supporting", overlap=0.9),
        _member(None, "supporting", overlap=0.9),
        _member(None, "supporting", overlap=0.9),
    ]

    assert classify_cluster(members, min_cluster_size=3) is None


def test_only_motivation_members_is_insufficient_evidence() -> None:
    members = [_member(None, "motivation"), _member(None, "motivation"), _member(None, "motivation")]

    assert classify_cluster(members, min_cluster_size=3) is None


def test_same_paper_repeating_does_not_count_twice_toward_the_minimum() -> None:
    same_paper = uuid.uuid4()
    members = [
        _member(same_paper, "supporting", overlap=0.1),
        _member(same_paper, "supporting", overlap=0.1),
        _member(same_paper, "supporting", overlap=0.1),
    ]

    assert classify_cluster(members, min_cluster_size=3) is None
