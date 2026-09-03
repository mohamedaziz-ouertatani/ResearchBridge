from __future__ import annotations

import uuid

from researchbridge.assessment.coverage import DimensionCoverage
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


def test_low_novelty_at_exact_near_distance_boundary() -> None:
    # NEAR_DISTANCE=0.35 is compared with <=, so the boundary value itself
    # must still land "low", not slip into "medium"
    result = assess_novelty([("Boundary Paper", 0.35, _claims("a claim"))])

    assert result.level == "low"


def test_medium_novelty_just_above_near_distance_boundary() -> None:
    result = assess_novelty([("Boundary Paper", 0.351, _claims("a claim"))])

    assert result.level == "medium"


def test_high_novelty_at_exact_far_distance_boundary() -> None:
    # FAR_DISTANCE=0.65 is compared with >=, so the boundary value itself
    # must already land "high", not stay "medium"
    result = assess_novelty([("Boundary Paper", 0.65, _claims("a claim"))])

    assert result.level == "high"


def test_medium_novelty_just_below_far_distance_boundary() -> None:
    result = assess_novelty([("Boundary Paper", 0.649, _claims("a claim"))])

    assert result.level == "medium"


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


def test_dimension_coverage_absent_keeps_legacy_nearest_distance_behavior() -> None:
    # no dimension_coverages passed at all - identical to every pre-existing
    # test in this file, confirming the parameter is truly additive
    result = assess_novelty([("Existing Solution Paper", 0.1, _claims("tested only offline"))])
    assert result.level == "low"
    assert result.dimension_coverage == []


def test_falls_back_to_nearest_distance_when_all_dimensions_not_assessed() -> None:
    coverages = [DimensionCoverage(dimension="concept drift", status="not_assessed")]
    result = assess_novelty(
        [("Existing Solution Paper", 0.1, _claims("tested only offline"))], dimension_coverages=coverages
    )
    assert result.level == "low"  # legacy nearest-distance rule, unaffected by not_assessed-only coverage


def test_low_novelty_when_most_dimensions_are_established() -> None:
    coverages = [
        DimensionCoverage(dimension="federated learning", status="established", evidence_ids=[uuid.uuid4()]),
        DimensionCoverage(dimension="privacy preservation", status="established", evidence_ids=[uuid.uuid4()]),
        DimensionCoverage(dimension="fraud detection", status="established", evidence_ids=[uuid.uuid4()]),
    ]
    result = assess_novelty(
        [("Somewhat Related Paper", 0.5, _claims("a moderately related claim"))], dimension_coverages=coverages
    )
    assert result.level == "low"


def test_high_novelty_when_most_dimensions_are_not_found() -> None:
    coverages = [
        DimensionCoverage(dimension="concept drift", status="not_found"),
        DimensionCoverage(dimension="class imbalance", status="not_found"),
        DimensionCoverage(dimension="cross-bank setting", status="weak_evidence", evidence_ids=[uuid.uuid4()]),
    ]
    result = assess_novelty(
        [("Somewhat Related Paper", 0.5, _claims("a moderately related claim"))], dimension_coverages=coverages
    )
    assert result.level == "high"


def test_high_novelty_reasoning_still_never_claims_absolute_novelty_with_coverage() -> None:
    coverages = [DimensionCoverage(dimension="concept drift", status="not_found")]
    result = assess_novelty(
        [("Distant Paper", 0.5, _claims("a claim"))], dimension_coverages=coverages
    )
    lowered = result.reasoning.lower()
    assert "this idea is novel" not in lowered
    assert "genuinely novel" not in lowered


def test_reasoning_includes_a_dimension_coverage_block() -> None:
    coverages = [
        DimensionCoverage(dimension="concept drift", status="not_found"),
        DimensionCoverage(dimension="federated learning", status="established", evidence_ids=[uuid.uuid4()]),
    ]
    result = assess_novelty([("Paper", 0.5, _claims("a claim"))], dimension_coverages=coverages)

    assert "Dimension coverage:" in result.reasoning
    assert "concept drift -> not_found" in result.reasoning
    assert "federated learning -> established" in result.reasoning


def test_low_novelty_at_exact_aggregate_coverage_boundary() -> None:
    # covered/n=0.7 and established/n=0.5 are both compared with >=, so the
    # exact boundary values must still land "low"
    coverages = [
        DimensionCoverage(dimension=f"established-{i}", status="established", evidence_ids=[uuid.uuid4()])
        for i in range(5)
    ] + [
        DimensionCoverage(dimension=f"partial-{i}", status="partially_addressed", evidence_ids=[uuid.uuid4()])
        for i in range(2)
    ] + [
        DimensionCoverage(dimension=f"not-found-{i}", status="not_found") for i in range(3)
    ]
    result = assess_novelty([("Paper", 0.5, _claims("a claim"))], dimension_coverages=coverages)
    assert result.level == "low"


def test_medium_novelty_just_below_aggregate_coverage_boundary() -> None:
    # one fewer covered dimension than the boundary test above (6/10=0.6
    # covered, still 5/10=0.5 established) must fall out of "low" into "medium"
    coverages = [
        DimensionCoverage(dimension=f"established-{i}", status="established", evidence_ids=[uuid.uuid4()])
        for i in range(5)
    ] + [
        DimensionCoverage(dimension="partial", status="partially_addressed", evidence_ids=[uuid.uuid4()])
    ] + [
        DimensionCoverage(dimension=f"not-found-{i}", status="not_found") for i in range(4)
    ]
    result = assess_novelty([("Paper", 0.5, _claims("a claim"))], dimension_coverages=coverages)
    assert result.level == "medium"


def test_high_novelty_at_exact_aggregate_coverage_boundary() -> None:
    # covered/n=0.3 is compared with <=, so the exact boundary value itself
    # must still land "high" - covered means established or
    # partially_addressed only (weak_evidence does NOT count as covered)
    coverages = [
        DimensionCoverage(dimension=f"partial-{i}", status="partially_addressed", evidence_ids=[uuid.uuid4()])
        for i in range(3)
    ] + [DimensionCoverage(dimension=f"not-found-{i}", status="not_found") for i in range(7)]
    result = assess_novelty([("Paper", 0.5, _claims("a claim"))], dimension_coverages=coverages)
    assert result.level == "high"


def test_medium_novelty_just_above_aggregate_coverage_boundary() -> None:
    coverages = [
        DimensionCoverage(dimension=f"partial-{i}", status="partially_addressed", evidence_ids=[uuid.uuid4()])
        for i in range(4)
    ] + [DimensionCoverage(dimension=f"not-found-{i}", status="not_found") for i in range(6)]
    result = assess_novelty([("Paper", 0.5, _claims("a claim"))], dimension_coverages=coverages)
    assert result.level == "medium"


def test_medium_novelty_when_coverage_is_mixed() -> None:
    coverages = [
        DimensionCoverage(dimension="a", status="established", evidence_ids=[uuid.uuid4()]),
        DimensionCoverage(dimension="b", status="not_found"),
    ]
    result = assess_novelty([("Paper", 0.5, _claims("a claim"))], dimension_coverages=coverages)
    assert result.level == "medium"
