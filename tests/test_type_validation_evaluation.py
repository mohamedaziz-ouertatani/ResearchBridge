from __future__ import annotations

from pathlib import Path

from researchbridge.benchmark.store import Annotation
from researchbridge.extraction.type_validation_evaluation import evaluate_claim_type_validation


def _annotation(**fields: str) -> Annotation:
    research_gap = fields.pop("research_gap", "")
    return Annotation(
        source_id="p1",
        path=Path("unused.yaml"),
        fields=fields,
        research_gap={"remaining": research_gap},
    )


def test_own_field_correct_text_is_accepted() -> None:
    annotations = [
        _annotation(
            method="We propose a lock-free queue for high throughput systems.",
            results="Our results show a 3x speedup over the baseline.",
        )
    ]
    scores = evaluate_claim_type_validation(annotations, fields=frozenset({"method", "results"}))

    assert scores["method"].own_field_total == 1
    assert scores["method"].own_field_accepted == 1
    assert scores["results"].own_field_total == 1
    assert scores["results"].own_field_accepted == 1


def test_reported_bug_is_caught_as_cross_field_rejection() -> None:
    # exactly the reported failure mode: a results sentence checked against
    # research_gap must be rejected
    annotations = [
        _annotation(
            results="Our model achieves 94.2% AUC and an F1 score of 0.89 on the benchmark.",
            research_gap="Extending this approach to multilingual settings remains an open problem.",
        )
    ]
    scores = evaluate_claim_type_validation(annotations, fields=frozenset({"results", "research_gap"}))

    gap_score = scores["research_gap"]
    assert gap_score.cross_field_total == 1
    assert gap_score.cross_field_rejected == 1
    assert gap_score.cross_field_rejection_rate == 1.0


def test_empty_ground_truth_fields_are_excluded_from_both_measures() -> None:
    annotations = [_annotation(method="", results="Our results show a 3x speedup.")]
    scores = evaluate_claim_type_validation(annotations, fields=frozenset({"method", "results"}))

    assert scores["method"].own_field_total == 0
    assert scores["results"].cross_field_total == 0  # method's text was empty, nothing to cross-check
