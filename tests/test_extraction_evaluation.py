from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from researchbridge.extraction.base import ClaimCandidate
from researchbridge.extraction.evaluation import FieldScore, evaluate

_WORD = re.compile(r"[a-z]+")


@dataclass
class WordOverlapEmbedder:
    """Bag-of-words cosine similarity, standing in for real embeddings.

    Deterministic and legible: similarity is exactly the fraction of shared
    vocabulary, so tests can construct "close paraphrase" vs "unrelated"
    pairs and reason about the resulting score directly, instead of trusting
    an opaque hash-based vector to land above/below a threshold by luck.
    """

    model_name: str = "word-overlap-fake"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        vocab = sorted({w for t in texts for w in _WORD.findall(t.lower())})
        vectors = []
        for t in texts:
            words = set(_WORD.findall(t.lower()))
            raw = [1.0 if w in words else 0.0 for w in vocab]
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            vectors.append([x / norm for x in raw])
        return vectors


def _claim(field: str, text: str) -> ClaimCandidate:
    return ClaimCandidate(claim_type=field, claim_text=text, evidence_quote=text, confidence="medium")


def test_matching_prediction_counts_as_true_positive() -> None:
    predictions = {"p1": [_claim("problem", "latency is too high under load")]}
    ground_truth = {"p1": {"problem": "latency is too high under load"}}

    scores = evaluate(predictions, ground_truth, ["problem"], WordOverlapEmbedder(), threshold=0.5)

    assert scores["problem"].true_positives == 1
    assert scores["problem"].false_positives == 0
    assert scores["problem"].false_negatives == 0


def test_semantically_distant_prediction_counts_as_both_fp_and_fn() -> None:
    predictions = {"p1": [_claim("problem", "gardening tips for beginners")]}
    ground_truth = {"p1": {"problem": "latency is too high under load"}}

    scores = evaluate(predictions, ground_truth, ["problem"], WordOverlapEmbedder(), threshold=0.5)

    assert scores["problem"].true_positives == 0
    assert scores["problem"].false_positives == 1
    assert scores["problem"].false_negatives == 1


def test_missing_prediction_with_ground_truth_is_false_negative_only() -> None:
    predictions = {"p1": []}
    ground_truth = {"p1": {"problem": "latency is too high under load"}}

    scores = evaluate(predictions, ground_truth, ["problem"], WordOverlapEmbedder())

    assert scores["problem"].false_negatives == 1
    assert scores["problem"].false_positives == 0


def test_prediction_with_no_ground_truth_is_false_positive_only() -> None:
    # the annotation genuinely left this field blank - extractor guessed anyway
    predictions = {"p1": [_claim("dataset", "trained on ImageNet")]}
    ground_truth = {"p1": {"dataset": ""}}

    scores = evaluate(predictions, ground_truth, ["dataset"], WordOverlapEmbedder())

    assert scores["dataset"].false_positives == 1
    assert scores["dataset"].false_negatives == 0
    assert scores["dataset"].true_positives == 0


def test_neither_prediction_nor_ground_truth_scores_nothing() -> None:
    predictions = {"p1": []}
    ground_truth = {"p1": {"applications": ""}}

    scores = evaluate(predictions, ground_truth, ["applications"], WordOverlapEmbedder())

    s = scores["applications"]
    assert (s.true_positives, s.false_positives, s.false_negatives) == (0, 0, 0)


def test_scores_aggregate_across_multiple_papers() -> None:
    predictions = {
        "p1": [_claim("problem", "high latency under load")],
        "p2": [],
    }
    ground_truth = {
        "p1": {"problem": "high latency under load"},
        "p2": {"problem": "memory leaks over time"},
    }

    scores = evaluate(predictions, ground_truth, ["problem"], WordOverlapEmbedder())

    assert scores["problem"].true_positives == 1  # p1 matched
    assert scores["problem"].false_negatives == 1  # p2 missed entirely


def test_fields_evaluated_independently() -> None:
    predictions = {"p1": [_claim("method", "we use a transformer")]}
    ground_truth = {"p1": {"method": "we use a transformer", "results": "achieves 90% accuracy"}}

    scores = evaluate(predictions, ground_truth, ["method", "results"], WordOverlapEmbedder())

    assert scores["method"].true_positives == 1
    assert scores["results"].false_negatives == 1  # no prediction at all for results


def test_field_score_precision_recall_f1() -> None:
    s = FieldScore("problem", true_positives=3, false_positives=1, false_negatives=1)

    assert s.precision == 0.75
    assert s.recall == 0.75
    assert s.f1 == 0.75


def test_field_score_with_no_predictions_or_truth_is_zero_not_divide_by_zero() -> None:
    s = FieldScore("problem")
    assert (s.precision, s.recall, s.f1) == (0.0, 0.0, 0.0)
