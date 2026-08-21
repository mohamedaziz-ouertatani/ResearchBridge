"""Sec 28 extraction evaluation: Precision/Recall/F1, evaluated per field.

Ground truth is free text written independently during annotation, and an
extracted claim is a sentence pulled from the abstract - exact string match
would essentially never fire, so "correct" here means semantically close
enough, measured by embedding cosine similarity against a threshold. This
reuses the same embedder as retrieval rather than adding a second
similarity mechanism, and is a deliberately loose comparison (Sec 28 asks
to evaluate the fields, not to also solve free-text answer matching) - the
threshold is a judgment call, documented below, not a validated constant.

Sec 28's field list is used as written where it maps onto the benchmark:
"contribution" there is main_contribution in the Sec 25 schema, and
research_gap is left out of this first pass - it's a nested {addressed,
remaining} pair in the benchmark, structurally different from the other
flat text fields, and a fair comparison for it needs its own design instead
of being forced into this one. This is a known scoping gap, not silence -
see cli_evaluate.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from researchbridge.embedding.base import Embedder
from researchbridge.extraction.base import ClaimCandidate

DEFAULT_SIMILARITY_THRESHOLD = 0.5


@dataclass
class FieldScore:
    field: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate(
    predictions: dict[str, list[ClaimCandidate]],
    ground_truth: dict[str, dict[str, str]],
    fields: list[str],
    embedder: Embedder,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict[str, FieldScore]:
    """predictions: {paper_key: [candidates]}. ground_truth: {paper_key: {field: text}}."""
    scores = {field: FieldScore(field) for field in fields}

    for paper_key, truth in ground_truth.items():
        candidates = predictions.get(paper_key, [])
        predicted_by_field = {c.claim_type: c.claim_text for c in candidates}

        for field in fields:
            predicted_text = predicted_by_field.get(field)
            truth_text = (truth.get(field) or "").strip()

            if predicted_text is None:
                if truth_text:
                    scores[field].false_negatives += 1
                continue

            if not truth_text:
                scores[field].false_positives += 1
                continue

            if _is_match(predicted_text, truth_text, embedder, threshold):
                scores[field].true_positives += 1
            else:
                scores[field].false_positives += 1
                scores[field].false_negatives += 1

    return scores


def _is_match(predicted: str, truth: str, embedder: Embedder, threshold: float) -> bool:
    predicted_vector, truth_vector = embedder.embed_texts([predicted, truth])
    similarity = sum(p * t for p, t in zip(predicted_vector, truth_vector, strict=True))
    return similarity >= threshold
