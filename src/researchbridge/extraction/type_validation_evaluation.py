"""Evaluates validate_claim_type() itself against the Sec 25 benchmark (Sec 34/10).

Precision/recall against embedding similarity (evaluation.py) measures
whether an extractor found text similar to the right ground-truth field.
It cannot measure the failure mode this module targets: a claim that is
similar to the WRONG field's ground truth text and was accepted anyway
because nothing checked its semantic category. This reuses the same 40
hand-annotated papers, no new annotation effort, by treating each
annotation's own field texts as both positive and negative examples:

- own-field acceptance: for each validated field, does the field's own
  ground-truth text pass validate_claim_type(field, text)? A validator
  that rejects real, correctly-labeled text is producing false negatives
  - this is a false-rejection-rate check, not just a happy-path smoke test.

- cross-field rejection: for each *other* field's ground-truth text, does
  validate_claim_type(field, other_text) correctly reject it? This is the
  direct measurement of the reported bug: a results sentence must be
  rejected when checked as a research_gap, an method/architecture
  sentence must be rejected when checked as an application, etc.

Both are computed only from real annotation text, never synthetic
examples, and only over non-empty ground-truth fields (an empty
"annotator left this blank" field is not a labeled negative).
"""

from __future__ import annotations

from dataclasses import dataclass

from researchbridge.benchmark.store import Annotation
from researchbridge.extraction.validation import VALIDATABLE_CLAIM_TYPES, validate_claim_type


@dataclass
class TypeValidationScore:
    field: str
    own_field_total: int = 0
    own_field_accepted: int = 0
    cross_field_total: int = 0
    cross_field_rejected: int = 0

    @property
    def own_field_acceptance_rate(self) -> float:
        return self.own_field_accepted / self.own_field_total if self.own_field_total else 0.0

    @property
    def cross_field_rejection_rate(self) -> float:
        return self.cross_field_rejected / self.cross_field_total if self.cross_field_total else 0.0


def _field_text(annotation: Annotation, field: str) -> str:
    if field == "research_gap":
        return (annotation.research_gap.get("remaining") or "").strip()
    return (annotation.fields.get(field) or "").strip()


def evaluate_claim_type_validation(
    annotations: list[Annotation],
    fields: frozenset[str] = VALIDATABLE_CLAIM_TYPES,
) -> dict[str, TypeValidationScore]:
    scores = {field: TypeValidationScore(field) for field in fields}

    for annotation in annotations:
        texts_by_field = {field: _field_text(annotation, field) for field in fields}

        for field in fields:
            own_text = texts_by_field[field]
            if own_text:
                scores[field].own_field_total += 1
                if validate_claim_type(field, own_text).is_valid:
                    scores[field].own_field_accepted += 1

            for other_field, other_text in texts_by_field.items():
                if other_field == field or not other_text:
                    continue
                scores[field].cross_field_total += 1
                if not validate_claim_type(field, other_text).is_valid:
                    scores[field].cross_field_rejected += 1

    return scores
