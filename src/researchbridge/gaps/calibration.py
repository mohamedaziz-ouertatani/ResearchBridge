"""Calibrates OWN_CONTRIBUTION_OVERLAP_THRESHOLD and ADDRESSING_SIMILARITY_THRESHOLD
(gaps/signals.py) against real, hand-labeled data before either constant is
treated as final - see docs/superpowers/plans/2026-09-01-gap-evidence-
grounding.md Tasks 13-14.

Mirrors extraction/type_validation_evaluation.py's own-field/cross-field
pattern: measure real ground-truth pairs that SHOULD overlap against real
pairs that should NOT, rather than eyeballing a single number.

Two data sources, covering all six required calibration categories:

1. The Sec 25 hand-annotated benchmark (benchmark.store.load_all) already
   contains real ground truth for two categories, no new annotation
   needed: research_gap.remaining vs main_contribution is a real "should
   NOT overlap" pair (genuine_unresolved_gap); research_gap.addressed vs
   main_contribution is a real "SHOULD overlap" pair (motivation_addressed).

2. benchmark/gap_calibration_groups.yaml, a small hand-curated companion
   file, for the categories that need a curated group of thematically-
   connected real papers to find examples of.

The two thresholds this calibrates measure different things and are never
scored together: OWN_CONTRIBUTION_OVERLAP_THRESHOLD asks "does this claim
overlap with what its OWN paper contributed" (own_contribution_overlap_
samples, own_contribution_overlap_group_samples) - a comparatively clear
question. ADDRESSING_SIMILARITY_THRESHOLD asks "is this claim similar
enough to a DIFFERENT paper's contribution to be worth a reviewer warning"
(addressing_signal_samples) - a deliberately weaker question, since high
similarity to another paper's benchmark/evaluation/partial-solution
contribution is never proof that paper resolved the gap. Mixing the two
into one pool would blur two different failure costs: a false positive on
the own-contribution side wrongly excludes real evidence, while a false
positive on the addressing side just erodes trust in an already-advisory
warning - see Task 14's separate decision rules for each.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from researchbridge.benchmark.store import Annotation
from researchbridge.embedding.base import Embedder
from researchbridge.gaps.signals import cosine_similarity


@dataclass(frozen=True)
class ThresholdSample:
    category: str
    paper_a: str
    paper_b: str
    text_a: str
    text_b: str
    similarity: float
    expected_high_overlap: bool
    """Ground truth: True if this pair SHOULD clear the threshold (a real
    motivation/addressed/benchmark-not-solving pair), False if it should
    NOT (a real genuine-unresolved-gap or topically-unrelated pair)."""


@dataclass(frozen=True)
class CalibrationGroup:
    category: str
    source_ids: list[str]
    note: str


def load_calibration_groups(path: str | Path) -> list[CalibrationGroup]:
    path = Path(path)
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [
        CalibrationGroup(category=g["category"], source_ids=list(g["source_ids"]), note=g.get("note", ""))
        for g in raw.get("groups", [])
    ]


def own_contribution_overlap_samples(annotations: list[Annotation], embedder: Embedder) -> list[ThresholdSample]:
    texts: list[str] = []
    meta: list[tuple[str, str, str, str, bool]] = []
    for ann in annotations:
        contribution = (ann.fields.get("main_contribution") or "").strip()
        remaining = (ann.research_gap.get("remaining") or "").strip()
        addressed = (ann.research_gap.get("addressed") or "").strip()
        if contribution and remaining:
            texts += [contribution, remaining]
            meta.append(("genuine_unresolved_gap", ann.source_id, contribution, remaining, False))
        if contribution and addressed:
            texts += [contribution, addressed]
            meta.append(("motivation_addressed", ann.source_id, contribution, addressed, True))

    if not meta:
        return []

    vectors = embedder.embed_texts(texts)
    samples = []
    for i, (category, source_id, text_a, text_b, expected) in enumerate(meta):
        va, vb = vectors[2 * i], vectors[2 * i + 1]
        samples.append(
            ThresholdSample(
                category=category, paper_a=source_id, paper_b=source_id,
                text_a=text_a, text_b=text_b, similarity=cosine_similarity(va, vb), expected_high_overlap=expected,
            )
        )
    return samples


# These two categories still need a curated GROUP to find real examples of
# in the corpus, but each computed sample is a same-paper self-pair (this
# paper's own limitation text against this SAME paper's own contribution) -
# calibrates OWN_CONTRIBUTION_OVERLAP_THRESHOLD, same as
# own_contribution_overlap_samples above, NOT the cross-paper addressing
# signal below. See classify_cluster's independent_papers filter
# (gaps/signals.py) - this is exactly what gates that.
_OWN_OVERLAP_GROUP_EXPECTATIONS = {
    "recurring_limitation_genuinely_unresolved": False,
    "recurring_limitation_topical_convergence": True,
}


def own_contribution_overlap_group_samples(
    annotations: list[Annotation], groups: list[CalibrationGroup], embedder: Embedder
) -> list[ThresholdSample]:
    annotations_by_source_id = {a.source_id: a for a in annotations}

    texts: list[str] = []
    meta: list[tuple[str, str, str, str, bool]] = []
    for group in groups:
        expected = _OWN_OVERLAP_GROUP_EXPECTATIONS.get(group.category)
        if expected is None:
            continue
        for source_id in group.source_ids:
            ann = annotations_by_source_id.get(source_id)
            if ann is None:
                continue
            contribution = (ann.fields.get("main_contribution") or "").strip()
            limitation_text = (ann.research_gap.get("remaining") or ann.fields.get("limitations") or "").strip()
            if not contribution or not limitation_text:
                continue
            texts += [limitation_text, contribution]
            meta.append((group.category, source_id, limitation_text, contribution, expected))

    if not meta:
        return []

    vectors = embedder.embed_texts(texts)
    samples = []
    for i, (category, source_id, text_a, text_b, expected) in enumerate(meta):
        va, vb = vectors[2 * i], vectors[2 * i + 1]
        samples.append(
            ThresholdSample(
                category=category, paper_a=source_id, paper_b=source_id,
                text_a=text_a, text_b=text_b, similarity=cosine_similarity(va, vb), expected_high_overlap=expected,
            )
        )
    return samples


# Cross-paper only: does Paper A's gap text warrant a reviewer warning
# against Paper B's (a DIFFERENT paper's) contribution? expected_high_overlap
# here means "should trigger a warning" - it is NEVER interpreted as "B
# solved A's gap". benchmark_evaluation_partial_solution is the category
# that makes this explicit: it expects a warning (True) precisely because a
# benchmark/evaluation/partial-solution contribution is exactly the case
# apply_addressing_downgrade exists to flag, while still leaving the
# cluster's status a downgrade, never an invalidation (see Task 4, Task 10
# golden test 7).
_ADDRESSING_GROUP_EXPECTATIONS = {
    "high_value_addressing_match": True,
    "false_warning_topical_only": False,
    "benchmark_evaluation_partial_solution": True,
}


def addressing_signal_samples(
    annotations: list[Annotation], groups: list[CalibrationGroup], embedder: Embedder
) -> list[ThresholdSample]:
    annotations_by_source_id = {a.source_id: a for a in annotations}

    texts: list[str] = []
    meta: list[tuple[str, str, str, str, str, bool]] = []
    for group in groups:
        expected = _ADDRESSING_GROUP_EXPECTATIONS.get(group.category)
        if expected is None or len(group.source_ids) < 2:
            continue
        members = [annotations_by_source_id[sid] for sid in group.source_ids if sid in annotations_by_source_id]
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                text_a = (a.research_gap.get("remaining") or a.fields.get("limitations") or "").strip()
                text_b = (b.fields.get("main_contribution") or "").strip()
                if not text_a or not text_b:
                    continue
                texts += [text_a, text_b]
                meta.append((group.category, a.source_id, b.source_id, text_a, text_b, expected))

    if not meta:
        return []

    vectors = embedder.embed_texts(texts)
    samples = []
    for i, (category, paper_a, paper_b, text_a, text_b, expected) in enumerate(meta):
        va, vb = vectors[2 * i], vectors[2 * i + 1]
        samples.append(
            ThresholdSample(
                category=category, paper_a=paper_a, paper_b=paper_b,
                text_a=text_a, text_b=text_b, similarity=cosine_similarity(va, vb), expected_high_overlap=expected,
            )
        )
    return samples


@dataclass
class ThresholdSweepRow:
    threshold: float
    category: str
    total: int = 0
    correctly_flagged: int = 0
    incorrectly_flagged: int = 0
    missed: int = 0

    @property
    def false_positive_rate(self) -> float:
        """Fraction of "should NOT overlap" samples that wrongly cleared the
        threshold - the dangerous direction: a genuine gap getting demoted
        or a topically-unrelated pair getting flagged as addressing."""
        negatives = self.total - (self.correctly_flagged + self.missed)
        return self.incorrectly_flagged / negatives if negatives else 0.0

    @property
    def false_negative_rate(self) -> float:
        """Fraction of "SHOULD overlap" samples that were missed - motivation/
        addressed language or a real benchmark-overlap that didn't clear
        the threshold and so failed to downgrade/flag anything."""
        positives = self.correctly_flagged + self.missed
        return self.missed / positives if positives else 0.0


def sweep_thresholds(samples: list[ThresholdSample], thresholds: list[float]) -> list[ThresholdSweepRow]:
    categories = sorted({s.category for s in samples})
    rows = []
    for threshold in thresholds:
        for category in categories:
            row = ThresholdSweepRow(threshold=threshold, category=category)
            for sample in samples:
                if sample.category != category:
                    continue
                row.total += 1
                cleared = sample.similarity >= threshold
                if sample.expected_high_overlap and cleared:
                    row.correctly_flagged += 1
                elif sample.expected_high_overlap and not cleared:
                    row.missed += 1
                elif not sample.expected_high_overlap and cleared:
                    row.incorrectly_flagged += 1
            rows.append(row)
    return rows
