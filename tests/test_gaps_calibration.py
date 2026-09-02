from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from researchbridge.benchmark.store import Annotation
from researchbridge.gaps.calibration import (
    CalibrationGroup,
    ThresholdSample,
    addressing_signal_samples,
    load_calibration_groups,
    own_contribution_overlap_group_samples,
    own_contribution_overlap_samples,
    sweep_thresholds,
)

_WORD = re.compile(r"[a-z]+")


@dataclass
class WordOverlapEmbedder:
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


def _annotation(source_id: str, contribution: str, remaining: str = "", addressed: str = "") -> Annotation:
    return Annotation(
        source_id=source_id,
        path=None,  # not read by these functions
        identity={"domain": "Machine Learning"},
        fields={"main_contribution": contribution},
        research_gap={"remaining": remaining, "addressed": addressed},
    )


def test_own_contribution_overlap_samples_pairs_remaining_against_contribution() -> None:
    annotations = [
        _annotation("p1", contribution="we build a robustness benchmark", remaining="fairness across languages is untested")
    ]

    samples = own_contribution_overlap_samples(annotations, WordOverlapEmbedder())

    remaining_samples = [s for s in samples if s.category == "genuine_unresolved_gap"]
    assert len(remaining_samples) == 1
    assert remaining_samples[0].expected_high_overlap is False
    assert remaining_samples[0].paper_a == "p1"


def test_own_contribution_overlap_samples_pairs_addressed_against_contribution() -> None:
    annotations = [
        _annotation("p1", contribution="we build a robustness benchmark", addressed="prior work lacked a robustness benchmark")
    ]

    samples = own_contribution_overlap_samples(annotations, WordOverlapEmbedder())

    addressed_samples = [s for s in samples if s.category == "motivation_addressed"]
    assert len(addressed_samples) == 1
    assert addressed_samples[0].expected_high_overlap is True


def test_own_contribution_overlap_samples_skips_blank_fields() -> None:
    annotations = [_annotation("p1", contribution="we build a benchmark")]  # no remaining/addressed text

    samples = own_contribution_overlap_samples(annotations, WordOverlapEmbedder())

    assert samples == []


def test_sweep_thresholds_scores_each_category_at_each_threshold() -> None:
    samples = [
        ThresholdSample("genuine_unresolved_gap", "p1", "p1", "a", "b", similarity=0.2, expected_high_overlap=False),
        ThresholdSample("genuine_unresolved_gap", "p2", "p2", "a", "b", similarity=0.6, expected_high_overlap=False),
        ThresholdSample("motivation_addressed", "p1", "p1", "a", "b", similarity=0.7, expected_high_overlap=True),
    ]

    rows = sweep_thresholds(samples, thresholds=[0.5])

    by_category = {r.category: r for r in rows}
    # genuine_unresolved_gap: one sample (0.6) incorrectly clears 0.5 - a false positive (a real gap
    # would get wrongly downgraded)
    assert by_category["genuine_unresolved_gap"].false_positive_rate == 0.5
    # motivation_addressed: the one sample (0.7) correctly clears 0.5
    assert by_category["motivation_addressed"].false_negative_rate == 0.0


def test_load_calibration_groups_reads_the_yaml_schema(tmp_path) -> None:
    path = tmp_path / "groups.yaml"
    path.write_text(
        "groups:\n"
        "  - category: recurring_limitation_topical_convergence\n"
        "    source_ids: ['1111.11111', '2222.22222']\n"
        "    note: shared benchmark motivation\n"
    )

    groups = load_calibration_groups(path)

    assert len(groups) == 1
    assert groups[0].category == "recurring_limitation_topical_convergence"
    assert groups[0].source_ids == ["1111.11111", "2222.22222"]


def test_load_calibration_groups_missing_file_returns_empty() -> None:
    groups = load_calibration_groups("/does/not/exist.yaml")

    assert groups == []


def test_own_contribution_overlap_group_samples_pairs_each_member_against_itself() -> None:
    # recurring_limitation_* categories still need a curated group to FIND real
    # examples, but each sample is a same-paper self-pair - this calibrates
    # OWN_CONTRIBUTION_OVERLAP_THRESHOLD, not the cross-paper addressing signal
    annotations = [
        _annotation("p1", contribution="we introduce a benchmark for table robustness", remaining="no existing benchmark evaluates table robustness"),
        _annotation("p2", contribution="we introduce a benchmark for reasoning aggregation", remaining="no existing benchmark evaluates reasoning aggregation"),
    ]
    groups = [
        CalibrationGroup(category="recurring_limitation_topical_convergence", source_ids=["p1", "p2"], note="benchmark motivation")
    ]

    samples = own_contribution_overlap_group_samples(annotations, groups, WordOverlapEmbedder())

    assert len(samples) == 2  # one self-pair per member, not one pair across members
    assert {s.paper_a for s in samples} == {"p1", "p2"}
    assert all(s.paper_a == s.paper_b for s in samples)  # same-paper, not cross-paper
    assert all(s.expected_high_overlap is True for s in samples)


def test_own_contribution_overlap_group_samples_unknown_category_is_ignored() -> None:
    annotations = [_annotation("p1", contribution="c", remaining="r")]
    groups = [CalibrationGroup(category="high_value_addressing_match", source_ids=["p1"], note="")]

    samples = own_contribution_overlap_group_samples(annotations, groups, WordOverlapEmbedder())

    assert samples == []  # that category belongs to addressing_signal_samples, not this function


def test_addressing_signal_samples_needs_at_least_two_source_ids_per_group() -> None:
    annotations = [_annotation("solo", contribution="we build a benchmark")]
    groups = [CalibrationGroup(category="high_value_addressing_match", source_ids=["solo"], note="")]

    samples = addressing_signal_samples(annotations, groups, WordOverlapEmbedder())

    assert samples == []  # nothing to compare a lone paper against within its own group


def test_addressing_signal_samples_compares_across_different_papers_only() -> None:
    annotations = [
        _annotation("a", contribution="unrelated method", remaining="robustness to table perturbations is untested"),
        _annotation("b", contribution="we build a benchmark evaluating robustness to table perturbations"),
    ]
    groups = [CalibrationGroup(category="benchmark_evaluation_partial_solution", source_ids=["a", "b"], note="")]

    samples = addressing_signal_samples(annotations, groups, WordOverlapEmbedder())

    assert len(samples) == 1
    assert samples[0].paper_a == "a"
    assert samples[0].paper_b == "b"  # cross-paper: A's gap text against B's (different paper) contribution
    assert samples[0].expected_high_overlap is True  # should warn - never interpreted as "B solved it"
