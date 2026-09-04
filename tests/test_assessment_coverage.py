from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field

import pytest

from researchbridge.assessment.coverage import compute_dimension_coverage
from researchbridge.assessment.dimensions import IdeaDimension

_WORD = re.compile(r"[a-z]+")


@dataclass
class WordOverlapEmbedder:
    """Same fake used by tests/test_assessment_gap.py: cosine similarity of
    two texts is exactly their fraction of shared distinct words, so test
    expectations are easy to reason about by hand."""

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


@pytest.fixture()
def embedder() -> WordOverlapEmbedder:
    return WordOverlapEmbedder()


def _claim(claim_type: str, text: str) -> tuple[str, str, uuid.UUID]:
    return (claim_type, text, uuid.uuid4())


NEAR = 0.1
MID = 0.5  # within RELEVANCE_DISTANCE (0.65) but beyond coverage.py's own NEAR_DISTANCE (0.35)
FAR = 0.9


def test_not_assessed_when_no_relevant_papers_retrieved(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    papers = [("Off-topic Paper", FAR, [_claim("method", "concept drift concept drift concept drift")])]

    result = compute_dimension_coverage(dims, papers, embedder)

    assert len(result) == 1
    assert result[0].status == "not_assessed"
    assert result[0].evidence_ids == []


def test_not_found_when_relevant_papers_exist_but_none_mention_the_dimension(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    papers = [("Relevant Paper", NEAR, [_claim("method", "a completely unrelated technique")])]

    result = compute_dimension_coverage(dims, papers, embedder)

    assert result[0].status == "not_found"
    assert result[0].evidence_ids == []


def test_weak_evidence_when_exactly_one_relevant_paper_matches(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    papers = [
        ("Paper A", NEAR, [_claim("limitations", "concept drift concept drift remains unaddressed")]),
        ("Paper B", NEAR, [_claim("method", "an unrelated method entirely")]),
    ]

    result = compute_dimension_coverage(dims, papers, embedder)

    assert result[0].status == "weak_evidence"
    assert len(result[0].supporting_paper_titles) == 1
    assert result[0].supporting_paper_titles == ["Paper A"]


def test_partially_addressed_when_multiple_papers_only_flag_it_as_a_limitation(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    papers = [
        ("Paper A", NEAR, [_claim("limitations", "concept drift concept drift is unhandled")]),
        ("Paper B", NEAR, [_claim("research_gap", "concept drift concept drift remains open")]),
    ]

    result = compute_dimension_coverage(dims, papers, embedder)

    assert result[0].status == "partially_addressed"
    assert len(result[0].supporting_paper_titles) == 2


def test_established_when_multiple_papers_affirmatively_cover_it(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    papers = [
        ("Paper A", NEAR, [_claim("method", "handles concept drift concept drift directly")]),
        ("Paper B", NEAR, [_claim("results", "concept drift concept drift adaptation improved accuracy")]),
    ]

    result = compute_dimension_coverage(dims, papers, embedder)

    assert result[0].status == "established"
    assert len(result[0].supporting_paper_titles) == 2


def test_paper_at_exact_relevance_distance_boundary_still_counts(embedder) -> None:
    # default relevance_distance (0.65) is compared with <=, so a paper
    # exactly at the boundary must still be treated as relevant
    dims = [IdeaDimension(label="concept drift")]
    papers = [("Paper A", 0.65, [_claim("method", "concept drift concept drift handling")])]

    result = compute_dimension_coverage(dims, papers, embedder)

    assert result[0].status != "not_assessed"


def test_paper_just_beyond_relevance_distance_boundary_is_excluded(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    papers = [("Paper A", 0.651, [_claim("method", "concept drift concept drift handling")])]

    result = compute_dimension_coverage(dims, papers, embedder)

    assert result[0].status == "not_assessed"


def test_evidence_ids_always_trace_back_to_input_claims(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    method_claim = _claim("method", "handles concept drift concept drift directly")
    other_claim = _claim("method", "handles concept drift concept drift directly")
    papers = [("Paper A", NEAR, [method_claim]), ("Paper B", NEAR, [other_claim])]

    result = compute_dimension_coverage(dims, papers, embedder)

    all_input_evidence_ids = {method_claim[2], other_claim[2]}
    assert set(result[0].evidence_ids) <= all_input_evidence_ids


# --- NEAR_DISTANCE gate on established/partially_addressed: 2+ distinct
# papers is only trusted as established/partially_addressed if at least
# one is genuinely CLOSE (<=0.35), not merely topically adjacent (<=0.65) -
# see coverage.py's own NEAR_DISTANCE docstring for the real false
# positive this fixes.


def test_multiple_topically_adjacent_but_not_close_papers_downgrades_to_weak_evidence(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    papers = [
        ("Paper A", MID, [_claim("method", "handles concept drift concept drift directly")]),
        ("Paper B", MID, [_claim("results", "concept drift concept drift adaptation improved accuracy")]),
    ]

    result = compute_dimension_coverage(dims, papers, embedder)

    # would be "established" under the old paper-distance-blind rule (2
    # distinct papers, one affirmative) - downgraded because neither paper
    # is actually close to the idea
    assert result[0].status == "weak_evidence"
    assert len(result[0].supporting_paper_titles) == 2


def test_established_only_needs_one_of_several_papers_to_be_close(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    papers = [
        ("Near Paper", NEAR, [_claim("method", "handles concept drift concept drift directly")]),
        ("Mid Paper", MID, [_claim("results", "concept drift concept drift adaptation improved accuracy")]),
    ]

    result = compute_dimension_coverage(dims, papers, embedder)

    assert result[0].status == "established"
    assert len(result[0].supporting_paper_titles) == 2


def test_partially_addressed_also_requires_a_close_paper(embedder) -> None:
    dims = [IdeaDimension(label="concept drift")]
    papers = [
        ("Paper A", MID, [_claim("limitations", "concept drift concept drift is unhandled")]),
        ("Paper B", MID, [_claim("research_gap", "concept drift concept drift remains open")]),
    ]

    result = compute_dimension_coverage(dims, papers, embedder)

    assert result[0].status == "weak_evidence"


def test_multiple_dimensions_are_each_scored_independently(embedder) -> None:
    dims = [IdeaDimension(label="concept drift"), IdeaDimension(label="class imbalance")]
    papers = [
        ("Paper A", NEAR, [_claim("method", "handles concept drift concept drift directly")]),
    ]

    result = compute_dimension_coverage(dims, papers, embedder)

    by_label = {r.dimension: r for r in result}
    assert by_label["concept drift"].status == "weak_evidence"
    assert by_label["class imbalance"].status == "not_found"
