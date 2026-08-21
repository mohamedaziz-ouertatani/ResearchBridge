from __future__ import annotations

import uuid
from dataclasses import dataclass

from researchbridge.db.models import Paper
from researchbridge.extraction.hybrid import HybridExtractor


def _paper(abstract: str | None) -> Paper:
    return Paper(
        id=uuid.uuid4(), source="fake", source_id="x", title="t", abstract=abstract,
        raw_metadata={}, ingestion_metadata={},
    )


@dataclass
class FakeEmbedder:
    """Deterministic and simple, not realistic - these tests check routing
    (does the hybrid prefer the right sub-extractor's answer), not semantic
    quality, so any valid unit vectors are enough. HybridExtractor always
    runs both sub-extractors before routing (one batched embed_texts call
    is cheaper than deciding per-field whether it's needed), so every test
    here needs an embedder that actually works, not one that refuses."""

    model_name: str = "fake"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if len(t) % 2 == 0 else [0.0, 1.0] for t in texts]


def test_empty_abstract_yields_no_candidates() -> None:
    extractor = HybridExtractor(FakeEmbedder())
    assert extractor.extract(_paper(None)) == []


def test_problem_always_prefers_heuristic_even_at_low_confidence() -> None:
    # heuristic's problem answer is its first-sentence fallback (low
    # confidence by construction) - the routing rule still prefers it for
    # "problem" specifically, regardless of what semantic also found
    abstract = "Coordination across regions is expensive."
    extractor = HybridExtractor(FakeEmbedder())

    candidates = extractor.extract(_paper(abstract))

    problem = next(c for c in candidates if c.claim_type == "problem")
    assert problem.claim_text == "Coordination across regions is expensive."
    assert problem.confidence == "low"  # heuristic's own confidence is preserved, not overridden


def test_medium_confidence_heuristic_hit_wins_over_semantic() -> None:
    # "we propose" is a real cue phrase (heuristic/method) -> medium
    # confidence -> routing must prefer it over whatever semantic found
    abstract = "We propose a lock-free queue for high throughput systems."
    extractor = HybridExtractor(FakeEmbedder())

    candidates = extractor.extract(_paper(abstract))

    method = next(c for c in candidates if c.claim_type == "method")
    assert method.claim_text == "We propose a lock-free queue for high throughput systems."
    assert method.confidence == "medium"


def test_field_with_no_heuristic_match_falls_back_to_semantic() -> None:
    # no cue phrase in this abstract matches any field's phrase list, so
    # every non-problem field (if present at all) must have come from semantic
    abstract = "Something happens in this domain. Another sentence follows it."
    extractor = HybridExtractor(FakeEmbedder())

    candidates = extractor.extract(_paper(abstract))

    non_problem = [c for c in candidates if c.claim_type != "problem"]
    assert all(c.claim_text is not None for c in non_problem)  # came from somewhere, not a crash
    # every non-problem candidate's confidence is semantic's own banding
    # (medium/low from cosine similarity), never heuristic's - since
    # heuristic had no cue-phrase match to contribute for any of these
    assert all(c.confidence in ("medium", "low") for c in non_problem)


def test_grounding_holds_for_every_hybrid_candidate() -> None:
    abstract = "Coordination across regions is expensive. We propose a cheaper protocol."
    extractor = HybridExtractor(FakeEmbedder())

    candidates = extractor.extract(_paper(abstract))

    assert len(candidates) > 0
    for c in candidates:
        assert c.evidence_quote in abstract
        assert c.claim_text == c.evidence_quote


def test_no_duplicate_fields_in_output() -> None:
    abstract = "We propose a cheaper protocol for coordination. It reduces overhead significantly."
    extractor = HybridExtractor(FakeEmbedder())

    candidates = extractor.extract(_paper(abstract))

    types = [c.claim_type for c in candidates]
    assert len(types) == len(set(types))
