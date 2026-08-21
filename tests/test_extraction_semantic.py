from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field

from researchbridge.db.models import Paper
from researchbridge.extraction import semantic as semantic_module
from researchbridge.extraction.semantic import SemanticExtractor

_WORD = re.compile(r"[a-z]+")


@dataclass
class WordOverlapEmbedder:
    """Bag-of-words cosine similarity - deterministic and legible, unlike a
    hash-based fake, so tests can reason directly about which sentence
    should win for a given field query instead of trusting opaque luck."""

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


def _paper(abstract: str | None) -> Paper:
    return Paper(
        id=uuid.uuid4(), source="fake", source_id="x", title="t", abstract=abstract,
        raw_metadata={}, ingestion_metadata={},
    )


def _use_controlled_field_queries(monkeypatch, queries: dict[str, str]) -> None:
    """Swaps the production field-query prose for short controlled text, so
    tests reason about overlap counts they chose, not incidental word
    overlap in whatever the real prompts happen to say."""
    monkeypatch.setattr(semantic_module, "_FIELD_QUERIES", queries)


def test_empty_abstract_yields_no_candidates() -> None:
    extractor = SemanticExtractor(WordOverlapEmbedder())
    assert extractor.extract(_paper(None)) == []
    assert extractor.extract(_paper("   ")) == []


def test_picks_the_more_similar_sentence(monkeypatch) -> None:
    _use_controlled_field_queries(monkeypatch, {"method": "graph neural network approach"})
    abstract = "We use a graph neural network approach for this task. Weather today is unrelated and fine."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract))

    method = next(c for c in candidates if c.claim_type == "method")
    assert method.claim_text == "We use a graph neural network approach for this task."


def test_every_candidate_is_a_verbatim_substring_of_the_abstract(monkeypatch) -> None:
    _use_controlled_field_queries(
        monkeypatch,
        {"problem": "latency load system", "method": "graph neural network approach"},
    )
    abstract = "Latency under load breaks the system. We use a graph neural network approach here."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract))

    assert len(candidates) > 0
    for c in candidates:
        assert c.evidence_quote in abstract
        assert c.claim_text == c.evidence_quote


def test_zero_overlap_field_gets_no_candidate(monkeypatch) -> None:
    # the field query shares no vocabulary at all with any abstract sentence
    # -> similarity 0.0, well under MIN_SIMILARITY -> omitted, not guessed
    _use_controlled_field_queries(monkeypatch, {"applications": "zzqx wwky unrelated_token_xyz"})
    abstract = "We use a graph neural network approach for this task."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract))

    assert candidates == []


def test_strong_overlap_gets_medium_confidence(monkeypatch) -> None:
    # 2 of 2 field-query words present, diluted by the sentence's other 2
    # words -> cosine = 2/sqrt(2*4) ~= 0.707, above MEDIUM_CONFIDENCE_SIMILARITY (0.28)
    _use_controlled_field_queries(monkeypatch, {"method": "graph neural"})
    abstract = "Graph neural nets work."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract))

    expected_similarity = 2 / math.sqrt(2 * 4)
    assert expected_similarity >= semantic_module.MEDIUM_CONFIDENCE_SIMILARITY
    assert candidates[0].confidence == "medium"


def test_weak_but_above_threshold_overlap_gets_low_confidence(monkeypatch) -> None:
    assert semantic_module.MIN_SIMILARITY < semantic_module.MEDIUM_CONFIDENCE_SIMILARITY

    # 1 shared word ("graph") out of 3 field-query words and 6 sentence
    # words -> cosine = 1/sqrt(3*6) ~= 0.236, between MIN_SIMILARITY (0.15)
    # and MEDIUM_CONFIDENCE_SIMILARITY (0.28) - verified directly, not just
    # asserted, since bag-of-words cosine dilutes fast and isn't obvious
    # from word counts alone.
    _use_controlled_field_queries(monkeypatch, {"method": "graph neural network"})
    abstract = "Graph theory helps model networks well."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract))

    expected_similarity = 1 / math.sqrt(3 * 6)
    assert semantic_module.MIN_SIMILARITY <= expected_similarity < semantic_module.MEDIUM_CONFIDENCE_SIMILARITY
    assert len(candidates) == 1
    assert candidates[0].confidence == "low"


def test_fields_are_evaluated_independently(monkeypatch) -> None:
    _use_controlled_field_queries(
        monkeypatch,
        {"problem": "latency load", "dataset": "zzqx wwky unrelated_token_xyz"},
    )
    abstract = "Latency under load breaks the system. We use a graph approach here."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract))

    types = {c.claim_type for c in candidates}
    assert "problem" in types
    assert "dataset" not in types
