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
    assert extractor.extract(_paper(None), {}) == []
    assert extractor.extract(_paper("   "), {}) == []


def test_picks_the_more_similar_sentence(monkeypatch) -> None:
    _use_controlled_field_queries(monkeypatch, {"method": "graph neural network approach"})
    abstract = "We use a graph neural network approach for this task. Weather today is unrelated and fine."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), {})

    method = next(c for c in candidates if c.claim_type == "method")
    assert method.claim_text == "We use a graph neural network approach for this task."
    assert method.section == "abstract"


def test_every_candidate_is_a_verbatim_substring_of_the_abstract(monkeypatch) -> None:
    _use_controlled_field_queries(
        monkeypatch,
        {"problem": "latency load system", "method": "graph neural network approach"},
    )
    abstract = "Latency under load breaks the system. We use a graph neural network approach here."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), {})

    assert len(candidates) > 0
    for c in candidates:
        assert c.evidence_quote in abstract
        assert c.claim_text == c.evidence_quote


def test_zero_overlap_field_gets_no_candidate(monkeypatch) -> None:
    # the field query shares no vocabulary at all with any abstract sentence
    # -> similarity 0.0, well under MIN_SIMILARITY -> omitted, not guessed
    _use_controlled_field_queries(monkeypatch, {"applications": "zzqx wwky unrelated_token_xyz"})
    abstract = "We use a graph neural network approach for this task."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), {})

    assert candidates == []


def test_strong_overlap_gets_medium_confidence(monkeypatch) -> None:
    # 2 of 2 field-query words present, diluted by the sentence's other 2
    # words -> cosine = 2/sqrt(2*4) ~= 0.707, above MEDIUM_CONFIDENCE_SIMILARITY (0.28)
    _use_controlled_field_queries(monkeypatch, {"method": "graph neural"})
    abstract = "Graph neural nets work."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), {})

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

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), {})

    expected_similarity = 1 / math.sqrt(3 * 6)
    assert semantic_module.MIN_SIMILARITY <= expected_similarity < semantic_module.MEDIUM_CONFIDENCE_SIMILARITY
    assert len(candidates) == 1
    assert candidates[0].confidence == "low"


def test_one_sentence_cannot_win_two_fields(monkeypatch) -> None:
    # Reproduces a real reported bug: a generic sentence that scores
    # decently against several field anchors used to get duplicated across
    # all of them, crowding out a more specific sentence that was each
    # field's true best match but nobody's *own* best match. Here "generic"
    # is the top choice for both "problem" and "dataset", but scores higher
    # for "problem" - so "dataset" must fall back to its only remaining
    # suitor, "specific", rather than reuse "generic".
    _use_controlled_field_queries(
        monkeypatch,
        {
            "problem": "graph neural network approach data challenge",
            "dataset": "graph neural network data",
        },
    )
    abstract = "Generic graph neural network approach data challenge. Specific graph data only."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), {})

    by_field = {c.claim_type: c.claim_text for c in candidates}
    assert by_field["problem"].startswith("Generic")
    assert by_field["dataset"].startswith("Specific")
    assert by_field["problem"] != by_field["dataset"]


def test_fields_are_evaluated_independently(monkeypatch) -> None:
    _use_controlled_field_queries(
        monkeypatch,
        {"problem": "latency load", "dataset": "zzqx wwky unrelated_token_xyz"},
    )
    abstract = "Latency under load breaks the system. We use a graph approach here."

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), {})

    types = {c.claim_type for c in candidates}
    assert "problem" in types
    assert "dataset" not in types


def test_full_text_match_wins_over_an_abstract_match_for_the_same_field(monkeypatch) -> None:
    _use_controlled_field_queries(monkeypatch, {"method": "graph neural network approach"})
    abstract = "We use a graph neural network approach for this task."
    sections = {"methods": "We use a graph neural network approach in the methods section specifically."}

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), sections)

    method = next(c for c in candidates if c.claim_type == "method")
    assert method.section == "methods"
    assert "methods section specifically" in method.claim_text


def test_full_text_confidence_is_not_bumped_above_the_similarity_banding(monkeypatch) -> None:
    # unlike HeuristicExtractor, a semantic full-text hit keeps the same
    # medium/low confidence banding as an abstract hit would - similarity
    # scores aren't the same kind of certainty as a verbatim cue-phrase hit
    _use_controlled_field_queries(monkeypatch, {"method": "graph neural"})
    sections = {"methods": "Graph neural nets work."}

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(None), sections)

    method = next(c for c in candidates if c.claim_type == "method")
    assert method.confidence == "medium"  # same banding rule as the abstract-only test above
    assert method.section == "methods"


def test_two_fields_sharing_a_target_section_still_cannot_win_the_same_sentence(monkeypatch) -> None:
    # method and dataset both target ("methods", "experiments") in
    # FIELD_SECTIONS - this reproduces test_one_sentence_cannot_win_two_fields
    # but for two fields sharing a resolved full-text section, proving the
    # collision-avoidance grouping (not just the abstract-only pool) works
    _use_controlled_field_queries(
        monkeypatch,
        {
            "method": "graph neural network approach data challenge",
            "dataset": "graph neural network data",
        },
    )
    sections = {"methods": "Generic graph neural network approach data challenge. Specific graph data only."}

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(None), sections)

    by_field = {c.claim_type: c.claim_text for c in candidates}
    assert by_field["method"].startswith("Generic")
    assert by_field["dataset"].startswith("Specific")
    assert by_field["method"] != by_field["dataset"]


def test_field_abstains_rather_than_falling_back_to_abstract_when_its_section_has_zero_overlap(
    monkeypatch,
) -> None:
    # unlike HeuristicExtractor's cue-phrase presence/absence check,
    # SemanticExtractor commits to a field's resolved pool once
    # sentences_for_field returns sentences for it - a below-threshold
    # match there means the field abstains, it does not fall back and
    # re-search the abstract (that would reopen the exact "does this
    # field's own best-scoring sentence get to have it" ambiguity that
    # the pool grouping exists to avoid).
    _use_controlled_field_queries(monkeypatch, {"method": "graph neural network approach"})
    abstract = "We use a graph neural network approach for this task."
    sections = {"methods": "zzqx wwky unrelated_token_xyz completely different content here."}

    candidates = SemanticExtractor(WordOverlapEmbedder()).extract(_paper(abstract), sections)

    assert "method" not in {c.claim_type for c in candidates}
