from __future__ import annotations

import uuid

from researchbridge.db.models import Paper
from researchbridge.extraction.heuristic import HeuristicExtractor, split_sentences


def _paper(abstract: str | None) -> Paper:
    return Paper(
        id=uuid.uuid4(), source="fake", source_id="x", title="t", abstract=abstract,
        raw_metadata={}, ingestion_metadata={},
    )


def test_split_sentences_handles_basic_punctuation() -> None:
    assert split_sentences("We propose X. It works well! Does it scale?") == [
        "We propose X.", "It works well!", "Does it scale?",
    ]


def test_empty_abstract_yields_no_candidates() -> None:
    assert HeuristicExtractor().extract(_paper(None)) == []
    assert HeuristicExtractor().extract(_paper("   ")) == []


def test_method_cue_phrase_is_matched() -> None:
    abstract = "Prior work struggles with X. We propose a graph-based method for X. It works on real data."
    candidates = HeuristicExtractor().extract(_paper(abstract))

    method = next(c for c in candidates if c.claim_type == "method")
    assert method.claim_text == "We propose a graph-based method for X."
    assert method.confidence == "medium"


def test_every_candidate_is_a_verbatim_substring_of_the_abstract() -> None:
    abstract = (
        "Existing systems fail under load. We ask whether a simpler design suffices. "
        "We propose a lock-free queue. Our results show a 3x speedup. "
        "However, throughput degrades past 64 threads."
    )
    candidates = HeuristicExtractor().extract(_paper(abstract))

    assert len(candidates) > 0
    for c in candidates:
        assert c.evidence_quote in abstract
        assert c.claim_text == c.evidence_quote


def test_field_with_no_cue_match_is_omitted_not_fabricated() -> None:
    # no cue phrase for any field except an implicit "problem" opening sentence
    abstract = "Something happens in this domain."
    candidates = HeuristicExtractor().extract(_paper(abstract))

    types = {c.claim_type for c in candidates}
    assert "dataset" not in types
    assert "applications" not in types


def test_problem_falls_back_to_first_sentence_at_low_confidence() -> None:
    abstract = "Coordination across regions is expensive. We propose a cheaper protocol."
    candidates = HeuristicExtractor().extract(_paper(abstract))

    problem = next(c for c in candidates if c.claim_type == "problem")
    assert problem.claim_text == "Coordination across regions is expensive."
    assert problem.confidence == "low"


def test_problem_not_added_when_a_real_cue_phrase_already_covers_it() -> None:
    # first sentence itself matches a stronger cue phrase (method) -> no
    # separate low-confidence "problem" duplicate of the same sentence
    abstract = "We propose a new caching layer. It reduces latency significantly."
    candidates = HeuristicExtractor().extract(_paper(abstract))

    problems = [c for c in candidates if c.claim_type == "problem"]
    assert problems == []


def test_more_specific_phrase_takes_priority_over_vaguer_one() -> None:
    # "our contribution" (main_contribution) should win over a later, vaguer
    # "we show that" sentence also present in the same abstract
    abstract = "Our contribution is a new proof technique. Separately, we show that it generalizes."
    candidates = HeuristicExtractor().extract(_paper(abstract))

    contribution = next(c for c in candidates if c.claim_type == "main_contribution")
    assert contribution.claim_text == "Our contribution is a new proof technique."
