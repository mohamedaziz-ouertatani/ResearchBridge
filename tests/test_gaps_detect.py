from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field

import pytest

from researchbridge.db.models import EMBEDDING_DIM, Embedding, Evidence, ExtractedClaim, Paper
from researchbridge.embedding.pipeline import EMBEDDING_TYPE
from researchbridge.gaps.detect import (
    _load_contribution_claims,
    _load_gap_claims,
    _own_contribution_overlaps,
    detect_candidate_gaps,
)

_WORD = re.compile(r"[a-z]+")


@dataclass
class WordOverlapEmbedder:
    """Same fake as test_gaps_cluster.py - used here for both the paper-
    similarity vectors (so find_similar_to_paper returns every test paper)
    and the claim-text clustering (so the recurring-pattern test is
    meaningful, not just luck from a hash-based fake)."""

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


def _paper(session, embedder, source_id: str, title: str = "distinct unrelated title") -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    # a unique title per paper, padded so pairwise similarity stays low and
    # doesn't distort EMBEDDING_DIM-mismatched real vectors - only used so
    # find_similar_to_paper has something to rank, not for the real signal
    [vector] = embedder.embed_texts([f"{title} {source_id}"])
    padded = (vector + [0.0] * EMBEDDING_DIM)[:EMBEDDING_DIM]
    norm = math.sqrt(sum(x * x for x in padded)) or 1.0
    session.add(
        Embedding(
            paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name,
            vector=[x / norm for x in padded],
        )
    )
    return paper


def _claim(session, paper: Paper, claim_type: str, text: str) -> None:
    evidence = Evidence(
        paper_id=paper.id, evidence_type=claim_type, section=None, text=text,
        extraction_method="fake", model_version="fake-v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    session.add(ExtractedClaim(paper_id=paper.id, claim_type=claim_type, text=text, evidence_id=evidence.id, confidence="medium"))


@pytest.fixture()
def embedder() -> WordOverlapEmbedder:
    return WordOverlapEmbedder()


def test_raises_for_unknown_seed_paper(session_factory, embedder) -> None:
    session = session_factory()
    with pytest.raises(ValueError):
        detect_candidate_gaps(session, uuid.uuid4(), embedder)
    session.close()


def test_finds_a_pattern_recurring_across_the_neighborhood(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, embedder, "seed")
    a = _paper(session, embedder, "a")
    b = _paper(session, embedder, "b")
    c = _paper(session, embedder, "c")  # unrelated limitation - should not join the cluster

    _claim(session, seed, "limitations", "the system is tested only offline in this setup")
    _claim(session, a, "limitations", "we test the model only offline in our setup")
    _claim(session, b, "research_gap", "testing here happens only offline within this setup")
    _claim(session, c, "limitations", "training requires substantial gpu resources")
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "gaps_found"
    assert len(result.drafts) == 1
    assert result.drafts[0].contributing_paper_count == 3
    assert result.drafts[0].seed_paper_id == seed.id
    assert len(result.drafts[0].evidence_ids) == 3
    assert result.drafts[0].gap_status == "known_limitation"  # weak default tier, no strong anchor


def test_no_pattern_returns_empty_list(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, embedder, "seed")
    a = _paper(session, embedder, "a")

    _claim(session, seed, "limitations", "the system is tested only offline in this setup")
    _claim(session, a, "limitations", "training requires substantial gpu resources")
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "insufficient_evidence"
    assert result.drafts == []


def test_no_related_papers_at_all_is_a_distinct_status(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, embedder, "seed")
    session.commit()  # no other papers exist at all - nothing for find_similar_to_paper to return

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "no_relevant_papers"
    assert result.drafts == []


def test_neighborhood_size_counts_the_seed_and_its_related_papers(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, embedder, "seed")
    a = _paper(session, embedder, "a")
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.neighborhood_size == 2


def test_observation_names_the_inference_and_is_grounded_in_a_real_claim(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, embedder, "seed")
    a = _paper(session, embedder, "a")
    b = _paper(session, embedder, "b")

    texts = [
        "the system is tested only offline in this setup",
        "we test the model only offline in our setup",
        "testing here happens only offline within this setup",
    ]
    for paper, text in zip([seed, a, b], texts, strict=True):
        _claim(session, paper, "limitations", text)
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert "inference" in result.drafts[0].observation.lower()
    assert any(t in result.drafts[0].observation for t in texts)  # quotes a real claim, doesn't invent one


def test_load_gap_claims_reads_claim_type_and_tier(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1")
    _claim(session, paper, "research_gap", "remains an open problem for future work")
    session.commit()

    rows = _load_gap_claims(session, [paper.id])

    session.close()
    assert len(rows) == 1
    assert rows[0].claim_type == "research_gap"
    assert rows[0].paper_id == paper.id


def test_load_contribution_claims_only_loads_contribution_and_results(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1")
    _claim(session, paper, "main_contribution", "we build a new benchmark")
    _claim(session, paper, "results", "our method achieves 90% accuracy")
    _claim(session, paper, "method", "we use a transformer")
    session.commit()

    rows = _load_contribution_claims(session, [paper.id])

    session.close()
    assert {r.claim_type for r in rows} == {"main_contribution", "results"}


def test_own_contribution_overlap_is_high_for_near_identical_text(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1")
    _claim(session, paper, "research_gap", "no existing benchmark evaluates robustness to table perturbations")
    _claim(session, paper, "main_contribution", "we introduce a benchmark evaluating robustness to table perturbations")
    session.commit()

    gap_rows = _load_gap_claims(session, [paper.id])
    contribution_rows = _load_contribution_claims(session, [paper.id])
    overlaps = _own_contribution_overlaps(gap_rows, contribution_rows, embedder)

    session.close()
    assert overlaps[gap_rows[0].evidence_id] > 0.5


def test_own_contribution_overlap_is_zero_when_paper_has_no_contribution_claims(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, embedder, "p1")
    _claim(session, paper, "research_gap", "remains an open problem for future work")
    session.commit()

    gap_rows = _load_gap_claims(session, [paper.id])
    overlaps = _own_contribution_overlaps(gap_rows, [], embedder)

    session.close()
    assert overlaps[gap_rows[0].evidence_id] == 0.0


def test_own_contribution_overlap_only_compares_within_the_same_paper(session_factory, embedder) -> None:
    session = session_factory()
    a = _paper(session, embedder, "a")
    b = _paper(session, embedder, "b")
    _claim(session, a, "research_gap", "no existing benchmark evaluates robustness to table perturbations")
    _claim(session, b, "main_contribution", "we introduce a benchmark evaluating robustness to table perturbations")
    session.commit()

    gap_rows = _load_gap_claims(session, [a.id, b.id])
    contribution_rows = _load_contribution_claims(session, [a.id, b.id])
    overlaps = _own_contribution_overlaps(gap_rows, contribution_rows, embedder)

    session.close()
    # b's near-identical contribution must NOT count toward a's overlap -
    # cross-paper similarity is find_addressing_papers's job (Task 4/7), not this
    assert overlaps[gap_rows[0].evidence_id] == 0.0
