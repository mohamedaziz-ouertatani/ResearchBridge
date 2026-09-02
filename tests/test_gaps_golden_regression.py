"""Golden regression set for the gap-evidence-grounding design (see
docs/superpowers/plans/2026-09-01-gap-evidence-grounding.md). Each test
pins down one real or representative false-positive/false-negative pattern
found during manual review of production candidate gaps - a regression in
any of these means the precision-over-recall guarantee has broken.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import dataclass, field

from researchbridge.db.models import EMBEDDING_DIM, Embedding, Evidence, ExtractedClaim, Paper
from researchbridge.embedding.pipeline import EMBEDDING_TYPE
from researchbridge.gaps.detect import detect_candidate_gaps

_WORD = re.compile(r"[a-z]+")


def _word_index(word: str) -> int:
    """Stable hash-bucket for a word, independent of process hash-seed and
    of what other texts happen to be in the same embed_texts() call."""
    return int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % EMBEDDING_DIM


@dataclass
class WordOverlapEmbedder:
    """Bag-of-words cosine-similarity fake - no semantic understanding, only
    literal shared-word overlap (see test_gaps_cluster.py/test_gaps_detect.py
    for the same convention). Unlike those files' copy, this one hashes each
    word into a FIXED-size vector (EMBEDDING_DIM buckets) instead of building
    a fresh vocabulary per call: detect_candidate_gaps makes two independent
    embed_texts() calls that later get cosine-compared against each other
    (contribution-claim vectors vs. cluster representative-text vectors in
    find_addressing_papers) - a per-call vocabulary would size those two
    calls' vectors differently and cosine_similarity's zip(strict=True)
    would raise. A real embedder's dimensionality never depends on what else
    is in the batch, so this fixes the fake to match that property without
    giving it any actual semantic understanding."""

    model_name: str = "word-overlap-fake"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        vectors = []
        for t in texts:
            words = set(_WORD.findall(t.lower()))
            raw = [0.0] * EMBEDDING_DIM
            for w in words:
                raw[_word_index(w)] = 1.0
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            vectors.append([x / norm for x in raw])
        return vectors


def _paper(session, embedder, source_id: str, title: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
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


def _claim(session, paper: Paper, claim_type: str, text: str, validation_tier: str | None = None) -> None:
    evidence = Evidence(
        paper_id=paper.id, evidence_type=claim_type, section=None, text=text,
        extraction_method="fake", model_version="fake-v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    session.add(
        ExtractedClaim(
            paper_id=paper.id, claim_type=claim_type, text=text, evidence_id=evidence.id,
            confidence="medium", validation_tier=validation_tier,
        )
    )


def test_1_paper_motivation_bridging_a_gap_is_excluded(session_factory) -> None:
    """CentaurBench-style: 'this paper investigates whether X can help
    bridge this gap' paired with a near-identical own contribution - both
    self_resolution and own_contribution_overlap fire, so the claim is
    "motivation" and can't seed a cluster."""
    embedder = WordOverlapEmbedder()
    session = session_factory()
    seed = _paper(session, embedder, "seed", "centaur seed")
    a = _paper(session, embedder, "a", "centaur a")
    b = _paper(session, embedder, "b", "centaur b")
    for paper in (seed, a, b):
        _claim(
            session, paper, "research_gap",
            "this paper investigates whether large language models can help bridge this gap in benchmark coverage",
            validation_tier="strong",
        )
        _claim(
            session, paper, "main_contribution",
            "we investigate whether large language models can help bridge this gap in benchmark coverage",
        )
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "insufficient_evidence"
    assert result.drafts == []


def test_2_recurring_benchmark_motivation_language_is_excluded(session_factory) -> None:
    """FinRCA-Bench-style: three benchmark papers each honestly motivate
    their own contribution with near-identical 'existing benchmarks lack X'
    phrasing, where X is exactly what their own contribution measures. No
    self-resolution phrase in the limitation sentence itself, but the
    overlap with each paper's OWN contribution is what excludes it."""
    embedder = WordOverlapEmbedder()
    session = session_factory()
    seed = _paper(session, embedder, "seed", "finrca seed")
    a = _paper(session, embedder, "a", "finrca a")
    b = _paper(session, embedder, "b", "finrca b")
    for paper in (seed, a, b):
        _claim(
            session, paper, "research_gap",
            "no existing benchmark evaluates robustness to table perturbations and reasoning pathway aggregation",
            validation_tier="strong",
        )
        _claim(
            session, paper, "main_contribution",
            "we introduce a benchmark evaluating robustness to table perturbations and reasoning pathway aggregation",
        )
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "insufficient_evidence"
    assert result.drafts == []


def test_3_contribution_statement_mistaken_for_a_gap_is_excluded(session_factory) -> None:
    """A results-flavored sentence should never even reach ExtractedClaim
    typed as research_gap - extraction/validation.py's existing gate
    (unchanged by this plan) is what stops it. Regression-proofs that this
    plan didn't weaken it."""
    from researchbridge.extraction.pipeline import ExtractionPipeline
    from researchbridge.extraction.base import ClaimCandidate

    embedder = WordOverlapEmbedder()
    session = session_factory()
    text = "Our model achieves 94.2% AUC and an F1 score of 0.89 on the benchmark."
    paper = Paper(
        source="fake", source_id="contrib-1", title="t", abstract=text, raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.commit()

    class _Extractor:
        extraction_method = "fake"
        model_version = "fake-v1"

        def extract(self, p):
            return [ClaimCandidate(claim_type="research_gap", claim_text=text, evidence_quote=text, confidence="medium")]

    ExtractionPipeline(_Extractor(), session_factory).run()

    claims = session.execute(
        __import__("sqlalchemy").select(ExtractedClaim).where(ExtractedClaim.paper_id == paper.id)
    ).scalars().all()
    session.close()
    assert claims == []  # rejected at extraction validation, never reaches the gap pipeline at all


def test_4_shared_topic_with_no_unresolvedness_evidence_is_excluded(session_factory) -> None:
    """Three papers sharing a research topic (typed as 'problem', which is
    outside RELEVANT_CLAIM_TYPES) never enter gap clustering in the first
    place - there are related papers, but zero limitation/research_gap
    claims among them at all."""
    embedder = WordOverlapEmbedder()
    session = session_factory()
    seed = _paper(session, embedder, "seed", "topic seed")
    a = _paper(session, embedder, "a", "topic a")
    b = _paper(session, embedder, "b", "topic b")
    for paper in (seed, a, b):
        _claim(session, paper, "problem", "long-context reasoning in large language models is challenging")
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "insufficient_evidence"
    assert result.drafts == []


def test_5_explicit_field_level_unresolved_gap_is_a_strong_gap(session_factory) -> None:
    """The positive case: two independent papers state unambiguous,
    field-scoped unresolved-gap language, with no self-resolution and no
    overlap with their own (unrelated) contributions."""
    embedder = WordOverlapEmbedder()
    session = session_factory()
    seed = _paper(session, embedder, "seed", "strong seed")
    a = _paper(session, embedder, "a", "strong a")
    b = _paper(session, embedder, "b", "strong b")
    for paper in (seed, a, b):
        _claim(
            session, paper, "research_gap",
            "despite recent progress robustness to adversarial perturbations remains unresolved for existing methods",
            validation_tier="strong",
        )
        _claim(session, paper, "main_contribution", "we propose a transformer based text classification architecture")
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "gaps_found"
    assert len(result.drafts) == 1
    assert result.drafts[0].gap_status == "strong_gap"


def test_6_genuine_recurring_limitation_without_gap_vocabulary_is_known_limitation(session_factory) -> None:
    """Three papers each admit the same residual weakness in plain
    limitation language (not the stronger research_gap vocabulary), with no
    overlap against their own unrelated contributions - a real, if lower-
    confidence, candidate."""
    embedder = WordOverlapEmbedder()
    session = session_factory()
    seed = _paper(session, embedder, "seed", "limitation seed")
    a = _paper(session, embedder, "a", "limitation a")
    b = _paper(session, embedder, "b", "limitation b")
    for paper in (seed, a, b):
        _claim(session, paper, "limitations", "however the approach does not scale to graphs with a million nodes")
        _claim(session, paper, "main_contribution", "we propose a lock free queue for high throughput systems")
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "gaps_found"
    assert len(result.drafts) == 1
    assert result.drafts[0].gap_status == "known_limitation"


def test_7_benchmark_paper_similar_to_but_not_solving_the_gap_still_corroborates(session_factory) -> None:
    """Two independent papers anchor a strong_gap; a third paper only
    benchmarks/evaluates the same problem (its own contribution overlaps
    heavily, but it never uses self-resolution language) - it must be
    demoted to supporting, not dropped, and must not block the strong_gap
    the other two already earned."""
    embedder = WordOverlapEmbedder()
    session = session_factory()
    seed = _paper(session, embedder, "seed", "mixed seed")
    a = _paper(session, embedder, "a", "mixed a")
    benchmark_paper = _paper(session, embedder, "bench", "mixed bench")

    for paper in (seed, a):
        _claim(
            session, paper, "research_gap",
            "despite recent progress evaluation under adversarial perturbations remains unresolved for existing benchmarks",
            validation_tier="strong",
        )
        _claim(session, paper, "main_contribution", "we propose a transformer based text classification architecture")

    _claim(
        session, benchmark_paper, "research_gap",
        "no existing benchmark evaluates adversarial perturbations under realistic constraints for existing benchmarks",
        validation_tier="strong",
    )
    _claim(
        session, benchmark_paper, "main_contribution",
        "we introduce a benchmark for evaluating adversarial perturbations under realistic constraints",
    )
    session.commit()

    result = detect_candidate_gaps(session, seed.id, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3)

    session.close()
    assert result.status == "gaps_found"
    assert len(result.drafts) == 1
    draft = result.drafts[0]
    assert draft.gap_status == "strong_gap"
    assert draft.contributing_paper_count == 3  # the benchmark paper's evidence still shows up...
    benchmark_classification = next(
        c for eid, c in draft.evidence_roles.items() if c.own_contribution_overlap >= 0.5
    )
    assert benchmark_classification.role == "supporting"  # ...but demoted, not counted as an anchor
