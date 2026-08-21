from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from researchbridge.db.models import EMBEDDING_DIM, CandidateGap, Embedding, Evidence, ExtractedClaim, Paper
from researchbridge.embedding.pipeline import EMBEDDING_TYPE
from researchbridge.gaps.batch import run_all

_WORD = re.compile(r"[a-z]+")


@dataclass
class WordOverlapEmbedder:
    """Same fake used in test_gaps_detect.py - word-overlap vectors so
    clustering behaves meaningfully rather than being luck from a hash."""

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


def _paper(session, embedder, source_id: str, title: str = "distinct unrelated title", embed: bool = True) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    if embed:
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


def _recurring_pattern(session, embedder, names: tuple[str, str, str]) -> list[Paper]:
    """Three papers sharing one limitation - a real cluster detect_candidate_gaps will find."""
    papers = [_paper(session, embedder, n) for n in names]
    texts = [
        "the system is tested only offline in this setup",
        "we test the model only offline in our setup",
        "testing here happens only offline within this setup",
    ]
    for paper, text in zip(papers, texts, strict=True):
        _claim(session, paper, "limitations", text)
    return papers


def test_skips_papers_without_an_embedding(session_factory, embedder) -> None:
    session = session_factory()
    unembedded = _paper(session, embedder, "no-embedding", embed=False)
    session.commit()

    summary = run_all(session, embedder, min_cluster_size=3, similarity_threshold=0.3, save=False)

    session.close()
    assert summary.papers_seen == 0


def test_skips_excluded_papers_as_seeds(session_factory, embedder) -> None:
    session = session_factory()
    excluded = _paper(session, embedder, "excluded")
    excluded.excluded_at = datetime.now(timezone.utc)
    session.commit()

    summary = run_all(session, embedder, min_cluster_size=3, similarity_threshold=0.3, save=False)

    session.close()
    assert summary.papers_seen == 0


def test_finds_and_reports_gaps_without_persisting_when_save_is_false(session_factory, embedder) -> None:
    session = session_factory()
    seed, a, b = _recurring_pattern(session, embedder, ("seed", "a", "b"))
    session.commit()

    summary = run_all(session, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3, save=False)

    assert summary.papers_seen == 3
    assert summary.gaps_found >= 1
    assert summary.gaps_saved == 0
    assert session.query(CandidateGap).count() == 0
    session.close()


def test_save_true_persists_found_gaps(session_factory, embedder) -> None:
    session = session_factory()
    _recurring_pattern(session, embedder, ("seed", "a", "b"))
    session.commit()

    summary = run_all(session, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3, save=True)

    assert summary.gaps_saved >= 1
    assert session.query(CandidateGap).count() == summary.gaps_saved
    session.close()


def test_skips_seed_papers_that_already_have_a_candidate_gap(session_factory, embedder) -> None:
    session = session_factory()
    seed, a, b = _recurring_pattern(session, embedder, ("seed", "a", "b"))
    session.commit()

    first = run_all(session, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3, save=True)
    assert first.papers_seen == 3

    second = run_all(session, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3, save=True)

    session.close()
    assert second.papers_seen == 0
    assert second.papers_skipped == 3


def test_force_reprocesses_seeds_that_already_have_a_candidate_gap(session_factory, embedder) -> None:
    session = session_factory()
    _recurring_pattern(session, embedder, ("seed", "a", "b"))
    session.commit()

    run_all(session, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3, save=True)
    second = run_all(session, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3, save=True, force=True)

    session.close()
    assert second.papers_seen == 3
    assert second.papers_skipped == 0


def test_continues_past_a_failure_for_one_paper(session_factory, embedder, monkeypatch) -> None:
    session = session_factory()
    seed, a, b = _recurring_pattern(session, embedder, ("seed", "a", "b"))
    session.commit()

    import researchbridge.gaps.batch as batch_module

    real_detect = batch_module.detect_candidate_gaps

    def _flaky_detect(session_arg, paper_id, *args, **kwargs):
        if paper_id == seed.id:
            raise RuntimeError("boom")
        return real_detect(session_arg, paper_id, *args, **kwargs)

    monkeypatch.setattr(batch_module, "detect_candidate_gaps", _flaky_detect)

    summary = run_all(session, embedder, top_k=10, min_cluster_size=3, similarity_threshold=0.3, save=False)

    session.close()
    assert summary.papers_failed == 1
    assert summary.papers_seen == 3  # the failing paper still counts as seen, not silently dropped
