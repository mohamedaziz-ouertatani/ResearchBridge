"""Deterministic claim-level lexical relevance for feasibility's widened band
(0.35 < distance <= WIDENED_DISTANCE) - see
docs/superpowers/specs and feasibility.py's docstring for the calibration
history behind these two constants and this scoring approach.

No LLM, no embeddings here - IDF-weighted, stopword-filtered token overlap
between the research input's own text and a candidate claim's text. Reuses
the same style of corpus-wide IDF weighting retrieval/bm25.py already
computes for BM25, so generic/common corpus terms ("ai", "privacy",
"healthcare", "learning") automatically score low without a hand-maintained
domain-word list, while rare/specific terms ("zkfl", "byzantine",
"homomorphic") drive the score.

Calibrated against the real corpus and all real ResearchInputs (see
feasibility.py's docstring): WIDENED_DISTANCE=0.40 combined with
CLAIM_OVERLAP_THRESHOLD=0.20 recovered 8 genuinely on-technique papers
across 5 inputs with zero confirmed false positives, while a known
deep-paraphrase case (claims sharing almost no vocabulary with the query
despite being semantically close) remains - and is expected to remain -
unrecovered. This is a lexical mechanism; it cannot bridge a paraphrase gap,
only a vocabulary-sparse-but-shared-terms gap.
"""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.db.models import Paper
from researchbridge.retrieval.text import document_text

# Feasibility-specific: NOT reused by novelty/gap/applications, which answer
# a different question ("is this paper relevant at all") with their own
# shared NEAR_DISTANCE/FAR_DISTANCE. These two only ever gate feasibility's
# widened corroboration band.
WIDENED_DISTANCE = 0.40
CLAIM_OVERLAP_THRESHOLD = 0.20

_TOKEN = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "for", "on", "with", "without",
    "that", "this", "would", "using", "across", "while", "system", "systems", "based",
    "learning", "model", "models", "data", "each", "their", "its", "into", "from", "by",
    "we", "is", "are", "be", "as", "at", "it", "can", "which",
}


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS}


def claim_overlap(query_text: str, claim_text: str, idf: dict[str, float]) -> float:
    """IDF-weighted overlap between query_text and claim_text, normalized by
    the query's own total IDF weight (so a short/generic query can't be
    trivially satisfied, but a claim need not repeat every query term).
    idf.get(token, 0.0) for unknown tokens - never assume relevance for a
    term the corpus has no frequency statistics for. Negative idf values
    (rank_bm25 can produce these for extremely common terms) are clamped to
    0.0, never subtracted - a common shared term contributes nothing to the
    score, it never penalizes it.

    Pure function, no DB/network access - independently testable, and this
    is the only place either evidence-admission decision is made from
    memorized token overlap rather than corpus lookups.
    """
    query_tokens = _tokenize(query_text)
    claim_tokens = _tokenize(claim_text)
    shared = query_tokens & claim_tokens
    if not shared:
        return 0.0

    numerator = sum(max(idf.get(t, 0.0), 0.0) for t in shared)
    denominator = sum(max(idf.get(t, 0.0), 0.0) for t in query_tokens)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def build_corpus_idf(session: Session) -> dict[str, float]:
    """Corpus-wide IDF weights, computed fresh from whatever this session
    currently sees. Reuses rank_bm25.BM25Okapi purely for its .idf attribute
    - no ranking machinery, no search, no new dependency (rank_bm25 is
    already used by retrieval/bm25.py).

    Deliberately NOT cached here - this module stays a pure/near-pure
    computation layer. Process-wide caching (this is expensive: ~7-10s over
    the full corpus, so it must not be redone per request) lives at the
    dependency-injection boundary instead - see api/deps.py's
    get_corpus_idf(), which wraps this the same way get_embedder() wraps
    SentenceTransformerEmbedder(): built once per process via @lru_cache,
    reused across requests, going stale as new papers are ingested until
    the next process restart. No invalidation/refresh mechanism is added
    this pass."""
    from rank_bm25 import BM25Okapi

    docs = []
    for paper in session.execute(select(Paper).where(Paper.excluded_at.is_(None))).scalars():
        text = document_text(paper)
        if text is not None:
            docs.append(list(_TOKEN.findall(text.lower())))

    return BM25Okapi(docs).idf if docs else {}
