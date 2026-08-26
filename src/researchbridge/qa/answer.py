"""Extractive Q&A over the corpus.

Retrieves candidate papers by embedding similarity (reusing
embedding/search.py's search_by_text, unchanged), then re-ranks those
candidates' already-extracted claims/evidence against the exact question.
Every returned quote is a verbatim ExtractedClaim.text, already grounded by
its Evidence row (extraction/pipeline.py's _quote_is_grounded ran at
extraction time) - there is no generation step here, matching the "never
invent" rule the rest of the app follows (see Evidence's docstring in
db/models.py on why extraction_method="stub" rows are filtered out).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.db.models import Evidence, ExtractedClaim
from researchbridge.embedding.base import Embedder
from researchbridge.embedding.search import search_by_text


@dataclass
class QuoteHit:
    paper_id: uuid.UUID
    paper_title: str
    paper_source: str
    claim_type: str
    text: str
    section: str | None
    confidence: str
    score: float
    """Cosine similarity to the question (embeddings are L2-normalized, so
    this is a plain dot product) - higher is more relevant. For display/sort
    only, not a probability."""


def answer_question(
    session: Session,
    embedder: Embedder,
    question: str,
    top_k_papers: int = 10,
    top_k_quotes: int = 8,
) -> list[QuoteHit]:
    candidates = search_by_text(session, question, embedder, top_k=top_k_papers)
    if not candidates:
        return []
    papers_by_id = {paper.id: paper for paper, _ in candidates}

    rows = session.execute(
        select(ExtractedClaim, Evidence)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .where(
            ExtractedClaim.paper_id.in_(papers_by_id.keys()),
            Evidence.extraction_method != "stub",
        )
    ).all()
    if not rows:
        return []

    [question_vector] = embedder.embed_texts([question])
    quote_vectors = embedder.embed_texts([claim.text for claim, _ in rows])

    scored = [
        (_dot(question_vector, vector), claim, evidence)
        for vector, (claim, evidence) in zip(quote_vectors, rows, strict=True)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)

    return [
        QuoteHit(
            paper_id=claim.paper_id,
            paper_title=papers_by_id[claim.paper_id].title,
            paper_source=papers_by_id[claim.paper_id].source,
            claim_type=claim.claim_type,
            text=claim.text,
            section=evidence.section,
            confidence=claim.confidence,
            score=score,
        )
        for score, claim, evidence in scored[:top_k_quotes]
    ]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
