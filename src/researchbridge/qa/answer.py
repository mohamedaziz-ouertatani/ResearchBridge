"""Extractive Q&A over the corpus.

Retrieves candidate papers by embedding similarity - reusing
embedding/search.py's search_papers_with_fulltext, which considers both a
paper's title/abstract embedding and its best-matching full-text paragraph
(see that function's docstring). Stage two re-ranks two kinds of candidate
quote against the exact question: already-extracted claims (verbatim
ExtractedClaim.text, grounded by extraction/pipeline.py's
_quote_is_grounded at extraction time) and raw full-text paragraphs
(PaperFullTextChunk - also verbatim, straight from PaperFullText.sections,
just never run through claim extraction). There is no generation step
here, matching the "never invent" rule the rest of the app follows (see
Evidence's docstring in db/models.py on why extraction_method="stub" rows
are filtered out).

A full-text paragraph that already contains a candidate claim's exact text
is dropped before ranking - claims are always verbatim substrings of some
paragraph (that's what "grounded" means here), so keeping both would show
the same content twice under two different labels.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.db.models import Evidence, ExtractedClaim, PaperFullTextChunk
from researchbridge.embedding.base import Embedder
from researchbridge.embedding.search import search_papers_with_fulltext


@dataclass
class QuoteHit:
    evidence_id: uuid.UUID | None
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
    source: Literal["claim", "passage"]
    """"claim": a vetted ExtractedClaim, backed by an Evidence row.
    "passage": a raw full-text paragraph that was never run through claim
    extraction - still a verbatim quote from the paper, just not
    claim-typed or confidence-rated (confidence is "n/a")."""


def answer_question(
    session: Session,
    embedder: Embedder,
    question: str,
    top_k_papers: int = 10,
    top_k_quotes: int = 8,
) -> list[QuoteHit]:
    candidates = search_papers_with_fulltext(session, question, embedder, top_k=top_k_papers)
    if not candidates:
        return []
    papers_by_id = {paper.id: paper for paper, _ in candidates}

    claim_rows = session.execute(
        select(ExtractedClaim, Evidence)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .where(
            ExtractedClaim.paper_id.in_(papers_by_id.keys()),
            Evidence.extraction_method != "stub",
        )
    ).all()

    chunk_rows = list(
        session.execute(
            select(PaperFullTextChunk).where(
                PaperFullTextChunk.paper_id.in_(papers_by_id.keys()),
                PaperFullTextChunk.model_name == embedder.model_name,
            )
        ).scalars()
    )
    chunk_rows = _drop_chunks_matching_claims(chunk_rows, claim_rows)

    if not claim_rows and not chunk_rows:
        return []

    claim_texts = [claim.text for claim, _ in claim_rows]
    vectors = embedder.embed_texts([question] + claim_texts)
    question_vector, claim_vectors = vectors[0], vectors[1:]

    scored: list[tuple[float, QuoteHit]] = []
    for vector, (claim, evidence) in zip(claim_vectors, claim_rows, strict=True):
        paper = papers_by_id[claim.paper_id]
        scored.append((
            _dot(question_vector, vector),
            QuoteHit(
                evidence_id=evidence.id,
                paper_id=claim.paper_id,
                paper_title=paper.title,
                paper_source=paper.source,
                claim_type=claim.claim_type,
                text=claim.text,
                section=evidence.section,
                confidence=claim.confidence,
                score=0.0,
                source="claim",
            ),
        ))

    for chunk in chunk_rows:
        paper = papers_by_id[chunk.paper_id]
        scored.append((
            _dot(question_vector, chunk.embedding),
            QuoteHit(
                evidence_id=None,
                paper_id=chunk.paper_id,
                paper_title=paper.title,
                paper_source=paper.source,
                claim_type="full_text_passage",
                text=chunk.text,
                section=chunk.section,
                confidence="n/a",
                score=0.0,
                source="passage",
            ),
        ))

    scored.sort(key=lambda item: item[0], reverse=True)

    results = []
    for score, hit in scored[:top_k_quotes]:
        hit.score = score
        results.append(hit)
    return results


def _drop_chunks_matching_claims(
    chunks: list[PaperFullTextChunk], claim_rows: list[tuple[ExtractedClaim, Evidence]]
) -> list[PaperFullTextChunk]:
    claim_texts_by_paper: dict[uuid.UUID, list[str]] = {}
    for claim, _ in claim_rows:
        claim_texts_by_paper.setdefault(claim.paper_id, []).append(claim.text)

    kept = []
    for chunk in chunks:
        claim_texts = claim_texts_by_paper.get(chunk.paper_id, [])
        if any(text in chunk.text for text in claim_texts):
            continue
        kept.append(chunk)
    return kept


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
