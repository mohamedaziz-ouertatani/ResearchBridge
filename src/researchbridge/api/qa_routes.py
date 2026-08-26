"""Extractive Q&A over the corpus (blueprint RAG slice).

POST /api/ask retrieves candidate papers by embedding similarity, then
re-ranks their already-extracted claims/evidence against the question - see
qa/answer.py's module docstring for why this is retrieval, not generation.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_embedder, get_session
from researchbridge.api.schemas import AskRequest, AskResponse, QuoteHitOut
from researchbridge.embedding.base import Embedder
from researchbridge.qa.answer import answer_question

router = APIRouter(prefix="/api")


@router.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    session: Session = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
) -> AskResponse:
    hits = answer_question(session, embedder, payload.question)
    return AskResponse(
        hits=[
            QuoteHitOut(
                paper_id=hit.paper_id,
                paper_title=hit.paper_title,
                paper_source=hit.paper_source,
                claim_type=hit.claim_type,
                text=hit.text,
                section=hit.section,
                confidence=hit.confidence,
                score=hit.score,
            )
            for hit in hits
        ]
    )
