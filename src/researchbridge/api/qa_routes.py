"""Extractive Q&A over the corpus (blueprint RAG slice), plus an optional
local-LLM summarization layer.

POST /api/ask retrieves candidate papers by embedding similarity, then
re-ranks their already-extracted claims/evidence against the question -
see qa/answer.py's module docstring for why this is retrieval, not
generation. That endpoint's behavior is unchanged by the layer below.

POST /api/ask/summarize is the optional layer: it takes the exact hits
a client already received from /api/ask and asks a local Ollama model
to synthesize a short, cited summary of them - see qa/summarize.py's
module docstring and docs/superpowers/specs/
2026-08-26-ollama-summary-layer-design.md for the grounding guarantees.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_embedder, get_session
from researchbridge.api.schemas import (
    AskRequest,
    AskResponse,
    QuoteHitOut,
    SummarizeRequest,
    SummarizeResponse,
)
from researchbridge.embedding.base import Embedder
from researchbridge.qa.answer import answer_question
from researchbridge.qa.summarize import SummarizationUnavailable, ollama_enabled, summarize_quotes

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
        ],
        summarization_available=ollama_enabled(),
    )


@router.post("/ask/summarize", response_model=SummarizeResponse)
def summarize(payload: SummarizeRequest) -> SummarizeResponse:
    try:
        result = summarize_quotes(payload.question, payload.hits)
    except SummarizationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SummarizeResponse(summary=result.summary, citations=result.citations)
