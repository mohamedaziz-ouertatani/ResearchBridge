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

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from researchbridge.api.deps import get_embedder, get_session
from researchbridge.api.schemas import (
    AskRequest,
    AskResponse,
    QaCollectionCreate,
    QaCollectionOut,
    QaCollectionSummaryOut,
    QaQuestionCreate,
    QaQuestionOut,
    QuoteHitOut,
    SummarizeRequest,
    SummarizeResponse,
)
from researchbridge.embedding.base import Embedder
from researchbridge.db.models import QaCollection, QaQuestion
from researchbridge.qa.answer import answer_question
from researchbridge.qa.summarize import SummarizationUnavailable, ollama_enabled, summarize_quotes

router = APIRouter(prefix="/api")


def _question_out(question: QaQuestion) -> QaQuestionOut:
    return QaQuestionOut(
        id=question.id,
        collection_id=question.collection_id,
        question=question.question,
        hits=[QuoteHitOut.model_validate(hit) for hit in question.hits],
        summary=question.summary,
        summary_citations=question.summary_citations,
        created_at=question.created_at,
    )


def _collection_or_404(session: Session, collection_id: uuid.UUID) -> QaCollection:
    collection = session.get(QaCollection, collection_id)
    if collection is None:
        raise HTTPException(status_code=404, detail=f"No Q&A collection with id {collection_id}")
    return collection


@router.post("/qa/collections", response_model=QaCollectionSummaryOut, status_code=status.HTTP_201_CREATED)
def create_collection(payload: QaCollectionCreate, session: Session = Depends(get_session)) -> QaCollectionSummaryOut:
    collection = QaCollection(title=payload.title)
    session.add(collection)
    session.commit()
    return QaCollectionSummaryOut(
        id=collection.id,
        title=collection.title,
        created_at=collection.created_at,
        updated_at=collection.updated_at,
        question_count=0,
    )


@router.get("/qa/collections", response_model=list[QaCollectionSummaryOut])
def list_collections(session: Session = Depends(get_session)) -> list[QaCollectionSummaryOut]:
    rows = session.execute(
        select(QaCollection, func.count(QaQuestion.id))
        .outerjoin(QaQuestion, QaQuestion.collection_id == QaCollection.id)
        .group_by(QaCollection.id)
        .order_by(QaCollection.updated_at.desc(), QaCollection.created_at.desc())
    ).all()
    return [
        QaCollectionSummaryOut(
            id=collection.id,
            title=collection.title,
            created_at=collection.created_at,
            updated_at=collection.updated_at,
            question_count=count,
        )
        for collection, count in rows
    ]


@router.get("/qa/collections/{collection_id}", response_model=QaCollectionOut)
def get_collection(collection_id: uuid.UUID, session: Session = Depends(get_session)) -> QaCollectionOut:
    collection = _collection_or_404(session, collection_id)
    questions = session.execute(
        select(QaQuestion).where(QaQuestion.collection_id == collection.id).order_by(QaQuestion.created_at.desc())
    ).scalars()
    return QaCollectionOut(
        id=collection.id,
        title=collection.title,
        created_at=collection.created_at,
        updated_at=collection.updated_at,
        questions=[_question_out(question) for question in questions],
    )


@router.patch("/qa/collections/{collection_id}", response_model=QaCollectionSummaryOut)
def rename_collection(
    collection_id: uuid.UUID, payload: QaCollectionCreate, session: Session = Depends(get_session)
) -> QaCollectionSummaryOut:
    collection = _collection_or_404(session, collection_id)
    collection.title = payload.title
    collection.updated_at = datetime.now(UTC)
    question_count = session.execute(
        select(func.count(QaQuestion.id)).where(QaQuestion.collection_id == collection.id)
    ).scalar_one()
    session.commit()
    return QaCollectionSummaryOut(
        id=collection.id,
        title=collection.title,
        created_at=collection.created_at,
        updated_at=collection.updated_at,
        question_count=question_count,
    )


@router.delete("/qa/collections/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_collection(collection_id: uuid.UUID, session: Session = Depends(get_session)) -> Response:
    collection = _collection_or_404(session, collection_id)
    session.delete(collection)
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/qa/collections/{collection_id}/questions", response_model=QaQuestionOut, status_code=status.HTTP_201_CREATED)
def save_question(
    collection_id: uuid.UUID,
    payload: QaQuestionCreate,
    session: Session = Depends(get_session),
    embedder: Embedder = Depends(get_embedder),
) -> QaQuestionOut:
    collection = _collection_or_404(session, collection_id)
    hits = answer_question(session, embedder, payload.question)
    hit_payload = [
        QuoteHitOut(
            evidence_id=hit.evidence_id,
            paper_id=hit.paper_id,
            paper_title=hit.paper_title,
            paper_source=hit.paper_source,
            claim_type=hit.claim_type,
            text=hit.text,
            section=hit.section,
            confidence=hit.confidence,
            score=hit.score,
        ).model_dump(mode="json")
        for hit in hits
    ]
    question = QaQuestion(collection_id=collection.id, question=payload.question, hits=hit_payload)
    collection.updated_at = datetime.now(UTC)
    session.add(question)
    session.commit()
    return _question_out(question)


@router.post("/qa/questions/{question_id}/summarize", response_model=QaQuestionOut)
def summarize_saved_question(question_id: uuid.UUID, session: Session = Depends(get_session)) -> QaQuestionOut:
    question = session.get(QaQuestion, question_id)
    if question is None:
        raise HTTPException(status_code=404, detail=f"No saved Q&A question with id {question_id}")
    hits = [QuoteHitOut.model_validate(hit) for hit in question.hits]
    if not hits:
        raise HTTPException(status_code=422, detail="saved question has no grounded hits to summarize")
    try:
        result = summarize_quotes(question.question, hits)
    except SummarizationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    question.summary = result.summary
    question.summary_citations = result.citations
    session.commit()
    return _question_out(question)


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
                evidence_id=hit.evidence_id,
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
