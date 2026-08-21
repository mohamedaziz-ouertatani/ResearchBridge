"""Response models for the corpus explorer API.

Deliberately exposes only ingested corpus data (papers, authors,
categories, embedding similarity). Extracted claims and evidence are NOT
exposed here: real (non-stub) extraction now exists and has been run over
the full corpus (extraction/hybrid.py), but nothing has designed how
imperfect, per-field-confidence extracted claims should be surfaced in
this API yet - that's a UI/API design task of its own, not a data
availability problem. See the Evidence docstring in db/models.py for the
stub-vs-real distinction extraction_method still carries.
"""

from __future__ import annotations

import uuid
from datetime import date

from pydantic import BaseModel


class PaperSummary(BaseModel):
    id: uuid.UUID
    source: str
    source_id: str
    title: str
    abstract: str | None
    publication_date: date | None
    url: str | None
    primary_category: str | None
    categories: list[str]
    authors: list[str]

    model_config = {"from_attributes": True}


class SearchHit(BaseModel):
    paper: PaperSummary
    distance: float
    """pgvector cosine distance: 0.0 is identical, 2.0 is maximally opposed."""


class PaperPage(BaseModel):
    items: list[PaperSummary]
    total: int
    limit: int
    offset: int


class CorpusStats(BaseModel):
    total_papers: int
    total_authors: int
    embedded_papers: int
    papers_by_year: dict[int, int]
    papers_by_category: dict[str, int]
