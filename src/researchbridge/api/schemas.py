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

from pydantic import BaseModel, Field


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


class ExtractedClaimOut(BaseModel):
    claim_type: str
    text: str
    confidence: str
    """"medium" or "low" - the extractor's own self-reported confidence,
    not a validated accuracy signal. Measured per-field precision varies
    widely (0.10-0.88 F1 depending on the field) and does not track this
    label cleanly: "problem" claims are labeled "low" whenever the
    extractor falls back to an abstract's opening sentence, yet that field
    measures as the single most reliable one. Treat this as provenance,
    not as a trustworthiness score."""
    section: str | None
    extraction_method: str

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


class GapEvidenceOut(BaseModel):
    paper_id: uuid.UUID
    paper_title: str
    text: str
    section: str | None

    model_config = {"from_attributes": True}


class CandidateGapOut(BaseModel):
    id: uuid.UUID
    seed_paper_id: uuid.UUID
    seed_paper_title: str
    observation: str
    gap_type: str
    status: str
    contributing_paper_count: int
    similarity_threshold: float
    detection_method: str
    review_note: str | None
    evidence: list[GapEvidenceOut]
    """Never presented as validated: gap_type is always "inference" (Sec 34),
    and status stays "pending" until a human reviews it here (Sec 35/44)."""

    model_config = {"from_attributes": True}


class CandidateGapReview(BaseModel):
    status: str
    review_note: str | None = None


class CandidateGapPage(BaseModel):
    items: list[CandidateGapOut]
    total: int
    limit: int
    offset: int


class ResearchInputOut(BaseModel):
    id: uuid.UUID
    input_type: str
    raw_text: str
    title: str | None

    model_config = {"from_attributes": True}


class ResearchAssessmentOut(BaseModel):
    id: uuid.UUID
    research_input: ResearchInputOut
    status: str
    retrieved_paper_ids: list[str]
    comparison_summary: str | None
    novelty_level: str
    novelty_reasoning: str | None
    research_gap_text: str | None
    research_gap_source: str | None
    """"reused_candidate_gap" | "input_specific" | None. Distinguishes an
    explicit, author-stated gap (never "inference" in the text) from a
    cross-paper inference (always labeled as such) - see assessment/gap.py."""
    candidate_gap_id: uuid.UUID | None
    potential_applications: list[dict] | None
    """Each item: {application, source_paper, paper_id} - an application a
    retrieved paper explicitly states, never a synthesized/invented one
    (see assessment/applications.py)."""
    technical_feasibility_level: str
    technical_feasibility_reasoning: str | None
    potential_opportunities: list[dict] | None
    """Deliberately always NULL for now - see assessment/opportunities.py:
    genuine Direct/Adjacent/Speculative opportunity framing requires
    synthesis this project's deterministic toolbox can't do without either
    fabricating content or adding a generative model, neither done yet."""
    risks_and_limitations: str | None
    """One line per relevant retrieved paper's own explicit limitations
    claim, grounded the same way as potential_applications (see
    assessment/risks.py) - never a synthesized new risk."""
    external_validation_needed: str
    """Always the same structural disclaimer (blueprint Sec 21): market/
    economic/commercialization judgments are never inferred from
    literature alone - see assessment/external_validation.py."""
    recommendation: str | None
    confidence: str | None
    """Recommendation is a later enrichment pass and stays NULL until
    actually computed (blueprint Sec 22: NULL is preferable to fabricated
    certainty)."""


class ResearchAssessmentCreate(BaseModel):
    raw_text: str = Field(min_length=1)


class CorpusStats(BaseModel):
    total_papers: int
    total_authors: int
    embedded_papers: int
    papers_by_year: dict[int, int]
    papers_by_category: dict[str, int]
