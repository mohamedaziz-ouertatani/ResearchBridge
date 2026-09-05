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
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


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
    excluded_at: datetime | None
    """Set when an operator has excluded this paper from search, retrieval,
    and gap-detection seeding (blueprint corpus curation). NULL = included."""

    model_config = {"from_attributes": True}


class PaperFullTextOut(BaseModel):
    sections: dict[str, str]
    source_url: str

    model_config = {"from_attributes": True}


class ExtractedClaimOut(BaseModel):
    claim_type: str
    text: str
    confidence: str
    """"high", "medium", or "low" - the extractor's own self-reported
    confidence, not a validated accuracy signal ("high" only occurs for a
    cue-phrase hit in a named full-text section; abstract-only matches top
    out at "medium"). Measured per-field precision varies widely
    (0.10-0.88 F1 depending on the field) and does not track this label
    cleanly: "problem" claims are labeled "low" whenever the extractor
    falls back to an abstract's opening sentence, yet that field measures
    as the single most reliable one. Treat this as provenance, not as a
    trustworthiness score."""
    section: str | None
    """"abstract", or a full-text section name (e.g. "methods",
    "discussion") when the paper has full text and the claim was drawn
    from it instead."""
    extraction_method: str

    model_config = {"from_attributes": True}


class SearchHit(BaseModel):
    paper: PaperSummary
    distance: float
    """pgvector cosine distance: 0.0 is identical, 2.0 is maximally opposed."""


class AskRequest(BaseModel):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class QuoteHitOut(BaseModel):
    paper_id: uuid.UUID
    paper_title: str
    paper_source: str
    claim_type: str
    text: str
    section: str | None
    confidence: str
    score: float


class AskResponse(BaseModel):
    hits: list[QuoteHitOut]
    summarization_available: bool


class SummarizeRequest(BaseModel):
    question: str = Field(min_length=1)
    hits: list[QuoteHitOut] = Field(min_length=1, max_length=20)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class SummarizeResponse(BaseModel):
    summary: str
    citations: list[int]


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
    claim_type: str
    validation_tier: str | None
    """"strong"/"weak" for a research_gap claim, None for limitations or any
    claim predating this column (see extraction/validation.py)."""
    claim_role: str | None
    """"anchor"/"supporting" - this evidence's role within its cluster. None
    means it was classified "motivation" (excluded from corroborating the
    gap, but still shown for transparency - see gaps/persistence.py)."""
    self_resolution_signal: bool
    field_scope_signal: bool
    own_contribution_overlap: float

    model_config = {"from_attributes": True}


class AnalysisClaimOut(BaseModel):
    id: uuid.UUID
    claim_type: str
    claim_text: str
    confidence: str
    status: str
    """For a gap-derived claim: mirrors the linked CandidateGap's status
    (see gaps/claims.py). For an assessment-derived claim: mirrors
    ResearchAssessment.human_reviewed - "approved" once reviewed, "pending"
    otherwise (see assessment/claims.py's sync_claim_status)."""

    model_config = {"from_attributes": True}


class ClaimEvidenceOut(BaseModel):
    paper_id: uuid.UUID
    paper_title: str
    text: str
    section: str | None
    relationship: str

    model_config = {"from_attributes": True}


class AnalysisClaimDetailOut(BaseModel):
    id: uuid.UUID
    claim_type: str
    claim_text: str
    confidence: str
    status: str
    source_table: str
    source_id: uuid.UUID
    created_at: datetime
    evidence: list[ClaimEvidenceOut]
    """The real quoted passages backing this claim - what makes it
    inspectable rather than trusted prose (Sec 15/16)."""

    model_config = {"from_attributes": True}


class AnalysisClaimPage(BaseModel):
    items: list[AnalysisClaimDetailOut]
    total: int
    limit: int
    offset: int


class CandidateGapOut(BaseModel):
    id: uuid.UUID
    seed_paper_id: uuid.UUID
    seed_paper_title: str
    observation: str
    gap_type: str
    status: str
    gap_status: str | None
    resolution_note: str | None
    contributing_paper_count: int
    similarity_threshold: float
    detection_method: str
    review_note: str | None
    correctness_rating: int | None
    relevance_rating: int | None
    novelty_rating: int | None
    evidence_support_rating: int | None
    usefulness_rating: int | None
    """Sec 44's five human-evaluation dimensions, 0-3 - see CandidateGap's
    docstring. Optional even on a reviewed gap."""
    evidence: list[GapEvidenceOut]
    """Never presented as validated: gap_type is always "inference" (Sec 34),
    and status stays "pending" until a human reviews it here (Sec 35/44)."""
    claim: AnalysisClaimOut | None
    """The Sec 16 structured-reasoning mirror of this gap, if one was
    created (see gaps/claims.py). None for gaps saved before that module
    existed - not an error."""

    model_config = {"from_attributes": True}


class CandidateGapReview(BaseModel):
    status: str
    review_note: str | None = None
    correctness_rating: int | None = None
    relevance_rating: int | None = None
    novelty_rating: int | None = None
    evidence_support_rating: int | None = None
    usefulness_rating: int | None = None


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
    matched_paper_id: uuid.UUID | None
    """Set only for input_type = document, when the upload could be
    identified as an already-ingested corpus paper (Sec 2A). An
    optimization signal only - never required for the assessment to
    proceed, and never set for idea-text input."""

    model_config = {"from_attributes": True}


class AssessmentEvidenceOut(BaseModel):
    role: str
    """Which report field this passage backs: comparison | novelty |
    research_gap | application | feasibility | risk | opportunity."""
    paper_id: uuid.UUID
    paper_title: str
    text: str
    section: str | None
    paper_url: str | None = None
    paper_doi: str | None = None

    model_config = {"from_attributes": True}


class ResearchAssessmentOut(BaseModel):
    id: uuid.UUID
    created_at: datetime | None = None
    research_input: ResearchInputOut
    status: str
    retrieved_paper_ids: list[str]
    comparison_summary: str | None
    novelty_level: str
    novelty_reasoning: str | None
    research_gap_text: str | None
    research_gap_source: str | None
    """"reused_candidate_gap" | "input_specific" | "no_relevant_evidence" |
    "checked_no_gap_found". Distinguishes an explicit, author-stated gap
    (never "inference" in the text) from a cross-paper inference (always
    labeled as such) - see assessment/gap.py. When research_gap_text is
    None, this still distinguishes "no_relevant_evidence" (insufficient
    retrieved evidence to investigate at all) from "checked_no_gap_found"
    (relevant evidence was checked, nothing surfaced) - never collapse
    both into the same message."""
    candidate_gap_id: uuid.UUID | None
    potential_applications: list[dict] | None
    """Each item: {application, source_paper, paper_id} - an application a
    retrieved paper explicitly states, never a synthesized/invented one
    (see assessment/applications.py)."""
    technical_feasibility_level: str
    technical_feasibility_reasoning: str | None
    potential_opportunities: list[dict] | None
    """NULL until a user explicitly requests synthesis via POST
    /api/assessments/{id}/opportunities (see assessment/opportunity_
    synthesis.py) - the one field in this report generated by a local LLM
    rather than extracted, since genuine Direct/Adjacent/Speculative
    opportunity framing requires inventing a concept beyond what any paper
    states. Each item: {tier, opportunity, source_applications: [{application,
    paper_id, paper_title}]}, every citation checked against the
    assessment's own potential_applications before being shown."""
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
    human_reviewed: bool
    """Set only via PUT /api/assessments/{id}/review (Sec 35) - never by the
    pipeline itself. A boolean, not a status enum: unlike candidate_gaps,
    an assessment isn't accepted/rejected as a unit, just marked looked-at."""
    evidence: list[AssessmentEvidenceOut]
    """Every populated field above traces back to real quoted passages here
    (Sec 15/17). An assessment is a lightweight summary, never its own
    source of truth - if a field can't point at evidence, it is NULL."""
    claims: list[AnalysisClaimOut]
    """The Sec 16 structured-reasoning mirror of this assessment's
    comparison/novelty/research_gap/feasibility/risk text fields (see
    assessment/claims.py) - empty for an assessment predating that module,
    or one where none of those five fields were populated."""


class ResearchAssessmentCreate(BaseModel):
    raw_text: str = Field(min_length=1)

    @field_validator("raw_text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("raw_text must not be blank")
        return v


class GraphNodeOut(BaseModel):
    id: str
    type: Literal["input", "paper"]
    title: str
    distance_to_input: float | None
    claim_counts: dict[str, int]


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    distance: float


class SimilarityGraphOut(BaseModel):
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]


class CitationNodeOut(BaseModel):
    id: str
    type: Literal["center", "paper"]
    title: str
    hop: int
    """Citation distance from the center paper: 0 for the center itself, 1/2
    for direct/second-hop citations (see _build_citation_graph)."""


class CitationEdgeOut(BaseModel):
    source: str
    target: str
    direction: Literal["cites", "cited_by"]
    sources: list[str]
    """Which citation source(s) assert this edge (e.g. ["semantic_scholar"],
    ["crossref"], or both) - one visual edge even when multiple sources
    independently found the same citing/cited pair (see paper_citations'
    (citing, cited, source) unique constraint, which stores one row per
    source)."""


class CitationGraphOut(BaseModel):
    nodes: list[CitationNodeOut]
    edges: list[CitationEdgeOut]


class ResearchAssessmentHistoryItem(BaseModel):
    """One entry in a research_input's assessment history (re-runs, Sec 2A).

    Deliberately a small subset of ResearchAssessmentOut's fields - a
    history list is for telling sibling assessments apart at a glance
    (when, what novelty/status, reviewed or not), not for shipping every
    field's full evidence payload for every entry."""

    id: uuid.UUID
    status: str
    novelty_level: str
    human_reviewed: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ResearchAssessmentReview(BaseModel):
    human_reviewed: bool


class PaperExclude(BaseModel):
    excluded: bool


class CorpusStats(BaseModel):
    total_papers: int
    total_authors: int
    embedded_papers: int
    papers_with_claims: int
    papers_by_year: dict[int, int]
    papers_by_category: dict[str, int]
    papers_by_source: dict[str, int]


class TrendsOut(BaseModel):
    category: str
    years: list[int]
    series: dict[str, list[int]]
    """claim_type -> counts, each list the same length as `years` and aligned
    to it by index. Only claim_types with at least one non-stub claim in this
    category appear as keys."""


class PipelineRunOut(BaseModel):
    id: uuid.UUID
    status: str
    started_at: datetime
    finished_at: datetime | None
    error_summary: str | None
    source: str | None
    """Set only for ingestion runs (e.g. "arxiv"/"springer") - lets the UI
    split one combined history into per-source sections. Always None for
    extraction/embedding runs, which have no equivalent notion of source."""
    counts: dict[str, int]
    """Run-type-specific numeric fields, e.g. records_fetched/inserted/
    duplicate/failed for an ingestion run - kept as a dict rather than a
    field per run type so one schema covers all three pipeline stages."""
    pid: int | None
    """The OS process id this run's own row recorded at creation time (see
    db/models.py's IngestionRun.pid docstring) - shown for a still-"running"
    row so an operator can correlate it with `ps`/Task Manager, e.g. to
    confirm a CLI-started run the admin panel now also detects and to stop
    it manually if needed. None for runs from before this column existed,
    and for any run once it's no longer "running" (kept only while it's
    live and might need correlating)."""


class AssessmentStats(BaseModel):
    total: int
    """The latest assessment per research_input - re-run history collapses
    to one, same as GET /api/assessments (Sec 2A)."""
    needs_review: int


class CorpusHealth(BaseModel):
    missing_doi: int
    """Papers with no DOI - unreachable by the CrossRef citation pass."""
    excluded: int
    """Papers with excluded_at set (see PUT /api/admin/papers/{id}/exclude)."""
    claims_without_embeddings: int
    """Papers that have at least one ExtractedClaim but no Embedding yet -
    stuck between the extraction and embedding pipeline stages."""
    no_citation_coverage: int
    """Papers eligible for at least one citation source (has a DOI, or
    source="semantic_scholar") with zero outgoing PaperCitation edges from
    any source yet."""
    not_gap_processed: int
    """Embedded, non-excluded papers gap detection hasn't seen yet - the
    same seed-selection criterion as gaps/batch.py's _select_seed_papers
    (force=False), so this is exactly the backlog a plain `rb-gaps-detect
    --all --save` run would work through next."""


class GapReviewStats(BaseModel):
    pending: int
    approved: int
    rejected: int
    mean_correctness: float | None
    mean_relevance: float | None
    mean_novelty: float | None
    mean_evidence_support: float | None
    mean_usefulness: float | None
    """Mean of each Sec 44 rating dimension across gaps that have been rated
    on it - None when nothing has a rating yet. This is the RQ3/RQ4 number:
    how the reviewer scored candidate gaps, not just how many were approved."""


class PipelineStatus(BaseModel):
    total_papers: int
    papers_with_claims: int
    papers_with_embeddings: int
    papers_with_fulltext: int
    papers_by_source: dict[str, int]
    corpus_health: CorpusHealth
    assessment_stats: AssessmentStats
    gap_stats: GapReviewStats
    ingestion_errors_by_type: dict[str, int]
    """Grouped counts of the most recent ingestion errors (see
    _ingestion_errors_by_type's ERROR_SAMPLE_LIMIT) - a sample for spotting
    what's failing and why, not an exhaustive historical count."""
    analysis_claims_by_type: dict[str, int]
    """All-time count of every analysis_claims row by claim_type (fact/
    inference/hypothesis/opportunity/speculation), across both producers
    (gaps/claims.py, assessment/claims.py)."""
    ingestion_runs: list[PipelineRunOut]
    extraction_runs: list[PipelineRunOut]
    embedding_runs: list[PipelineRunOut]
    citation_fetch_runs: list[PipelineRunOut]
    gap_detection_runs: list[PipelineRunOut]
    """Run history for rb-gaps-detect --all, same shape as the other *_runs
    lists. Started via POST /api/gaps/detect (gaps_routes.py's own trigger,
    always --all --save, no configurable params) rather than one of this
    router's own /{stage}/run endpoints, but tracked in the same
    pipeline_triggers subprocess registry under key "gaps" - see
    `running` below."""
    fulltext_fetch_runs: list[PipelineRunOut]
    """Run history for rb-fulltext-fetch, same shape as the other *_runs
    lists. Started via POST /api/admin/fulltext/run."""
    running: dict[str, bool]
    """Whether a pipeline is actually running right now, per pipeline key -
    see PIPELINE_KEYS in admin_routes.py for the full, current set
    (ingestion x4, extraction, embedding, retrieval_eval, extraction_eval,
    citations_fetch, fulltext). True when either this server has a live
    subprocess handle for it (started via the trigger button), or its own
    *_runs row still says status="running" AND that row's pid is a
    process that's actually alive right now - covering a run started
    directly from the CLI, or one that outlived a server restart. A row
    that just says "running" with no verifiably-live pid behind it (a
    crashed or killed process that never reached the code to mark itself
    failed/stopped) does NOT count - see admin_routes.py's
    _has_running_db_row for the full reasoning."""


class Notification(BaseModel):
    id: str
    """Stable per underlying event so a client can track which ones it has
    already shown. Run-based notifications key off the run's own id
    (unique forever); aggregate ones (needs_review/gaps_pending) key off
    their current count, so a client that has already seen "needs_review:5"
    treats "needs_review:6" as new but never re-notifies on an unchanged
    count."""
    type: str
    """One of: ingestion_completed, ingestion_failed, extraction_completed,
    extraction_failed, embedding_completed, embedding_failed,
    citations_fetch_completed, citations_fetch_failed,
    gap_detection_completed, gap_detection_failed, needs_review,
    gaps_pending. Any of the run-based types can also end in "_stopped"
    (an operator-stopped run, same severity as "_completed")."""
    severity: str
    """"info" or "error" - error for a failed run, info otherwise."""
    message: str
    created_at: datetime


class ArxivIngestionTrigger(BaseModel):
    search_query: str | None = None
    page_size: int | None = None
    max_pages: int | None = None


class SpringerIngestionTrigger(BaseModel):
    query: str | None = None
    page_size: int | None = None
    max_pages: int | None = None


class SemanticScholarIngestionTrigger(BaseModel):
    query: str | None = None
    max_pages: int | None = None


class CoreIngestionTrigger(BaseModel):
    query: str | None = None
    page_size: int | None = None
    max_pages: int | None = None


class ExtractionTrigger(BaseModel):
    limit: int | None = None
    extractor: str | None = None
    force: bool = False


class EmbeddingTrigger(BaseModel):
    limit: int | None = None
    force: bool = False


class FullTextFetchTrigger(BaseModel):
    limit: int | None = None
    force: bool = False


class RetrievalEvalTrigger(BaseModel):
    k: int | None = None


class PipelineTriggerOut(BaseModel):
    started: bool
    pipeline: str
    log_file: str


class RetrievalEvalMethodResult(BaseModel):
    method: str
    precision: float
    recall: float
    ndcg: float
    mrr: float


class RetrievalEvalQuerySet(BaseModel):
    queries: int
    skipped: int
    results: list[RetrievalEvalMethodResult]


class RetrievalEvalOut(BaseModel):
    available: bool
    """False when rb-retrieval-evaluate has never been run - see
    admin_routes.py's RETRIEVAL_EVAL_RESULTS_PATH. No run-history table for
    this (same choice as candidate-gap detection): it's a one-off
    diagnostic, not a repeating pipeline stage."""
    generated_at: datetime | None
    k: int | None
    query_sets: dict[str, RetrievalEvalQuerySet] | None


class ExtractionEvalTrigger(BaseModel):
    threshold: float | None = None
    extractor: str | None = None


class ExtractionEvalFieldScore(BaseModel):
    precision: float
    recall: float
    f1: float


class ExtractionEvalOut(BaseModel):
    available: bool
    """False when rb-extract-evaluate has never been run - same one-off-
    diagnostic, no-run-history-table choice as RetrievalEvalOut."""
    generated_at: datetime | None
    threshold: float | None
    paper_count: int | None
    extractors: dict[str, dict[str, ExtractionEvalFieldScore]] | None
    """extractor name -> field name -> score, e.g.
    {"hybrid": {"problem": {"precision": 0.8, ...}}}."""


class CitationsFetchTrigger(BaseModel):
    source: Literal["semantic_scholar", "crossref"] = "semantic_scholar"
    force: bool = False


class PipelineStopOut(BaseModel):
    stopped: bool
    pipeline: str


class GapsDetectStatus(BaseModel):
    running: bool
    log: str


class ResearchAssessmentSummaryOut(BaseModel):
    """One dashboard row (GET /api/assessments) - the latest assessment for
    one research_input, not every re-run (see assessment_routes.py's list
    query). A small subset of ResearchAssessmentOut's fields, same reasoning
    as ResearchAssessmentHistoryItem: enough to tell entries apart in a list,
    not the full evidence-backed report."""

    id: uuid.UUID
    created_at: datetime
    status: str
    novelty_level: str
    technical_feasibility_level: str
    recommendation: str | None
    confidence: str | None
    human_reviewed: bool
    research_input_id: uuid.UUID
    input_type: str
    input_preview: str
    """raw_text (or filename, for an upload) truncated to a list-friendly length."""


class ResearchAssessmentSummaryPage(BaseModel):
    items: list[ResearchAssessmentSummaryOut]
    total: int
    limit: int
    offset: int
