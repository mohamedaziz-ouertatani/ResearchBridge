"""SQLAlchemy models for the Phase 1 (weeks 1-4) ingestion slice."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, Date, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size — see embedding/model.py


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"
    __table_args__ = (UniqueConstraint("source", "source_id", name="uq_papers_source_source_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String, nullable=False)
    source_id: Mapped[str] = mapped_column(String, nullable=False)
    doi: Mapped[str | None] = mapped_column(String, nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    publication_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    venue: Mapped[str | None] = mapped_column(Text, nullable=True)
    document_type: Mapped[str | None] = mapped_column(String, nullable=True)
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    open_access: Mapped[bool | None] = mapped_column(nullable=True)
    raw_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    ingestion_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
    excluded_at: Mapped[datetime | None] = mapped_column(nullable=True)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    """The OS process id that created this row, set once at row-creation
    time (never reused for a later re-check). admin_routes.py's liveness
    check uses this to tell a still-alive run apart from one whose process
    crashed or was killed without ever reaching the code that marks
    status "failed"/"stopped" - a status="running" row alone, with no
    process actually behind it, is not itself proof anything is running.
    NULL for rows written before this column existed."""
    resume_state: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    records_fetched: Mapped[int] = mapped_column(Integer, default=0)
    records_inserted: Mapped[int] = mapped_column(Integer, default=0)
    records_duplicate: Mapped[int] = mapped_column(Integer, default=0)
    records_failed: Mapped[int] = mapped_column(Integer, default=0)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)


class Author(Base):
    """An author identity.

    Identity here is a provisional heuristic: rows are deduped by exact
    name-string match only (arXiv gives no ORCID or other stable
    identifier). Two different people who publish under the same name
    string will incorrectly collapse into one row; two spellings of the
    same person will incorrectly stay separate. This is a known MVP
    limitation, not a correctness guarantee.

    The schema doesn't need special "merge" columns to stay extensible: a
    future identity-resolution step can merge two Author rows by
    repointing their paper_authors.author_id rows onto one survivor and
    removing the other — an ordinary FK repoint, not a schema change.
    """

    __tablename__ = "authors"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    orcid: Mapped[str | None] = mapped_column(String, nullable=True)
    author_metadata: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class PaperAuthor(Base):
    __tablename__ = "paper_authors"

    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id"), primary_key=True)
    author_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("authors.id"), primary_key=True)
    author_order: Mapped[int] = mapped_column(Integer, nullable=False)


class PaperCategory(Base):
    """A category label attached to a paper.

    `confidence="high"` for arXiv rows means the category was directly
    provided by arXiv (author-assigned at submission) — it describes the
    provenance of the label, not an independent check that the category is
    actually correct or well-chosen.
    """

    __tablename__ = "paper_categories"
    __table_args__ = (UniqueConstraint("paper_id", "category", "source", name="uq_paper_categories_paper_category_source"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)
    category: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)


class PaperCitation(Base):
    """Citation edges between papers.

    Schema-only for now: no connector currently supplies citation data (see
    §12/§30 of the blueprint — that's Semantic Scholar's job, added in a
    later phase). This table stays empty until a citation-capable source is
    integrated; nothing in the current pipeline writes to it.
    """

    __tablename__ = "paper_citations"
    __table_args__ = (
        UniqueConstraint("citing_paper_id", "cited_paper_id", "source", name="uq_paper_citations_edge_source"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    citing_paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)
    cited_paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[str] = mapped_column(String, nullable=False)


class CitationFetchRun(Base):
    """One rb-citations-fetch --all run - same run-history shape as
    IngestionRun/ExtractionRun/EmbeddingRun, updated incrementally as the
    batch processes papers (see citations/batch.py's _sync_run) rather
    than only once the whole run finishes."""

    __tablename__ = "citation_fetch_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    """See IngestionRun.pid's docstring - the liveness check this backs."""
    papers_seen: Mapped[int] = mapped_column(Integer, default=0)
    papers_failed: Mapped[int] = mapped_column(Integer, default=0)
    edges_created: Mapped[int] = mapped_column(Integer, default=0)
    edges_already_existed: Mapped[int] = mapped_column(Integer, default=0)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)


class IngestionError(Base):
    __tablename__ = "ingestion_errors"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ingestion_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ingestion_runs.id"), nullable=False
    )
    source: Mapped[str] = mapped_column(String, nullable=False)
    raw_payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    error_type: Mapped[str] = mapped_column(String, nullable=False)
    error_detail: Mapped[str] = mapped_column(Text, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Evidence(Base):
    """A grounded, source-anchored text span backing an ExtractedClaim.

    `extraction_method`/`model_version` record provenance (e.g. "stub"/
    "stub-v1" for the placeholder extractor, per the blueprint's
    Free/Open-Source First provider-agnostic design — see §29). Rows with
    extraction_method="stub" are synthetic placeholder data for exercising
    the pipeline, not real analysis: never treat them as benchmark ground
    truth, and filter them out of any real evaluation or research-content
    query.
    """

    __tablename__ = "evidence"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)
    evidence_type: Mapped[str] = mapped_column(String, nullable=False)
    section: Mapped[str | None] = mapped_column(String, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source_locator: Mapped[str | None] = mapped_column(String, nullable=True)
    extraction_method: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ExtractedClaim(Base):
    """A system-generated claim about a paper, grounded by one Evidence row.

    Same caveat as Evidence: check the linked evidence.extraction_method
    before treating a row as real research content. Stub-generated rows
    (extraction_method="stub" on their evidence) are synthetic test data
    only.
    """

    __tablename__ = "extracted_claims"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)
    claim_type: Mapped[str] = mapped_column(String, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("evidence.id"), nullable=False)
    confidence: Mapped[str] = mapped_column(String, nullable=False)
    validation_tier: Mapped[str | None] = mapped_column(String, nullable=True)
    """"strong"/"weak" for a research_gap claim that passed extraction
    validation (see extraction/validation.py::ValidationResult.tier) - None
    for every other claim_type. Read by gaps/detect.py to decide whether a
    claim can anchor a candidate gap on its own."""
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ExtractionRun(Base):
    __tablename__ = "extraction_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    extractor_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    """See IngestionRun.pid's docstring - the liveness check this backs."""
    papers_processed: Mapped[int] = mapped_column(Integer, default=0)
    claims_created: Mapped[int] = mapped_column(Integer, default=0)
    candidates_rejected: Mapped[int] = mapped_column(Integer, default=0)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    force: Mapped[bool] = mapped_column(nullable=False, default=False)


class ExtractionError(Base):
    __tablename__ = "extraction_errors"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    extraction_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("extraction_runs.id"), nullable=False
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)
    error_type: Mapped[str] = mapped_column(String, nullable=False)
    error_detail: Mapped[str] = mapped_column(Text, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Embedding(Base):
    """A dense vector representation of a paper's text (title + abstract, for now).

    `embedding_type`/`model_name` let more than one embedding coexist per
    paper later (e.g. a different model, or a full-text embedding once
    Phase 1's abstract-only scope is revisited) without a schema change —
    the unique constraint is per (paper, type, model), not per paper alone.
    Vectors are stored L2-normalized, so pgvector's cosine-distance operator
    (`<=>`) and inner-product operator agree — see embedding/model.py.
    """

    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint("paper_id", "embedding_type", "model_name", name="uq_embeddings_paper_type_model"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)
    embedding_type: Mapped[str] = mapped_column(String, nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    vector: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class EmbeddingRun(Base):
    __tablename__ = "embedding_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    """See IngestionRun.pid's docstring - the liveness check this backs."""
    papers_processed: Mapped[int] = mapped_column(Integer, default=0)
    papers_skipped: Mapped[int] = mapped_column(Integer, default=0)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    force: Mapped[bool] = mapped_column(nullable=False, default=False)


class GapDetectionRun(Base):
    """One rb-gaps-detect --all run - same run-history shape as
    IngestionRun/ExtractionRun/EmbeddingRun/CitationFetchRun, updated
    incrementally as the batch processes seed papers (see gaps/batch.py's
    _sync_run) rather than only once the whole run finishes.

    Distinct from CandidateGap.status: this table tracks whether the batch
    itself ran and how far it got, not the review state of what it found -
    a run can complete with gaps_saved=0 and that's a real, non-error
    outcome (see BatchSummary's no_relevant_papers/insufficient_evidence)."""

    __tablename__ = "gap_detection_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    """See IngestionRun.pid's docstring - the liveness check this backs."""
    papers_seen: Mapped[int] = mapped_column(Integer, default=0)
    papers_skipped: Mapped[int] = mapped_column(Integer, default=0)
    papers_failed: Mapped[int] = mapped_column(Integer, default=0)
    no_relevant_papers: Mapped[int] = mapped_column(Integer, default=0)
    insufficient_evidence: Mapped[int] = mapped_column(Integer, default=0)
    gaps_found: Mapped[int] = mapped_column(Integer, default=0)
    gaps_saved: Mapped[int] = mapped_column(Integer, default=0)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    force: Mapped[bool] = mapped_column(nullable=False, default=False)


class CandidateGap(Base):
    """Phase 3 (Sec 32): an inferred recurring pattern across a paper's neighborhood.

    NOT an ExtractedClaim: a claim is one statement about one paper, grounded
    by one Evidence row in that same paper. A candidate gap is structurally
    different - an observation synthesized ACROSS several papers' limitation/
    research_gap claims, grounded by evidence spread across all of them (see
    CandidateGapEvidence). That cross-paper shape doesn't fit the claims
    model, which is why this is a new table rather than a new claim_type.

    gap_type is "inference", always - never "fact". Sec 34 requires the UI
    and data model to distinguish evidence/fact from inference from
    hypothesis; this table only ever produces the middle one. Turning a
    recurring pattern into an actual proposed opportunity (a hypothesis) is
    a further, even-less-grounded step this table does not attempt - see
    gaps/cluster.py.

    status starts "pending" and stays there until a human reviews it (Sec 35,
    Sec 44: gap detection is evaluated by a person - correctness, relevance,
    novelty, evidence support, usefulness - never by a formula). Nothing
    here should be presented to an end user as validated until reviewed.
    Valid values: "pending", "approved", "rejected" - enforced both by a DB
    CHECK constraint (see __table_args__) and at the API layer
    (api/gaps_routes.py), so a bad status can't reach the table via any path.
    """

    __tablename__ = "candidate_gaps"
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'approved', 'rejected')", name="ck_candidate_gaps_status"),
        CheckConstraint(
            "gap_status IS NULL OR gap_status IN ('strong_gap', 'potential_gap', 'known_limitation')",
            name="ck_candidate_gaps_gap_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    seed_paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)
    observation: Mapped[str] = mapped_column(Text, nullable=False)
    gap_type: Mapped[str] = mapped_column(String, nullable=False, default="inference")
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    contributing_paper_count: Mapped[int] = mapped_column(Integer, nullable=False)
    similarity_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    detection_method: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    reviewed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    correctness_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    relevance_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    novelty_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    evidence_support_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    usefulness_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    """Sec 44's five human-evaluation dimensions, each 0-3 (irrelevant/weak/
    plausible/highly relevant) - enforced by ck_candidate_gaps_*_rating_range
    (see migration 0012). Optional: a reviewer can approve/reject without
    rating, same as review_note already being optional."""
    gap_status: Mapped[str | None] = mapped_column(String, nullable=True)
    """"strong_gap"/"potential_gap"/"known_limitation" - the deterministic
    classification from gaps/signals.py::classify_cluster, computed once at
    detection time and stored so the review UI and API don't recompute it.
    NULL for any CandidateGap saved before this column existed."""
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Set only when gaps/signals.py::find_addressing_papers found another
    paper in the corpus whose contribution/results claim closely resembles
    this pattern - a downgrade note for the reviewer, never proof the gap
    is closed (see apply_addressing_downgrade)."""


class CandidateGapEvidence(Base):
    """Links one CandidateGap to one contributing paper's Evidence row.

    One row per contributing paper (typically 3+, per gaps/cluster.py's
    minimum cluster size) - this is what makes a candidate gap's support
    inspectable: which papers, which exact quoted passages.
    """

    __tablename__ = "candidate_gap_evidence"
    __table_args__ = (
        UniqueConstraint("candidate_gap_id", "evidence_id", name="uq_candidate_gap_evidence_gap_evidence"),
        CheckConstraint(
            "claim_role IS NULL OR claim_role IN ('anchor', 'supporting')",
            name="ck_candidate_gap_evidence_claim_role",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_gap_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("candidate_gaps.id"), nullable=False
    )
    evidence_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("evidence.id"), nullable=False)
    claim_role: Mapped[str | None] = mapped_column(String, nullable=True)
    """"anchor"/"supporting" - this evidence row's classify_claim role
    within its candidate gap's cluster (gaps/signals.py). NULL for rows
    saved before this column existed."""
    self_resolution_signal: Mapped[bool | None] = mapped_column(nullable=True)
    field_scope_signal: Mapped[bool | None] = mapped_column(nullable=True)
    own_contribution_overlap: Mapped[float | None] = mapped_column(Float, nullable=True)


class AnalysisClaim(Base):
    """A structured, typed piece of system-generated reasoning (Sec 16).

    Distinct from ExtractedClaim (one statement about one paper, grounded by
    one Evidence row in that same paper): an AnalysisClaim is reasoning
    ABOUT the corpus - it can synthesize across evidence from several
    papers, same shape as CandidateGap but not limited to gap detection.

    This first slice only ever writes claim_type="inference", one per
    CandidateGap (source_table="candidate_gaps", source_id=candidate_gap.id)
    - see gaps/claims.py. Other claim_type/source_table combinations are
    schema-only until a producer writes them, same pattern as
    PaperCitation staying empty until a citation-capable source exists.

    status mirrors whatever the linked source row's own review status is
    (see gaps/claims.py::sync_claim_status) rather than being reviewed
    independently - this table is a structured mirror of that reasoning,
    not a second place to approve/reject it.
    """

    __tablename__ = "analysis_claims"
    __table_args__ = (
        CheckConstraint(
            "claim_type IN ('fact', 'inference', 'hypothesis', 'opportunity', 'speculation')",
            name="ck_analysis_claims_claim_type",
        ),
        CheckConstraint("status IN ('pending', 'approved', 'rejected')", name="ck_analysis_claims_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_type: Mapped[str] = mapped_column(String, nullable=False)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    source_table: Mapped[str] = mapped_column(String, nullable=False)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ClaimEvidence(Base):
    """Links one AnalysisClaim to one Evidence row backing it (Sec 16).

    Mirrors CandidateGapEvidence/ResearchAssessmentEvidence: what makes a
    claim's text inspectable rather than trusted as its own source of truth.
    """

    __tablename__ = "claim_evidence"
    __table_args__ = (
        UniqueConstraint(
            "claim_id", "evidence_id", "relationship", name="uq_claim_evidence_claim_evidence_relationship"
        ),
        CheckConstraint(
            "relationship IN ('supports', 'contradicts', 'contextualizes')",
            name="ck_claim_evidence_relationship",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    claim_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("analysis_claims.id"), nullable=False)
    evidence_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("evidence.id"), nullable=False)
    relationship: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class PaperFullText(Base):
    """A paper's full text, fetched from its open-access PDF and split into
    heuristic sections (Sec 46). One row per paper - re-fetching under
    --force updates this row in place rather than inserting a duplicate.

    Consumed by extraction/pipeline.py, which looks up a paper's row here
    (if any) and passes its sections to the Extractor in use.
    """

    __tablename__ = "paper_fulltext"
    __table_args__ = (UniqueConstraint("paper_id", name="uq_paper_fulltext_paper_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)
    sections: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    fetch_method: Mapped[str] = mapped_column(String, nullable=False, default="pymupdf")
    fetched_at: Mapped[datetime] = mapped_column(server_default=func.now())


class FullTextFetchRun(Base):
    """One rb-fulltext-fetch run - same run-history shape as
    ExtractionRun/EmbeddingRun/CitationFetchRun/GapDetectionRun, updated
    incrementally as the batch processes papers."""

    __tablename__ = "fulltext_fetch_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    """See IngestionRun.pid's docstring - the liveness check this backs."""
    papers_seen: Mapped[int] = mapped_column(Integer, default=0)
    papers_fetched: Mapped[int] = mapped_column(Integer, default=0)
    papers_skipped_no_url: Mapped[int] = mapped_column(Integer, default=0)
    papers_failed: Mapped[int] = mapped_column(Integer, default=0)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    force: Mapped[bool] = mapped_column(nullable=False, default=False)


class FullTextFetchError(Base):
    __tablename__ = "fulltext_fetch_errors"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fulltext_fetch_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("fulltext_fetch_runs.id"), nullable=False
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)
    error_type: Mapped[str] = mapped_column(String, nullable=False)
    error_detail: Mapped[str] = mapped_column(Text, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ResearchInput(Base):
    """One user-submitted item to be assessed (blueprint Sec 2A).

    input_type is "idea" (free-text prompt, no attached document) for the
    first vertical slice - "document" (uploaded paper) is deliberately not
    yet supported, per the roadmap's build-order guidance (Sec 45): ship
    idea-only end-to-end before adding upload/PDF-parsing surface area.
    """

    __tablename__ = "research_inputs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    input_type: Mapped[str] = mapped_column(String, nullable=False, default="idea")
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    matched_paper_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id"), nullable=True
    )
    source_filename: Mapped[str | None] = mapped_column(String, nullable=True)
    submitted_by: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ResearchAssessment(Base):
    """One analysis session over one ResearchInput (blueprint Sec 2A).

    Deliberately lightweight: a summary snapshot, never the source of
    truth. That's ResearchAssessmentEvidence, which grounds every
    populated field in a real Evidence row - same "never invent" rule as
    CandidateGapEvidence/Sec 15.

    This first vertical slice only ever populates retrieved_paper_ids,
    comparison_summary, status, and completed_at. Every other field stays
    NULL/"not_assessed" until a later enrichment pass computes it for
    real - NULL is preferable to fabricated certainty (Sec 22).
    """

    __tablename__ = "research_assessments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    research_input_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_inputs.id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    retrieved_paper_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    comparison_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    novelty_level: Mapped[str] = mapped_column(String, nullable=False, default="not_assessed")
    novelty_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    research_gap_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    research_gap_source: Mapped[str | None] = mapped_column(String, nullable=True)
    candidate_gap_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("candidate_gaps.id"), nullable=True
    )
    potential_applications: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    potential_opportunities: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    technical_feasibility_level: Mapped[str] = mapped_column(String, nullable=False, default="not_assessed")
    technical_feasibility_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    risks_and_limitations: Mapped[str | None] = mapped_column(Text, nullable=True)
    external_validation_needed: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommendation: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[str | None] = mapped_column(String, nullable=True)
    human_reviewed: Mapped[bool] = mapped_column(nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)


class ResearchAssessmentEvidence(Base):
    """Grounds one ResearchAssessment field in a real Evidence row (Sec 2A).

    Mirrors CandidateGapEvidence exactly: this is what makes an
    assessment's comparison_summary inspectable rather than assessment
    text being trusted as its own source of truth.
    """

    __tablename__ = "research_assessment_evidence"
    __table_args__ = (
        UniqueConstraint(
            "research_assessment_id", "evidence_id", "role", name="uq_research_assessment_evidence_assessment_evidence_role"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    research_assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_assessments.id"), nullable=False
    )
    evidence_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("evidence.id"), nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
