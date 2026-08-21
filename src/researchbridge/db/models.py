"""SQLAlchemy models for the Phase 1 (weeks 1-4) ingestion slice."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, Float, ForeignKey, Integer, String, Text, UniqueConstraint
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


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
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
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ExtractionRun(Base):
    __tablename__ = "extraction_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    extractor_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    papers_processed: Mapped[int] = mapped_column(Integer, default=0)
    claims_created: Mapped[int] = mapped_column(Integer, default=0)
    candidates_rejected: Mapped[int] = mapped_column(Integer, default=0)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)


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
    papers_processed: Mapped[int] = mapped_column(Integer, default=0)
    papers_skipped: Mapped[int] = mapped_column(Integer, default=0)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)


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
    """

    __tablename__ = "candidate_gaps"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    seed_paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)
    observation: Mapped[str] = mapped_column(Text, nullable=False)
    gap_type: Mapped[str] = mapped_column(String, nullable=False, default="inference")
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    contributing_paper_count: Mapped[int] = mapped_column(Integer, nullable=False)
    similarity_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    detection_method: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class CandidateGapEvidence(Base):
    """Links one CandidateGap to one contributing paper's Evidence row.

    One row per contributing paper (typically 3+, per gaps/cluster.py's
    minimum cluster size) - this is what makes a candidate gap's support
    inspectable: which papers, which exact quoted passages.
    """

    __tablename__ = "candidate_gap_evidence"
    __table_args__ = (
        UniqueConstraint("candidate_gap_id", "evidence_id", name="uq_candidate_gap_evidence_gap_evidence"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_gap_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("candidate_gaps.id"), nullable=False
    )
    evidence_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("evidence.id"), nullable=False)
