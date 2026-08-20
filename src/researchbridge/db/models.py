"""SQLAlchemy models for the Phase 1 (weeks 1-4) ingestion slice."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import Date, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


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
