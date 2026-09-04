# Full-Text PDF Ingestion Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fetch, parse, and store full-text content for open-access papers, as a new standalone capability with no consumer yet (extraction stays abstract-only until a separate future slice).

**Architecture:** A new `src/researchbridge/fulltext/` package mirrors the existing `extraction/`/`embedding`-pipeline shape exactly: idempotent paper selection, per-paper try/except logging to a dedicated errors table, incremental run-history tracking, a CLI entry point, and one admin-panel trigger route. PDF text extraction reuses the already-tested `researchbridge.benchmark.fulltext.extract_text` (PyMuPDF-based, already proven on the assessment-upload path and the benchmark tooling) rather than re-implementing PDF parsing; this package adds only heading-based section-splitting and DB persistence on top of it.

**Tech Stack:** Python, SQLAlchemy, Alembic, FastAPI, `pymupdf` (already a dependency), `requests` (already a dependency).

**Spec:** `docs/superpowers/specs/2026-09-04-fulltext-pdf-ingestion-design.md`

## Global Constraints

- PyMuPDF only — no GROBID, no OCR, no new external service or Docker dependency.
- Sources covered: arXiv, CORE, Semantic Scholar (reliable fetchable PDF URL when `open_access=True`), plus best-effort Springer (its `url` is an HTML landing page — attempted anyway, parse failures caught and logged, never crash the run).
- Zero changes to `extraction/pipeline.py` or any `Extractor` implementation in this plan — full text has no consumer yet, matching this session's `analysis_claims` precedent.
- Every HTTP fetch uses `timeout=30` (the existing convention in `connectors/springer.py`/`connectors/semantic_scholar.py`).
- One paper's failure must never abort the run — fail loud per-item (logged to `fulltext_fetch_errors`), keep going, matching `ExtractionPipeline`'s contract.
- Idempotent by default (skip papers that already have a `paper_fulltext` row); `--force` re-fetches.

---

### Task 1: Schema — `paper_fulltext`, `fulltext_fetch_runs`, `fulltext_fetch_errors`

**Files:**
- Create: `migrations/versions/0018_paper_fulltext.py`
- Modify: `src/researchbridge/db/models.py` (add three model classes, after `AnalysisClaim`/`ClaimEvidence` and before `ResearchInput`)
- Test: `tests/test_fulltext_models.py`

**Interfaces:**
- Produces: `PaperFullText` (columns: `id`, `paper_id` UNIQUE FK→`papers.id`, `sections` JSONB, `source_url` Text, `fetch_method` String default `"pymupdf"`, `fetched_at` timestamp), `FullTextFetchRun` (columns: `id`, `status` default `"running"`, `started_at`, `finished_at`, `papers_seen`, `papers_fetched`, `papers_skipped_no_url`, `papers_failed`, `error_summary`, `force`), `FullTextFetchError` (columns: `id`, `fulltext_fetch_run_id` FK→`fulltext_fetch_runs.id`, `paper_id` FK→`papers.id`, `error_type`, `error_detail`, `occurred_at`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fulltext_models.py
from __future__ import annotations

import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from researchbridge.db.models import FullTextFetchError, FullTextFetchRun, Paper, PaperFullText


def _paper(session, source_id: str = "p1") -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=source_id, title="A Paper", abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def test_paper_fulltext_persists_sections(session_factory) -> None:
    session = session_factory()
    paper = _paper(session)
    session.commit()

    row = PaperFullText(
        paper_id=paper.id, sections={"introduction": "text"}, source_url="https://example.com/p1.pdf",
    )
    session.add(row)
    session.commit()

    fetched = session.get(PaperFullText, row.id)
    session.close()
    assert fetched.sections == {"introduction": "text"}
    assert fetched.fetch_method == "pymupdf"


def test_paper_fulltext_enforces_one_row_per_paper(session_factory) -> None:
    session = session_factory()
    paper = _paper(session)
    session.add(PaperFullText(paper_id=paper.id, sections={}, source_url="https://example.com/p1.pdf"))
    session.commit()

    session.add(PaperFullText(paper_id=paper.id, sections={}, source_url="https://example.com/p1-again.pdf"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
    session.close()


def test_fulltext_fetch_error_links_to_its_run(session_factory) -> None:
    session = session_factory()
    paper = _paper(session)
    run = FullTextFetchRun(status="running")
    session.add(run)
    session.flush()

    session.add(
        FullTextFetchError(
            fulltext_fetch_run_id=run.id, paper_id=paper.id, error_type="fetch_error", error_detail="boom",
        )
    )
    session.commit()

    errors = session.query(FullTextFetchError).filter_by(fulltext_fetch_run_id=run.id).all()
    session.close()
    assert len(errors) == 1
    assert errors[0].error_type == "fetch_error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fulltext_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'PaperFullText'`

- [ ] **Step 3: Add the model classes**

In `src/researchbridge/db/models.py`, insert after the `ClaimEvidence` class and before `class ResearchInput(Base):`:

```python
class PaperFullText(Base):
    """A paper's full text, fetched from its open-access PDF and split into
    heuristic sections (Sec 46). One row per paper - re-fetching under
    --force updates this row in place rather than inserting a duplicate.

    No consumer yet: extraction/pipeline.py stays abstract-only until a
    separate future slice extends it to use this table - see
    fulltext/pipeline.py's module docstring.
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
```

- [ ] **Step 4: Write the migration**

```python
# migrations/versions/0018_paper_fulltext.py
"""paper_fulltext, fulltext_fetch_runs, fulltext_fetch_errors (Sec 46)

Revision ID: 0018
Revises: 0017
Create Date: 2026-09-04
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0018"
down_revision: Union[str, None] = "0017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "paper_fulltext",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("sections", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("fetch_method", sa.String(), nullable=False, server_default="pymupdf"),
        sa.Column("fetched_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("paper_id", name="uq_paper_fulltext_paper_id"),
    )

    op.create_table(
        "fulltext_fetch_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(), nullable=False, server_default="running"),
        sa.Column("started_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("papers_seen", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("papers_fetched", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("papers_skipped_no_url", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("papers_failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("force", sa.Boolean(), nullable=False, server_default=sa.false()),
    )

    op.create_table(
        "fulltext_fetch_errors",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "fulltext_fetch_run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("fulltext_fetch_runs.id"),
            nullable=False,
        ),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("error_type", sa.String(), nullable=False),
        sa.Column("error_detail", sa.Text(), nullable=False),
        sa.Column("occurred_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_fulltext_fetch_errors_run_id", "fulltext_fetch_errors", ["fulltext_fetch_run_id"])


def downgrade() -> None:
    op.drop_table("fulltext_fetch_errors")
    op.drop_table("fulltext_fetch_runs")
    op.drop_table("paper_fulltext")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fulltext_models.py -v`
Expected: PASS (3 tests). The `session_factory` fixture creates tables via `Base.metadata.create_all`, so the migration file doesn't need to run for this test — but verify it applies cleanly too:

Run: `DATABASE_URL="postgresql+psycopg://researchbridge:researchbridge@localhost:5433/researchbridge_test" .venv/Scripts/python.exe -m alembic upgrade head` against a scratch database (create one with `CREATE DATABASE rb_migration_check`, point `DATABASE_URL` at it, run, then drop it) to confirm `0018` applies and downgrades cleanly, same verification done for `0017` earlier this session.

- [ ] **Step 6: Commit**

```bash
git add migrations/versions/0018_paper_fulltext.py src/researchbridge/db/models.py tests/test_fulltext_models.py
git commit -m "feat(fulltext): add paper_fulltext, fulltext_fetch_runs, fulltext_fetch_errors schema"
```

---

### Task 2: `fulltext/pdf_url.py` — per-source PDF URL resolution

**Files:**
- Create: `src/researchbridge/fulltext/__init__.py` (empty)
- Create: `src/researchbridge/fulltext/pdf_url.py`
- Test: `tests/test_fulltext_pdf_url.py`

**Interfaces:**
- Consumes: `Paper` model (`source`, `source_id`, `url`, `open_access` fields — `db/models.py`), `ARXIV_PDF_URL` constant from `researchbridge.benchmark.fulltext`.
- Produces: `resolve_pdf_url(paper: Paper) -> str | None`, used by Task 4's pipeline.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fulltext_pdf_url.py
from __future__ import annotations

import uuid

from researchbridge.db.models import Paper
from researchbridge.fulltext.pdf_url import resolve_pdf_url


def _paper(source: str, source_id: str, url: str | None, open_access: bool) -> Paper:
    return Paper(
        id=uuid.uuid4(), source=source, source_id=source_id, title="t", abstract="",
        url=url, open_access=open_access, raw_metadata={}, ingestion_metadata={},
    )


def test_returns_none_when_not_open_access() -> None:
    paper = _paper("arxiv", "2401.00001", "https://arxiv.org/abs/2401.00001", open_access=False)
    assert resolve_pdf_url(paper) is None


def test_arxiv_derives_pdf_url_from_source_id() -> None:
    paper = _paper("arxiv", "2401.00001", "https://arxiv.org/abs/2401.00001", open_access=True)
    assert resolve_pdf_url(paper) == "https://arxiv.org/pdf/2401.00001"


def test_core_uses_paper_url_as_is() -> None:
    paper = _paper("core", "123", "https://core.ac.uk/download/123.pdf", open_access=True)
    assert resolve_pdf_url(paper) == "https://core.ac.uk/download/123.pdf"


def test_semantic_scholar_uses_paper_url_as_is() -> None:
    paper = _paper("semantic_scholar", "abc", "https://example.com/openaccess.pdf", open_access=True)
    assert resolve_pdf_url(paper) == "https://example.com/openaccess.pdf"


def test_springer_uses_paper_url_as_is_even_though_its_an_html_page() -> None:
    paper = _paper("springer", "10.1007/xyz", "https://link.springer.com/article/10.1007/xyz", open_access=True)
    assert resolve_pdf_url(paper) == "https://link.springer.com/article/10.1007/xyz"


def test_returns_none_for_an_unknown_source_even_if_open_access() -> None:
    paper = _paper("some_future_source", "1", "https://example.com/x", open_access=True)
    assert resolve_pdf_url(paper) is None


def test_returns_none_when_url_is_missing_for_a_non_arxiv_source() -> None:
    paper = _paper("core", "123", None, open_access=True)
    assert resolve_pdf_url(paper) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fulltext_pdf_url.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.fulltext'`

- [ ] **Step 3: Implement**

```python
# src/researchbridge/fulltext/pdf_url.py
"""Resolves the fetchable PDF URL for one paper, per source (Sec 46).

Only ever called for open_access papers (Task 4's pipeline filters its
selection query on Paper.open_access before calling this) - the
open_access=False check here is a defensive second gate, not the primary
filter.
"""

from __future__ import annotations

from researchbridge.benchmark.fulltext import ARXIV_PDF_URL
from researchbridge.db.models import Paper

# Sources whose `paper.url` is already a genuinely fetchable PDF/full-text
# link (see the design spec's "Per-source PDF URL resolution" table):
# CORE's url is downloadUrl/a fulltext link, Semantic Scholar's is
# openAccessPdf.url (only ever set when open_access is True), Springer's is
# an HTML landing page attempted anyway per the design decision - parsing
# failures are caught and logged downstream, not silently avoided here.
_URL_AS_IS_SOURCES = {"core", "semantic_scholar", "springer"}


def resolve_pdf_url(paper: Paper) -> str | None:
    if not paper.open_access:
        return None
    if paper.source == "arxiv":
        return ARXIV_PDF_URL.format(source_id=paper.source_id)
    if paper.source in _URL_AS_IS_SOURCES:
        return paper.url
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fulltext_pdf_url.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/fulltext/__init__.py src/researchbridge/fulltext/pdf_url.py tests/test_fulltext_pdf_url.py
git commit -m "feat(fulltext): resolve per-source PDF URLs"
```

---

### Task 3: `fulltext/parse.py` — PDF-to-sections parsing

**Files:**
- Create: `src/researchbridge/fulltext/parse.py`
- Test: `tests/test_fulltext_parse.py`

**Interfaces:**
- Consumes: `researchbridge.benchmark.fulltext.extract_text(pdf_bytes: bytes) -> str` (already exists and is already tested — this task does NOT re-test PDF-to-raw-text extraction, only the new section-splitting layered on top of it).
- Produces: `parse_pdf(pdf_bytes: bytes) -> dict[str, str]`, `PdfParseError` exception, `split_sections(text: str) -> dict[str, str]` — used by Task 4's pipeline.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fulltext_parse.py
from __future__ import annotations

import pymupdf
import pytest

from researchbridge.fulltext.parse import PdfParseError, parse_pdf, split_sections


def test_split_sections_splits_on_known_headings() -> None:
    text = (
        "Introduction\n"
        "This paper studies X.\n"
        "Methods\n"
        "We use approach Y.\n"
        "Limitations\n"
        "This only works offline.\n"
    )
    sections = split_sections(text)
    assert sections["introduction"] == "This paper studies X."
    assert sections["methods"] == "We use approach Y."
    assert sections["limitations"] == "This only works offline."


def test_split_sections_recognizes_common_heading_variants() -> None:
    text = "Methodology\nDetails here.\nConclusion and Future Work\nWe conclude.\n"
    sections = split_sections(text)
    assert sections["methods"] == "Details here."
    assert sections["conclusion"] == "We conclude."


def test_split_sections_handles_numbered_headings() -> None:
    text = "1. Introduction\nSome text.\n2. Related Work\nOther text.\n"
    sections = split_sections(text)
    assert sections["introduction"] == "Some text."
    assert sections["related_work"] == "Other text."


def test_split_sections_falls_back_to_body_when_no_heading_matches() -> None:
    text = "Just some prose with no recognizable heading lines at all."
    sections = split_sections(text)
    assert sections == {"body": text}


def test_split_sections_drops_empty_sections() -> None:
    text = "Introduction\n\nMethods\nReal content.\n"
    sections = split_sections(text)
    assert "introduction" not in sections
    assert sections["methods"] == "Real content."


def test_parse_pdf_wraps_extract_text_and_splits_it(monkeypatch) -> None:
    import researchbridge.fulltext.parse as parse_module

    monkeypatch.setattr(parse_module, "extract_text", lambda pdf_bytes: "Introduction\nHello.\n")

    sections = parse_pdf(b"fake-pdf-bytes")
    assert sections == {"introduction": "Hello."}


def test_parse_pdf_raises_pdfparseerror_on_invalid_pdf_bytes() -> None:
    with pytest.raises(PdfParseError):
        parse_pdf(b"not a real pdf")


def test_parse_pdf_raises_pdfparseerror_on_empty_extracted_text(monkeypatch) -> None:
    import researchbridge.fulltext.parse as parse_module

    monkeypatch.setattr(parse_module, "extract_text", lambda pdf_bytes: "   ")

    with pytest.raises(PdfParseError):
        parse_pdf(b"fake-pdf-bytes")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fulltext_parse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.fulltext.parse'`

- [ ] **Step 3: Implement**

```python
# src/researchbridge/fulltext/parse.py
"""Splits a PDF's extracted text into heuristic sections (Sec 46).

Raw text extraction is delegated entirely to
researchbridge.benchmark.fulltext.extract_text (PyMuPDF-based, already
proven on the assessment-upload path and the benchmark tooling) - this
module adds only heading-based section splitting on top of it, so it
never re-implements or re-tests PDF-to-text extraction itself.

The heading vocabulary below is a starting heuristic (see the design
spec's "Open questions" - it wasn't checked against a large sample of real
PDFs before implementation). Revisit if real corpus runs show a common
heading variant being missed.
"""

from __future__ import annotations

import re

import pymupdf

from researchbridge.benchmark.fulltext import extract_text


class PdfParseError(Exception):
    """Raised when pdf_bytes isn't a parseable PDF, or no extractable text
    was found in it (see extract_text's own pymupdf.FileDataError, and the
    empty-text case checked explicitly below)."""


_SECTION_ALIASES: dict[str, str] = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "background": "related_work",
    "method": "methods",
    "methods": "methods",
    "methodology": "methods",
    "approach": "methods",
    "experiments": "experiments",
    "experimental setup": "experiments",
    "experimental results": "results",
    "results": "results",
    "discussion": "discussion",
    "limitations": "limitations",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "conclusion and future work": "conclusion",
    "future work": "conclusion",
    "acknowledgments": "acknowledgments",
    "acknowledgements": "acknowledgments",
    "references": "references",
    "bibliography": "references",
}

# Longest phrases first, so "conclusion and future work" matches whole
# before the shorter "conclusion" alias gets a chance to.
_HEADING_LINE_RE = re.compile(
    r"^\s*(?:[0-9]+\.?|[IVXLC]+\.?|[A-Z]\.)?\s*("
    + "|".join(re.escape(k) for k in sorted(_SECTION_ALIASES, key=len, reverse=True))
    + r")\s*$",
    re.IGNORECASE,
)


def split_sections(text: str) -> dict[str, str]:
    lines = text.split("\n")
    sections: dict[str, list[str]] = {"body": []}
    current = "body"
    matched_any = False

    for line in lines:
        match = _HEADING_LINE_RE.match(line)
        if match:
            current = _SECTION_ALIASES[match.group(1).lower()]
            sections.setdefault(current, [])
            matched_any = True
            continue
        sections[current].append(line)

    if not matched_any:
        return {"body": text.strip()}

    result = {name: "\n".join(section_lines).strip() for name, section_lines in sections.items()}
    return {name: content for name, content in result.items() if content}


def parse_pdf(pdf_bytes: bytes) -> dict[str, str]:
    try:
        text = extract_text(pdf_bytes)
    except pymupdf.FileDataError as exc:
        raise PdfParseError(f"not a valid PDF: {exc}") from exc

    if not text.strip():
        raise PdfParseError("no extractable text found in PDF")

    return split_sections(text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fulltext_parse.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/fulltext/parse.py tests/test_fulltext_parse.py
git commit -m "feat(fulltext): split extracted PDF text into heuristic sections"
```

---

### Task 4: `fulltext/pipeline.py` — fetch + parse + persist orchestration

**Files:**
- Create: `src/researchbridge/fulltext/pipeline.py`
- Test: `tests/test_fulltext_pipeline.py`

**Interfaces:**
- Consumes: `resolve_pdf_url` (Task 2), `parse_pdf`/`PdfParseError` (Task 3), `PaperFullText`/`FullTextFetchRun`/`FullTextFetchError`/`Paper` (Task 1).
- Produces: `class FullTextFetchPipeline: def __init__(self, session_factory: sessionmaker[Session]) -> None; def run(self, limit: int | None = None, force: bool = False) -> str` — returns the run id as a string, matching `ExtractionPipeline.run()`'s return shape. Used by Task 5's CLI.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fulltext_pipeline.py
from __future__ import annotations

import uuid
from unittest.mock import Mock

import pytest
import requests

from researchbridge.db.models import FullTextFetchError, FullTextFetchRun, Paper, PaperFullText
from researchbridge.fulltext.parse import PdfParseError
from researchbridge.fulltext.pipeline import FullTextFetchPipeline


def _paper(session, source: str = "arxiv", source_id: str = "2401.00001", open_access: bool = True) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source=source, source_id=source_id, title="t", abstract="",
        url=f"https://arxiv.org/abs/{source_id}", open_access=open_access,
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.commit()
    return paper


def test_fetches_and_persists_fulltext_for_an_open_access_paper(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    paper = _paper(session)
    session.close()

    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda pdf_bytes: {"introduction": "hello"})
    monkeypatch.setattr(
        pipeline_module.requests, "get", Mock(return_value=Mock(content=b"pdf-bytes", raise_for_status=Mock()))
    )

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(FullTextFetchRun, run_id)
        assert run.status == "completed"
        assert run.papers_seen == 1
        assert run.papers_fetched == 1
        assert run.papers_failed == 0

        row = session.query(PaperFullText).filter_by(paper_id=paper.id).one()
        assert row.sections == {"introduction": "hello"}
        assert row.source_url == "https://arxiv.org/pdf/2401.00001"
    finally:
        session.close()


def test_non_open_access_papers_are_never_selected(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    _paper(session, open_access=False)
    session.close()

    mock_get = Mock()
    monkeypatch.setattr(pipeline_module.requests, "get", mock_get)

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    run = session.get(FullTextFetchRun, run_id)
    session.close()
    assert run.papers_seen == 0
    mock_get.assert_not_called()


def test_fetch_failure_is_logged_and_run_continues(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    paper = _paper(session)
    session.close()

    monkeypatch.setattr(
        pipeline_module.requests, "get", Mock(side_effect=requests.ConnectionError("connection refused"))
    )

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(FullTextFetchRun, run_id)
        assert run.status == "completed"
        assert run.papers_failed == 1
        assert run.papers_fetched == 0

        errors = session.query(FullTextFetchError).filter_by(paper_id=paper.id).all()
        assert len(errors) == 1
        assert errors[0].error_type == "fetch_error"
        assert session.query(PaperFullText).filter_by(paper_id=paper.id).one_or_none() is None
    finally:
        session.close()


def test_parse_failure_is_logged_and_run_continues(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    paper = _paper(session)
    session.close()

    monkeypatch.setattr(
        pipeline_module, "parse_pdf", Mock(side_effect=PdfParseError("not a valid PDF"))
    )
    monkeypatch.setattr(
        pipeline_module.requests, "get", Mock(return_value=Mock(content=b"junk", raise_for_status=Mock()))
    )

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(FullTextFetchRun, run_id)
        assert run.papers_failed == 1
        errors = session.query(FullTextFetchError).filter_by(paper_id=paper.id).all()
        assert errors[0].error_type == "parse_error"
    finally:
        session.close()


def test_idempotent_skips_already_fetched_paper(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    paper = _paper(session)
    session.close()

    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda pdf_bytes: {"body": "hello"})
    monkeypatch.setattr(
        pipeline_module.requests, "get", Mock(return_value=Mock(content=b"pdf-bytes", raise_for_status=Mock()))
    )

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    pipeline.run()
    second_run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(FullTextFetchRun, second_run_id)
        assert run.papers_seen == 0  # already has a paper_fulltext row
        assert len(session.query(PaperFullText).filter_by(paper_id=paper.id).all()) == 1  # not duplicated
    finally:
        session.close()


def test_force_refetches_an_already_fetched_paper(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    paper = _paper(session)
    session.close()

    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda pdf_bytes: {"body": "first"})
    monkeypatch.setattr(
        pipeline_module.requests, "get", Mock(return_value=Mock(content=b"pdf-bytes", raise_for_status=Mock()))
    )
    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    pipeline.run()

    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda pdf_bytes: {"body": "second"})
    second_run_id = pipeline.run(force=True)

    session = session_factory()
    try:
        run = session.get(FullTextFetchRun, second_run_id)
        assert run.papers_seen == 1
        assert run.force is True
        row = session.query(PaperFullText).filter_by(paper_id=paper.id).one()
        assert row.sections == {"body": "second"}  # updated in place, not duplicated
    finally:
        session.close()


def test_limit_caps_papers_processed(session_factory, monkeypatch) -> None:
    import researchbridge.fulltext.pipeline as pipeline_module

    session = session_factory()
    _paper(session, source_id="2401.00001")
    _paper(session, source_id="2401.00002")
    session.close()

    monkeypatch.setattr(pipeline_module, "parse_pdf", lambda pdf_bytes: {"body": "hello"})
    monkeypatch.setattr(
        pipeline_module.requests, "get", Mock(return_value=Mock(content=b"pdf-bytes", raise_for_status=Mock()))
    )

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    run_id = pipeline.run(limit=1)

    session = session_factory()
    run = session.get(FullTextFetchRun, run_id)
    session.close()
    assert run.papers_seen == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fulltext_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.fulltext.pipeline'`

- [ ] **Step 3: Implement**

```python
# src/researchbridge/fulltext/pipeline.py
"""Fetch + parse + persist full text for open-access papers (Sec 46).

Structurally mirrors extraction/pipeline.py: idempotent paper selection,
per-paper try/except that logs to fulltext_fetch_errors and continues
rather than crashing the run, incremental run-history updates.

No consumer yet - extraction/pipeline.py and every Extractor implementation
stay abstract-only. Wiring full text into extraction is a separate, future
slice (see the design spec's "Two-slice split").
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from researchbridge.db.models import FullTextFetchError, FullTextFetchRun, Paper, PaperFullText
from researchbridge.fulltext.parse import PdfParseError, parse_pdf
from researchbridge.fulltext.pdf_url import resolve_pdf_url

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 30


class FullTextFetchPipeline:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def run(self, limit: int | None = None, force: bool = False) -> str:
        session = self.session_factory()
        run = FullTextFetchRun(status="running", force=force)
        session.add(run)
        session.commit()
        run_id = str(run.id)

        try:
            papers = self._select_papers(session, limit, force)
            total = len(papers)
            logger.info("Full-text fetch run %s starting: %d paper(s) to process", run_id, total)

            for i, paper in enumerate(papers, start=1):
                self._process_paper(session, run, paper, force)
                run.papers_seen += 1
                session.commit()
                if i % 10 == 0 or i == total:
                    logger.info(
                        "Full-text fetch run %s: %d/%d papers seen (fetched=%d, failed=%d)",
                        run_id, i, total, run.papers_fetched, run.papers_failed,
                    )

            run.status = "completed"
            run.finished_at = datetime.now(UTC)
            session.commit()
            logger.info(
                "Full-text fetch run %s completed: %d fetched, %d failed", run_id, run.papers_fetched, run.papers_failed
            )

        except Exception as exc:  # noqa: BLE001 - run must record failure, not crash silently
            logger.exception("Full-text fetch run %s failed", run_id)
            run.status = "failed"
            run.error_summary = str(exc)[:2000]
            run.finished_at = datetime.now(UTC)
            session.commit()
            raise
        finally:
            session.close()

        return run_id

    def _process_paper(self, session: Session, run: FullTextFetchRun, paper: Paper, force: bool) -> None:
        url = resolve_pdf_url(paper)
        if url is None:
            run.papers_skipped_no_url += 1
            return

        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Full-text fetch failed for paper %s: %s", paper.id, exc)
            self._record_error(session, run.id, paper.id, "fetch_error", str(exc)[:2000])
            run.papers_failed += 1
            return

        try:
            sections = parse_pdf(response.content)
        except PdfParseError as exc:
            logger.warning("Full-text parse failed for paper %s: %s", paper.id, exc)
            self._record_error(session, run.id, paper.id, "parse_error", str(exc)[:2000])
            run.papers_failed += 1
            return

        existing = session.execute(select(PaperFullText).where(PaperFullText.paper_id == paper.id)).scalar_one_or_none()
        if existing is not None:
            existing.sections = sections
            existing.source_url = url
        else:
            session.add(PaperFullText(paper_id=paper.id, sections=sections, source_url=url))
        run.papers_fetched += 1

    def _select_papers(self, session: Session, limit: int | None, force: bool) -> list[Paper]:
        query = select(Paper).where(Paper.open_access.is_(True))
        if not force:
            already_fetched = select(PaperFullText.paper_id)
            query = query.where(Paper.id.notin_(already_fetched))
        if limit is not None:
            query = query.limit(limit)
        return list(session.execute(query).scalars())

    def _record_error(self, session: Session, run_id: Any, paper_id: Any, error_type: str, detail: str) -> None:
        session.add(
            FullTextFetchError(
                fulltext_fetch_run_id=run_id, paper_id=paper_id, error_type=error_type, error_detail=detail,
            )
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fulltext_pipeline.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/fulltext/pipeline.py tests/test_fulltext_pipeline.py
git commit -m "feat(fulltext): add FullTextFetchPipeline orchestration"
```

---

### Task 5: CLI — `rb-fulltext-fetch`

**Files:**
- Create: `src/researchbridge/fulltext/cli.py`
- Modify: `pyproject.toml` (add script entry, alongside `rb-extract`/`rb-embed`)
- Test: `tests/test_fulltext_cli.py`

**Interfaces:**
- Consumes: `FullTextFetchPipeline` (Task 4), `researchbridge.config.load_config`, `researchbridge.db.session.make_engine`/`make_session_factory` (existing).
- Produces: `main() -> None`, `build_parser() -> argparse.ArgumentParser` — used by Task 6's admin trigger route (invoked as a subprocess module, same as every other pipeline CLI).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fulltext_cli.py
from __future__ import annotations

from unittest.mock import Mock

from researchbridge.fulltext.cli import build_parser, main


def test_parser_accepts_limit_and_force() -> None:
    parser = build_parser()
    args = parser.parse_args(["--limit", "5", "--force"])
    assert args.limit == 5
    assert args.force is True


def test_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.limit is None
    assert args.force is False


def test_main_runs_the_pipeline_with_parsed_args(monkeypatch, capsys) -> None:
    import researchbridge.fulltext.cli as cli_module

    mock_pipeline = Mock()
    mock_pipeline.run.return_value = "run-123"
    monkeypatch.setattr(cli_module, "FullTextFetchPipeline", Mock(return_value=mock_pipeline))
    monkeypatch.setattr(cli_module, "make_engine", Mock())
    monkeypatch.setattr(cli_module, "make_session_factory", Mock())
    monkeypatch.setattr(cli_module, "load_config", Mock())
    monkeypatch.setattr("sys.argv", ["rb-fulltext-fetch", "--limit", "3"])

    main()

    mock_pipeline.run.assert_called_once_with(limit=3, force=False)
    assert "run-123" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fulltext_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.fulltext.cli'`

- [ ] **Step 3: Implement**

```python
# src/researchbridge/fulltext/cli.py
from __future__ import annotations

import argparse
import logging

from researchbridge.config import load_config
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.fulltext.pipeline import FullTextFetchPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and parse full text for open-access papers.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of papers to process")
    parser.add_argument("--force", action="store_true", help="Re-fetch papers that already have full text")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_config()

    parser = build_parser()
    args = parser.parse_args()

    engine = make_engine()
    session_factory = make_session_factory(engine)

    pipeline = FullTextFetchPipeline(session_factory=session_factory)
    run_id = pipeline.run(limit=args.limit, force=args.force)
    print(f"Full-text fetch run {run_id} finished.")


if __name__ == "__main__":
    main()
```

Then add to `pyproject.toml`, in the `[project.scripts]` section next to `rb-extract`:

```toml
rb-fulltext-fetch = "researchbridge.fulltext.cli:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_fulltext_cli.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/fulltext/cli.py pyproject.toml tests/test_fulltext_cli.py
git commit -m "feat(fulltext): add rb-fulltext-fetch CLI"
```

---

### Task 6: Admin panel trigger route

**Files:**
- Modify: `src/researchbridge/api/schemas.py` (add `FullTextFetchTrigger`, next to `EmbeddingTrigger`)
- Modify: `src/researchbridge/api/admin_routes.py` (`PIPELINE_KEYS`, `RUN_MODEL_BY_KEY`, new trigger route, new import)
- Test: `tests/test_admin_api.py` (append tests, following the `test_trigger_embedding_*` pattern)

**Interfaces:**
- Consumes: `FullTextFetchRun` (Task 1), `_trigger_or_409` (existing helper in `admin_routes.py`).
- Produces: `POST /api/admin/fulltext/run` (payload: `{"limit": int | None, "force": bool}`, response: `PipelineTriggerOut`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_admin_api.py`, near the existing `test_trigger_embedding_*` tests:

```python
def test_trigger_fulltext_passes_overrides_as_flags(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post("/api/admin/fulltext/run", json={"limit": 25})

    assert calls == [("fulltext", "researchbridge.fulltext.cli", ["--limit", "25"])]


def test_trigger_fulltext_with_force_passes_force_flag(client, monkeypatch) -> None:
    import researchbridge.api.admin_routes as routes_module

    calls = []
    monkeypatch.setattr(
        routes_module, "trigger", lambda key, module, args: calls.append((key, module, args)) or Path("x.log")
    )

    client.post("/api/admin/fulltext/run", json={"limit": 5, "force": True})

    assert calls == [("fulltext", "researchbridge.fulltext.cli", ["--limit", "5", "--force"])]


def test_fulltext_is_in_pipeline_status(client) -> None:
    body = client.get("/api/admin/pipeline").json()
    assert "fulltext" in body["running"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_admin_api.py -k fulltext -v`
Expected: FAIL with 404 (no `/api/admin/fulltext/run` route yet)

- [ ] **Step 3: Implement**

In `src/researchbridge/api/schemas.py`, add next to `EmbeddingTrigger`:

```python
class FullTextFetchTrigger(BaseModel):
    limit: int | None = None
    force: bool = False
```

In `src/researchbridge/api/admin_routes.py`:

1. Add `FullTextFetchRun` to the `from researchbridge.db.models import (...)` block.
2. Add `FullTextFetchTrigger` to the `from researchbridge.api.schemas import (...)` block.
3. Add `"fulltext"` to `PIPELINE_KEYS`, alongside `"gaps"`.
4. Add `"fulltext": (FullTextFetchRun, None),` to `RUN_MODEL_BY_KEY`, alongside `"gaps"`.
5. Add the route, next to `trigger_embedding`:

```python
@router.post("/fulltext/run", response_model=PipelineTriggerOut)
def trigger_fulltext(payload: FullTextFetchTrigger) -> PipelineTriggerOut:
    args: list[str] = []
    if payload.limit is not None:
        args += ["--limit", str(payload.limit)]
    if payload.force:
        args += ["--force"]
    return _trigger_or_409("fulltext", "researchbridge.fulltext.cli", args)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_admin_api.py -k fulltext -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: all tests pass (existing suite + all fulltext tests from Tasks 1-6)

- [ ] **Step 6: Commit**

```bash
git add src/researchbridge/api/schemas.py src/researchbridge/api/admin_routes.py tests/test_admin_api.py
git commit -m "feat(fulltext): add admin panel trigger route"
```

---

## Explicitly not in this plan

- Any change to `extraction/pipeline.py` or `HeuristicExtractor`/`HybridExtractor`/`SemanticExtractor` — full text has no consumer after this plan ships, matching the design spec's two-slice split.
- A dedicated Springer PDF-URL resolver — Springer is attempted with its existing landing-page `url` and expected to often fail parsing; a real resolver is separate future work if the failure rate turns out to matter.
- Frontend/admin-panel UI for triggering this pipeline by button (only the backend route is added in Task 6) — matches how several other pipelines (e.g. `retrieval_eval`) are triggerable via API without dedicated UI polish in the same PR that added them; add a button in `frontend/app/admin/page.tsx` as a follow-up if this pipeline needs to be run manually more than occasionally.
