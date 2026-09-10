# Full-text retrieval for "Ask the corpus" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Widen "ask the corpus" retrieval to use each open-access paper's full text, not just its title/abstract and pre-extracted claims.

**Architecture:** A new `PaperFullTextChunk` table stores paragraph-level embeddings split from the existing `PaperFullText.sections`, populated by a new batch pipeline (mirrors `embedding/pipeline.py`'s shape). Stage 1 paper selection (`embedding/search.py`) takes the best of a paper's abstract-distance and its best chunk-distance. Stage 2 quote selection (`qa/answer.py`) merges `ExtractedClaim` quotes with full-text paragraphs into one ranked pool, tagging each hit `source: "claim" | "passage"` and dropping any passage that already contains a candidate claim's exact text (claims are always verbatim substrings of some paragraph).

**Tech Stack:** Python 3.12, SQLAlchemy 2.0, PostgreSQL + pgvector, FastAPI, pytest, Next.js/TypeScript frontend.

**Spec:** `docs/superpowers/specs/2026-09-10-fulltext-retrieval-design.md`

## Global Constraints

- Same embedding model everywhere: `all-MiniLM-L6-v2`, 384-dim, L2-normalized vectors (`EMBEDDING_DIM = 384` in `db/models.py`).
- No ANN index for the new table — sequential scan is the accepted tradeoff at current corpus scale, same as the existing `embeddings` table (deferred to a later sub-project).
- Papers without a `PaperFullText` row (non-open-access, or not yet fetched) must behave exactly as they do today — no regression to existing tests.
- `extraction_method == "stub"` evidence stays excluded from claim candidates (existing rule, unchanged).
- `Paper.excluded_at IS NOT NULL` papers stay excluded from all retrieval (existing rule, unchanged).
- Additive-only API/schema changes: existing `QuoteHitOut` consumers (saved `QaQuestion.hits` JSON rows already in the DB, which predate this change and have no `source` key) must still parse via `QuoteHitOut.model_validate(...)` — the new `source` field needs a default.

**Note on one deliberate refinement over the committed spec:** the spec's schema sketch for `PaperFullTextChunk` omits a `model_name` column. This plan adds one, mirroring `Embedding.model_name`'s documented reasoning (`db/models.py:283-286`: lets more than one embedding model's rows coexist without a migration if the model is ever swapped) — the existing `embeddings` table already follows this convention and there's no cost to matching it. Also, the spec describes the chunk pipeline running "incrementally whenever `FullTextFetchPipeline` persists a new/updated row" — in practice, no pipeline in this codebase is event-triggered; they're all idempotent batch/CLI jobs (`rb-embed`, `rb-fulltext-fetch`), some re-run via a `--force` flag to pick up updates. This plan follows that exact convention instead of inventing a new trigger mechanism.

---

### Task 1: `PaperFullTextChunk` model and migration

**Files:**
- Modify: `src/researchbridge/db/models.py` (add class after `PaperFullText`, ~line 530)
- Create: `migrations/versions/0027_paper_fulltext_chunk.py`
- Test: `tests/test_fulltext_models.py` (add a test to the existing file)

**Interfaces:**
- Produces: `PaperFullTextChunk` model with columns `id`, `paper_id`, `section: str`, `paragraph_index: int`, `text: str`, `model_name: str`, `embedding: list[float]` (`Vector(384)`), `created_at`. Table name `paper_fulltext_chunk`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_fulltext_models.py`:

```python
from researchbridge.db.models import PaperFullTextChunk


def test_paper_fulltext_chunk_persists_paragraph(session_factory) -> None:
    session = session_factory()
    paper = _paper(session)
    session.commit()

    vector = [0.1] * 384
    chunk = PaperFullTextChunk(
        paper_id=paper.id,
        section="introduction",
        paragraph_index=0,
        text="A full paragraph of introduction text.",
        model_name="fake-embedder-v1",
        embedding=vector,
    )
    session.add(chunk)
    session.commit()

    fetched = session.get(PaperFullTextChunk, chunk.id)
    session.close()
    assert fetched.section == "introduction"
    assert fetched.paragraph_index == 0
    assert fetched.text == "A full paragraph of introduction text."
    assert fetched.model_name == "fake-embedder-v1"
    assert len(fetched.embedding) == 384
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fulltext_models.py::test_paper_fulltext_chunk_persists_paragraph -v`
Expected: FAIL with `ImportError: cannot import name 'PaperFullTextChunk'`

- [ ] **Step 3: Add the model**

In `src/researchbridge/db/models.py`, insert after the `PaperFullText` class (after line 529, before `class FullTextFetchRun`):

```python
class PaperFullTextChunk(Base):
    """One paragraph-level chunk of a paper's full text, embedded for
    retrieval (full-text retrieval follow-on to Sec 46 / the corpus-qa
    design). Split from PaperFullText.sections by
    fulltext/chunking.py::split_paragraphs; populated by
    fulltext/chunk_pipeline.py.

    `model_name` mirrors Embedding.model_name's reasoning: lets more than
    one embedding model's chunks coexist without a schema change if the
    model in use is ever swapped.
    """

    __tablename__ = "paper_fulltext_chunk"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("papers.id"), nullable=False)
    section: Mapped[str] = mapped_column(String, nullable=False)
    paragraph_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fulltext_models.py::test_paper_fulltext_chunk_persists_paragraph -v`
Expected: PASS

- [ ] **Step 5: Write the migration**

Create `migrations/versions/0027_paper_fulltext_chunk.py`:

```python
"""paper_fulltext_chunk (full-text retrieval for ask-the-corpus)

Revision ID: 0027
Revises: 0026
Create Date: 2026-09-10
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0027"
down_revision: Union[str, None] = "0026"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


def upgrade() -> None:
    op.create_table(
        "paper_fulltext_chunk",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id"), nullable=False),
        sa.Column("section", sa.String(), nullable=False),
        sa.Column("paragraph_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_paper_fulltext_chunk_paper_id", "paper_fulltext_chunk", ["paper_id"])
    op.create_index("ix_paper_fulltext_chunk_model_name", "paper_fulltext_chunk", ["model_name"])


def downgrade() -> None:
    op.drop_table("paper_fulltext_chunk")
```

- [ ] **Step 6: Apply the migration against the dev database**

Run: `alembic upgrade head`
Expected: applies `0027` cleanly, no errors.

- [ ] **Step 7: Commit**

```bash
git add src/researchbridge/db/models.py migrations/versions/0027_paper_fulltext_chunk.py tests/test_fulltext_models.py
git commit -m "feat(fulltext): add paper_fulltext_chunk table for paragraph-level embeddings"
```

---

### Task 2: Paragraph splitting

**Files:**
- Create: `src/researchbridge/fulltext/chunking.py`
- Test: `tests/test_fulltext_chunking.py`

**Interfaces:**
- Produces: `split_paragraphs(section_text: str, min_words: int = 10) -> list[str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fulltext_chunking.py`:

```python
from __future__ import annotations

from researchbridge.fulltext.chunking import split_paragraphs


def test_splits_on_blank_lines():
    text = "First paragraph with enough words to stand alone here.\n\nSecond paragraph also has plenty of words in it."
    result = split_paragraphs(text)
    assert result == [
        "First paragraph with enough words to stand alone here.",
        "Second paragraph also has plenty of words in it.",
    ]


def test_merges_short_leading_paragraph_into_the_next_one():
    text = "Fig. 1.\n\nThis is a real paragraph with more than ten words describing the figure above."
    result = split_paragraphs(text)
    assert result == [
        "Fig. 1. This is a real paragraph with more than ten words describing the figure above."
    ]


def test_merges_short_trailing_paragraph_into_the_previous_one():
    text = "This is a real paragraph with more than ten words describing something important.\n\nThe end."
    result = split_paragraphs(text)
    assert result == [
        "This is a real paragraph with more than ten words describing something important. The end."
    ]


def test_lone_short_paragraph_is_kept_as_its_own_chunk():
    result = split_paragraphs("Too short.")
    assert result == ["Too short."]


def test_empty_section_returns_no_paragraphs():
    assert split_paragraphs("") == []
    assert split_paragraphs("   \n\n  ") == []


def test_collapses_runs_of_multiple_blank_lines():
    text = "Paragraph one has enough words to count on its own merits.\n\n\n\nParagraph two also has enough words to count."
    result = split_paragraphs(text)
    assert len(result) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fulltext_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.fulltext.chunking'`

- [ ] **Step 3: Implement `split_paragraphs`**

Create `src/researchbridge/fulltext/chunking.py`:

```python
"""Splits one PaperFullText section's text into paragraph-level chunks.

fulltext/parse.py's split_sections() preserves line structure within each
section (see its own docstring), and PDF extraction (benchmark/fulltext.py's
_tidy()) collapses runs of blank lines down to exactly one blank line
between paragraphs - so splitting a section's text on blank-line boundaries
recovers real paragraph breaks.

A paragraph under `min_words` words (stray figure/table captions, single
heading fragments that split_sections() didn't recognize as a heading) is
merged into an adjacent paragraph rather than becoming its own chunk, so
answer_question() never surfaces a near-empty "quote".
"""

from __future__ import annotations

import re

_BLANK_LINE_RE = re.compile(r"\n\s*\n+")


def split_paragraphs(section_text: str, min_words: int = 10) -> list[str]:
    raw = _BLANK_LINE_RE.split(section_text.strip())
    paragraphs = [p.strip() for p in raw if p.strip()]
    if not paragraphs:
        return []

    merged: list[str] = []
    buffer = ""
    for para in paragraphs:
        candidate = f"{buffer} {para}".strip() if buffer else para
        if len(candidate.split()) < min_words:
            buffer = candidate
            continue
        merged.append(candidate)
        buffer = ""

    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]} {buffer}".strip()
        else:
            merged.append(buffer)

    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fulltext_chunking.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/fulltext/chunking.py tests/test_fulltext_chunking.py
git commit -m "feat(fulltext): add paragraph-splitting for full-text chunking"
```

---

### Task 3: Chunk embedding pipeline + CLI

**Files:**
- Create: `src/researchbridge/fulltext/chunk_pipeline.py`
- Create: `src/researchbridge/fulltext/cli_chunk.py`
- Modify: `pyproject.toml` (add `rb-fulltext-chunk` script entry, after the `rb-fulltext-fetch` line)
- Test: `tests/test_fulltext_chunk_pipeline.py`

**Interfaces:**
- Consumes: `PaperFullText` (`db/models.py`), `PaperFullTextChunk` (Task 1), `split_paragraphs` (Task 2), `Embedder` protocol (`embedding/base.py`: `model_name: str`, `embed_texts(texts: list[str]) -> list[list[float]]`).
- Produces: `FullTextChunkPipeline(embedder, session_factory).run(limit=None) -> dict[str, int]` (keys `"papers_processed"`, `"chunks_created"`); `reset_chunk_data(session, model_name) -> int`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fulltext_chunk_pipeline.py`:

```python
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select

from researchbridge.db.models import EMBEDDING_DIM, Paper, PaperFullText, PaperFullTextChunk
from researchbridge.fulltext.chunk_pipeline import FullTextChunkPipeline, reset_chunk_data


def _hash_to_unit_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


@dataclass
class FakeEmbedder:
    model_name: str = "fake-embedder-v1"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_hash_to_unit_vector(t) for t in texts]


def _make_paper_with_fulltext(session, source_id: str, sections: dict[str, str]) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=source_id, title="A Paper", abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    session.add(PaperFullText(paper_id=paper.id, sections=sections, source_url="https://example.com/p.pdf"))
    session.commit()
    return paper


def test_chunks_and_embeds_every_paragraph(session_factory) -> None:
    session = session_factory()
    paper = _make_paper_with_fulltext(
        session,
        "p1",
        {
            "introduction": (
                "This is the first real paragraph of the introduction with enough words to count.\n\n"
                "This is the second real paragraph of the introduction, also long enough to count."
            ),
            "methods": "This methods paragraph has more than ten words describing the approach used here.",
        },
    )
    session.close()

    embedder = FakeEmbedder()
    pipeline = FullTextChunkPipeline(embedder=embedder, session_factory=session_factory)
    result = pipeline.run()

    assert result == {"papers_processed": 1, "chunks_created": 3}

    session = session_factory()
    chunks = list(session.execute(select(PaperFullTextChunk).where(PaperFullTextChunk.paper_id == paper.id)).scalars())
    session.close()
    assert len(chunks) == 3
    assert {c.section for c in chunks} == {"introduction", "methods"}
    intro_chunks = sorted([c for c in chunks if c.section == "introduction"], key=lambda c: c.paragraph_index)
    assert intro_chunks[0].paragraph_index == 0
    assert intro_chunks[1].paragraph_index == 1
    assert all(len(c.embedding) == EMBEDDING_DIM for c in chunks)
    assert all(c.model_name == "fake-embedder-v1" for c in chunks)


def test_is_idempotent_on_rerun(session_factory) -> None:
    session = session_factory()
    _make_paper_with_fulltext(session, "p1", {"introduction": "One paragraph with more than ten words in this section."})
    session.close()

    embedder = FakeEmbedder()
    pipeline = FullTextChunkPipeline(embedder=embedder, session_factory=session_factory)
    pipeline.run()
    second_result = pipeline.run()

    assert second_result == {"papers_processed": 0, "chunks_created": 0}


def test_skips_papers_with_no_fulltext_row(session_factory) -> None:
    session = session_factory()
    session.add(Paper(
        id=uuid.uuid4(), source="arxiv", source_id="p1", title="No fulltext", abstract="",
        raw_metadata={}, ingestion_metadata={},
    ))
    session.commit()
    session.close()

    embedder = FakeEmbedder()
    pipeline = FullTextChunkPipeline(embedder=embedder, session_factory=session_factory)
    result = pipeline.run()

    assert result == {"papers_processed": 0, "chunks_created": 0}


def test_reset_chunk_data_deletes_only_that_model(session_factory) -> None:
    session = session_factory()
    paper = _make_paper_with_fulltext(session, "p1", {"introduction": "One paragraph with more than ten words here."})
    session.close()

    FullTextChunkPipeline(embedder=FakeEmbedder(model_name="model-a"), session_factory=session_factory).run()
    FullTextChunkPipeline(embedder=FakeEmbedder(model_name="model-b"), session_factory=session_factory).run()

    session = session_factory()
    deleted = reset_chunk_data(session, "model-a")
    remaining = list(session.execute(select(PaperFullTextChunk).where(PaperFullTextChunk.paper_id == paper.id)).scalars())
    session.close()

    assert deleted == 1
    assert len(remaining) == 1
    assert remaining[0].model_name == "model-b"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fulltext_chunk_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'researchbridge.fulltext.chunk_pipeline'`

- [ ] **Step 3: Implement the pipeline**

Create `src/researchbridge/fulltext/chunk_pipeline.py`:

```python
"""Chunk each PaperFullText row into paragraphs and embed them.

Mirrors embedding/pipeline.py's shape and idempotency convention: selects
PaperFullText rows with no PaperFullTextChunk rows yet for this model, splits
each section into paragraphs (fulltext/chunking.py), embeds them in batches,
persists one row per paragraph. Re-running only processes newly-fetched full
text; to pick up a full-text re-fetch (paper.py updated via
FullTextFetchPipeline's --force), re-run this pipeline's own --force, which
deletes and rebuilds every chunk for the current model (same pattern as
embedding/pipeline.py's reset_embedding_data + cli_embed.py's --force).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from researchbridge.db.models import PaperFullText, PaperFullTextChunk
from researchbridge.embedding.base import Embedder
from researchbridge.fulltext.chunking import split_paragraphs

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


def reset_chunk_data(session: Session, model_name: str) -> int:
    result = session.execute(delete(PaperFullTextChunk).where(PaperFullTextChunk.model_name == model_name))
    session.commit()
    return result.rowcount


class FullTextChunkPipeline:
    def __init__(self, embedder: Embedder, session_factory: sessionmaker[Session]) -> None:
        self.embedder = embedder
        self.session_factory = session_factory

    def run(self, limit: int | None = None) -> dict[str, int]:
        session = self.session_factory()
        papers_processed = 0
        chunks_created = 0
        try:
            fulltexts = self._select_unchunked(session, limit)
            logger.info(
                "Full-text chunk run starting: %d paper(s) to chunk (model=%s)",
                len(fulltexts), self.embedder.model_name,
            )

            for i, fulltext in enumerate(fulltexts, start=1):
                paragraphs: list[tuple[str, int, str]] = []
                for section, text in fulltext.sections.items():
                    for idx, para in enumerate(split_paragraphs(text)):
                        paragraphs.append((section, idx, para))

                for batch in _chunks(paragraphs, BATCH_SIZE):
                    vectors = self.embedder.embed_texts([p[2] for p in batch])
                    for (section, idx, text), vector in zip(batch, vectors, strict=True):
                        session.add(
                            PaperFullTextChunk(
                                paper_id=fulltext.paper_id,
                                section=section,
                                paragraph_index=idx,
                                text=text,
                                model_name=self.embedder.model_name,
                                embedding=vector,
                            )
                        )
                        chunks_created += 1

                papers_processed += 1
                session.commit()
                if i % 10 == 0 or i == len(fulltexts):
                    logger.info(
                        "Full-text chunk run: %d/%d papers processed (chunks_created=%d)",
                        i, len(fulltexts), chunks_created,
                    )

            logger.info(
                "Full-text chunk run completed: %d papers processed, %d chunks created",
                papers_processed, chunks_created,
            )
        finally:
            session.close()

        return {"papers_processed": papers_processed, "chunks_created": chunks_created}

    def _select_unchunked(self, session: Session, limit: int | None) -> list[PaperFullText]:
        already_chunked = select(PaperFullTextChunk.paper_id).where(
            PaperFullTextChunk.model_name == self.embedder.model_name
        )
        query = select(PaperFullText).where(PaperFullText.paper_id.notin_(already_chunked))
        if limit is not None:
            query = query.limit(limit)
        return list(session.execute(query).scalars())


def _chunks(items: list[tuple[str, int, str]], size: int) -> Iterator[list[tuple[str, int, str]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fulltext_chunk_pipeline.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Add the CLI entry point**

Create `src/researchbridge/fulltext/cli_chunk.py`:

```python
from __future__ import annotations

import argparse
import logging

from researchbridge.config import load_config
from researchbridge.db.session import make_engine, make_session_factory
from researchbridge.embedding.model import SentenceTransformerEmbedder
from researchbridge.fulltext.chunk_pipeline import FullTextChunkPipeline, reset_chunk_data
from researchbridge.pipeline_logging import configure_pipeline_logging


def main() -> None:
    configure_pipeline_logging("fulltext-chunk", logging.INFO)
    load_config()

    parser = argparse.ArgumentParser(description="Chunk and embed full-text paragraphs for open-access papers.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of papers to chunk")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete all existing chunks for this model first, then re-chunk every paper with full text",
    )
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder()
    engine = make_engine()
    session_factory = make_session_factory(engine)

    if args.force:
        session = session_factory()
        try:
            deleted = reset_chunk_data(session, embedder.model_name)
        finally:
            session.close()
        print(f"--force: deleted {deleted} existing chunks for model {embedder.model_name!r}.")

    pipeline = FullTextChunkPipeline(embedder=embedder, session_factory=session_factory)
    result = pipeline.run(limit=args.limit)
    print(
        f"Full-text chunk run finished: {result['papers_processed']} papers processed, "
        f"{result['chunks_created']} chunks created."
    )


if __name__ == "__main__":
    main()
```

In `pyproject.toml`, add this line directly after `rb-fulltext-fetch = "researchbridge.fulltext.cli:main"`:

```toml
rb-fulltext-chunk = "researchbridge.fulltext.cli_chunk:main"
```

- [ ] **Step 6: Verify the CLI runs**

Run: `python -m researchbridge.fulltext.cli_chunk --limit 1`
Expected: prints a "Full-text chunk run finished: N papers processed, M chunks created." line (N/M may be 0 if no `PaperFullText` rows exist locally yet — that's fine, it should not error).

- [ ] **Step 7: Commit**

```bash
git add src/researchbridge/fulltext/chunk_pipeline.py src/researchbridge/fulltext/cli_chunk.py pyproject.toml tests/test_fulltext_chunk_pipeline.py
git commit -m "feat(fulltext): add chunk embedding pipeline and rb-fulltext-chunk CLI"
```

---

### Task 4: Widen Stage 1 paper selection

**Files:**
- Modify: `src/researchbridge/embedding/search.py`
- Test: `tests/test_embedding_search.py` (create if it doesn't already exist — check first with `ls tests/test_embedding_search.py`; if it exists, add to it)

**Interfaces:**
- Consumes: `PaperFullTextChunk` (Task 1), existing `Embedding`, `Paper` models, `Embedder` protocol.
- Produces: `search_papers_with_fulltext(session: Session, query_text: str, embedder: Embedder, top_k: int = 10) -> list[tuple[Paper, float]]` — same return shape as `search_by_text`, so it's a drop-in replacement at the call site.

- [ ] **Step 1: Check whether a search test file already exists**

Run: `ls tests/test_embedding_search.py 2>&1 || echo "does not exist"`

If it exists, read it first to match its fixture conventions before adding tests. If not, create it fresh per Step 2 below.

- [ ] **Step 2: Write the failing tests**

Create (or add to) `tests/test_embedding_search.py`:

```python
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from researchbridge.db.models import EMBEDDING_DIM, Embedding, Paper, PaperFullTextChunk
from researchbridge.embedding.pipeline import EMBEDDING_TYPE
from researchbridge.embedding.search import search_papers_with_fulltext


def _hash_to_unit_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


@dataclass
class FakeEmbedder:
    model_name: str = "fake-embedder-v1"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_hash_to_unit_vector(t) for t in texts]


def _paper(session, source_id: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=source_id, title="A Paper", abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def _add_abstract_embedding(session, embedder, paper, text) -> None:
    [vector] = embedder.embed_texts([text])
    session.add(
        Embedding(paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name, vector=vector)
    )


def _add_chunk(session, embedder, paper, text, section="introduction", index=0) -> None:
    [vector] = embedder.embed_texts([text])
    session.add(
        PaperFullTextChunk(
            paper_id=paper.id, section=section, paragraph_index=index, text=text,
            model_name=embedder.model_name, embedding=vector,
        )
    )


def test_surfaces_a_paper_that_only_matches_on_abstract(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, "p1")
    _add_abstract_embedding(session, embedder, paper, "the exact question text")
    session.commit()

    results = search_papers_with_fulltext(session, "the exact question text", embedder)

    assert [p.id for p, _ in results] == [paper.id]
    session.close()


def test_surfaces_a_paper_that_only_matches_on_a_fulltext_chunk(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, "p1")
    _add_abstract_embedding(session, embedder, paper, "unrelated abstract text")
    _add_chunk(session, embedder, paper, "the exact question text")
    session.commit()

    results = search_papers_with_fulltext(session, "the exact question text", embedder)

    assert [p.id for p, _ in results] == [paper.id]
    session.close()


def test_paper_with_only_abstract_still_surfaces(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, "p1")
    _add_abstract_embedding(session, embedder, paper, "the exact question text")
    session.commit()

    results = search_papers_with_fulltext(session, "the exact question text", embedder)

    assert len(results) == 1
    session.close()


def test_excludes_curated_out_papers(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, "p1")
    _add_abstract_embedding(session, embedder, paper, "the exact question text")
    paper.excluded_at = datetime.now(timezone.utc)
    session.commit()

    results = search_papers_with_fulltext(session, "the exact question text", embedder)

    assert results == []
    session.close()


def test_ignores_chunks_from_a_different_model(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, "p1")
    _add_chunk(session, FakeEmbedder(model_name="other-model"), paper, "the exact question text")
    session.commit()

    results = search_papers_with_fulltext(session, "the exact question text", embedder)

    assert results == []
    session.close()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_embedding_search.py -v`
Expected: FAIL with `ImportError: cannot import name 'search_papers_with_fulltext'`

- [ ] **Step 4: Implement `search_papers_with_fulltext`**

In `src/researchbridge/embedding/search.py`, add these imports at the top (extending the existing `from sqlalchemy import select` and `from researchbridge.db.models import Embedding, Paper` lines):

```python
from sqlalchemy import func, select
...
from researchbridge.db.models import Embedding, Paper, PaperFullTextChunk
```

Then append this function at the end of the file:

```python
def search_papers_with_fulltext(
    session: Session, query_text: str, embedder: Embedder, top_k: int = 10
) -> list[tuple[Paper, float]]:
    """Like search_by_text, but a paper can also surface via its
    best-matching full-text paragraph, not just its title/abstract
    embedding - a paper's score is whichever of the two is closer (LEAST:
    smaller cosine distance = more similar). A paper missing either side
    (no abstract embedding yet, or no full-text chunks) still surfaces on
    whichever side it has - COALESCE treats the missing side as maximally
    far (2.0, the max possible cosine distance) rather than excluding the
    paper.
    """
    [vector] = embedder.embed_texts([query_text])

    abstract_distance = Embedding.vector.cosine_distance(vector)
    best_chunk = (
        select(
            PaperFullTextChunk.paper_id.label("paper_id"),
            func.min(PaperFullTextChunk.embedding.cosine_distance(vector)).label("distance"),
        )
        .where(PaperFullTextChunk.model_name == embedder.model_name)
        .group_by(PaperFullTextChunk.paper_id)
        .subquery()
    )

    combined = func.least(
        func.coalesce(abstract_distance, 2.0), func.coalesce(best_chunk.c.distance, 2.0)
    ).label("combined")

    query = (
        select(Paper, combined)
        .select_from(Paper)
        .outerjoin(
            Embedding,
            (Embedding.paper_id == Paper.id)
            & (Embedding.embedding_type == EMBEDDING_TYPE)
            & (Embedding.model_name == embedder.model_name),
        )
        .outerjoin(best_chunk, best_chunk.c.paper_id == Paper.id)
        .where(
            Paper.excluded_at.is_(None),
            (Embedding.id.isnot(None)) | (best_chunk.c.paper_id.isnot(None)),
        )
        .order_by(combined.asc())
        .limit(top_k)
    )
    return [(row[0], row[1]) for row in session.execute(query).all()]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_embedding_search.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 6: Run the existing search tests to confirm no regression**

Run: `pytest tests/ -k "search_by_text or find_similar" -v`
Expected: PASS (unchanged — `search_by_text` itself was not modified)

- [ ] **Step 7: Commit**

```bash
git add src/researchbridge/embedding/search.py tests/test_embedding_search.py
git commit -m "feat(embedding): widen paper candidate search to include full-text chunk matches"
```

---

### Task 5: Merge full-text passages into Stage 2 quote ranking

**Files:**
- Modify: `src/researchbridge/qa/answer.py`
- Modify: `tests/test_qa_answer.py` (extend existing file)

**Interfaces:**
- Consumes: `search_papers_with_fulltext` (Task 4), `PaperFullTextChunk` (Task 1).
- Produces: `QuoteHit` gains a `source: Literal["claim", "passage"]` field; `evidence_id` becomes `uuid.UUID | None` (was `uuid.UUID`). `answer_question(...)` signature unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_qa_answer.py` (below the existing tests, using the file's existing `_add_paper`/`_add_claim`/`FakeEmbedder` helpers):

```python
from researchbridge.db.models import PaperFullTextChunk


def _add_chunk(session, embedder, paper, text, section="introduction", index=0) -> None:
    [vector] = embedder.embed_texts([text])
    session.add(
        PaperFullTextChunk(
            paper_id=paper.id, section=section, paragraph_index=index, text=text,
            model_name=embedder.model_name, embedding=vector,
        )
    )


def test_surfaces_a_fulltext_passage_alongside_claims(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    _add_chunk(session, embedder, paper, "the exact question text", section="results")
    session.commit()

    hits = answer_question(session, embedder, "the exact question text")

    assert hits[0].text == "the exact question text"
    assert hits[0].source == "passage"
    assert hits[0].evidence_id is None
    assert hits[0].section == "results"
    session.close()


def test_existing_claim_hits_are_tagged_source_claim(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    session.commit()

    hits = answer_question(session, embedder, "graph transformers for fraud detection")

    assert hits[0].source == "claim"
    assert hits[0].evidence_id is not None
    session.close()


def test_drops_a_passage_that_already_contains_a_candidate_claim(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    _add_chunk(
        session, embedder, paper,
        "In our experiments, the model was evaluated only on offline datasets due to access constraints.",
        section="results",
    )
    session.commit()

    hits = answer_question(session, embedder, "graph transformers for fraud detection")

    assert all(hit.text != "In our experiments, the model was evaluated only on offline datasets due to access constraints." for hit in hits)
    session.close()


def test_paper_with_only_passages_and_no_claims_still_returns_hits(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_chunk(session, embedder, paper, "the exact question text")
    session.commit()

    hits = answer_question(session, embedder, "the exact question text")

    assert len(hits) == 1
    assert hits[0].source == "passage"
    session.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_qa_answer.py -v`
Expected: the four new tests FAIL (`AttributeError: 'QuoteHit' object has no attribute 'source'`), all pre-existing tests still PASS.

- [ ] **Step 3: Rewrite `answer.py`**

Replace `src/researchbridge/qa/answer.py` in full:

```python
"""Extractive Q&A over the corpus.

Retrieves candidate papers by embedding similarity - reusing
embedding/search.py's search_papers_with_fulltext, which considers both a
paper's title/abstract embedding and its best-matching full-text paragraph
(see that function's docstring). Stage two re-ranks two kinds of candidate
quote against the exact question: already-extracted claims (verbatim
ExtractedClaim.text, grounded by extraction/pipeline.py's
_quote_is_grounded at extraction time) and raw full-text paragraphs
(PaperFullTextChunk - also verbatim, straight from PaperFullText.sections,
just never run through claim extraction). There is no generation step
here, matching the "never invent" rule the rest of the app follows (see
Evidence's docstring in db/models.py on why extraction_method="stub" rows
are filtered out).

A full-text paragraph that already contains a candidate claim's exact text
is dropped before ranking - claims are always verbatim substrings of some
paragraph (that's what "grounded" means here), so keeping both would show
the same content twice under two different labels.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.db.models import Evidence, ExtractedClaim, PaperFullTextChunk
from researchbridge.embedding.base import Embedder
from researchbridge.embedding.search import search_papers_with_fulltext


@dataclass
class QuoteHit:
    evidence_id: uuid.UUID | None
    paper_id: uuid.UUID
    paper_title: str
    paper_source: str
    claim_type: str
    text: str
    section: str | None
    confidence: str
    score: float
    """Cosine similarity to the question (embeddings are L2-normalized, so
    this is a plain dot product) - higher is more relevant. For display/sort
    only, not a probability."""
    source: Literal["claim", "passage"]
    """"claim": a vetted ExtractedClaim, backed by an Evidence row.
    "passage": a raw full-text paragraph that was never run through claim
    extraction - still a verbatim quote from the paper, just not
    claim-typed or confidence-rated (confidence is "n/a")."""


def answer_question(
    session: Session,
    embedder: Embedder,
    question: str,
    top_k_papers: int = 10,
    top_k_quotes: int = 8,
) -> list[QuoteHit]:
    candidates = search_papers_with_fulltext(session, question, embedder, top_k=top_k_papers)
    if not candidates:
        return []
    papers_by_id = {paper.id: paper for paper, _ in candidates}

    claim_rows = session.execute(
        select(ExtractedClaim, Evidence)
        .join(Evidence, Evidence.id == ExtractedClaim.evidence_id)
        .where(
            ExtractedClaim.paper_id.in_(papers_by_id.keys()),
            Evidence.extraction_method != "stub",
        )
    ).all()

    chunk_rows = list(
        session.execute(
            select(PaperFullTextChunk).where(
                PaperFullTextChunk.paper_id.in_(papers_by_id.keys()),
                PaperFullTextChunk.model_name == embedder.model_name,
            )
        ).scalars()
    )
    chunk_rows = _drop_chunks_matching_claims(chunk_rows, claim_rows)

    if not claim_rows and not chunk_rows:
        return []

    claim_texts = [claim.text for claim, _ in claim_rows]
    vectors = embedder.embed_texts([question] + claim_texts)
    question_vector, claim_vectors = vectors[0], vectors[1:]

    scored: list[tuple[float, QuoteHit]] = []
    for vector, (claim, evidence) in zip(claim_vectors, claim_rows, strict=True):
        paper = papers_by_id[claim.paper_id]
        scored.append((
            _dot(question_vector, vector),
            QuoteHit(
                evidence_id=evidence.id,
                paper_id=claim.paper_id,
                paper_title=paper.title,
                paper_source=paper.source,
                claim_type=claim.claim_type,
                text=claim.text,
                section=evidence.section,
                confidence=claim.confidence,
                score=0.0,
                source="claim",
            ),
        ))

    for chunk in chunk_rows:
        paper = papers_by_id[chunk.paper_id]
        scored.append((
            _dot(question_vector, chunk.embedding),
            QuoteHit(
                evidence_id=None,
                paper_id=chunk.paper_id,
                paper_title=paper.title,
                paper_source=paper.source,
                claim_type="full_text_passage",
                text=chunk.text,
                section=chunk.section,
                confidence="n/a",
                score=0.0,
                source="passage",
            ),
        ))

    scored.sort(key=lambda item: item[0], reverse=True)

    results = []
    for score, hit in scored[:top_k_quotes]:
        hit.score = score
        results.append(hit)
    return results


def _drop_chunks_matching_claims(
    chunks: list[PaperFullTextChunk], claim_rows: list[tuple[ExtractedClaim, Evidence]]
) -> list[PaperFullTextChunk]:
    claim_texts_by_paper: dict[uuid.UUID, list[str]] = {}
    for claim, _ in claim_rows:
        claim_texts_by_paper.setdefault(claim.paper_id, []).append(claim.text)

    kept = []
    for chunk in chunks:
        claim_texts = claim_texts_by_paper.get(chunk.paper_id, [])
        if any(text in chunk.text for text in claim_texts):
            continue
        kept.append(chunk)
    return kept


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
```

- [ ] **Step 4: Run all qa/answer tests to verify they pass**

Run: `pytest tests/test_qa_answer.py -v`
Expected: PASS — all pre-existing tests (unmodified behavior for papers without full text) and all 4 new tests.

- [ ] **Step 5: Commit**

```bash
git add src/researchbridge/qa/answer.py tests/test_qa_answer.py
git commit -m "feat(qa): merge full-text passages into ask-the-corpus quote ranking"
```

---

### Task 6: API schema and route wiring

**Files:**
- Modify: `src/researchbridge/api/schemas.py` (`QuoteHitOut`, ~line 111)
- Modify: `src/researchbridge/api/qa_routes.py` (both `QuoteHitOut(...)` construction sites, ~lines 152 and 199)
- Modify: `tests/test_qa_api.py` (extend existing file)

**Interfaces:**
- Consumes: `QuoteHit.source` (Task 5).
- Produces: `QuoteHitOut.source: Literal["claim", "passage"] = "claim"` (default keeps existing persisted `QaQuestion.hits` JSON, which predates this field, parseable).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_qa_api.py`:

```python
from researchbridge.db.models import PaperFullTextChunk


def _add_chunk(session, embedder, paper, text, section="introduction", index=0) -> None:
    [vector] = embedder.embed_texts([text])
    session.add(
        PaperFullTextChunk(
            paper_id=paper.id, section=section, paragraph_index=index, text=text,
            model_name=embedder.model_name, embedding=vector,
        )
    )


def test_ask_tags_fulltext_passages_with_source_passage(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_chunk(session, embedder, paper, "the exact question text")
    session.commit()

    response = client.post("/api/ask", json={"question": "the exact question text"})

    assert response.status_code == 200
    hit = response.json()["hits"][0]
    assert hit["source"] == "passage"
    assert hit["evidence_id"] is None


def test_ask_tags_claims_with_source_claim(client, session, embedder) -> None:
    paper = _add_paper(session, embedder, "p1", "graph transformers for fraud detection")
    _add_claim(session, paper, "limitations", "evaluated only on offline datasets")
    session.commit()

    response = client.post("/api/ask", json={"question": "graph transformers for fraud detection"})

    assert response.status_code == 200
    hit = response.json()["hits"][0]
    assert hit["source"] == "claim"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_qa_api.py -v`
Expected: the two new tests FAIL with a `KeyError: 'source'`, existing tests still PASS.

- [ ] **Step 3: Add `source` to `QuoteHitOut`**

In `src/researchbridge/api/schemas.py`, change the `QuoteHitOut` class (around line 111):

```python
class QuoteHitOut(BaseModel):
    evidence_id: uuid.UUID | None = None
    paper_id: uuid.UUID
    paper_title: str
    paper_source: str
    claim_type: str
    text: str
    section: str | None
    confidence: str
    score: float
    source: Literal["claim", "passage"] = "claim"
```

(`Literal` is already imported at the top of this file — line 17.)

- [ ] **Step 4: Wire `source` through both route construction sites**

In `src/researchbridge/api/qa_routes.py`, find the `QuoteHitOut(...)` construction inside `create_question` (~line 152) and add `source=hit.source,` after `score=hit.score,`:

```python
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
            source=hit.source,
        ).model_dump(mode="json")
        for hit in hits
    ]
```

Do the same in the `ask` route (~line 199):

```python
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
                source=hit.source,
            )
            for hit in hits
        ],
        summarization_available=ollama_enabled(),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_qa_api.py -v`
Expected: PASS (all tests, including the two new ones)

- [ ] **Step 6: Run the full backend QA test suite**

Run: `pytest tests/test_qa_answer.py tests/test_qa_api.py tests/test_qa_summarize.py tests/test_embedding_search.py tests/test_fulltext_models.py tests/test_fulltext_chunking.py tests/test_fulltext_chunk_pipeline.py -v`
Expected: PASS, no regressions.

- [ ] **Step 7: Commit**

```bash
git add src/researchbridge/api/schemas.py src/researchbridge/api/qa_routes.py tests/test_qa_api.py
git commit -m "feat(api): tag ask-the-corpus quote hits with claim/passage source"
```

---

### Task 7: Frontend type update

**Files:**
- Modify: `frontend/lib/qaApi.ts` (`QuoteHit` type, ~line 3)

**Interfaces:**
- Consumes: `source` field on `AskResponse`/`QaQuestion` hits (Task 6).

No behavior change is needed in `frontend/app/ask/page.tsx`'s `QuoteCard`: a passage hit already has `evidence_id: null` (the existing `{questionId && hit.evidence_id && <EvidenceReviewControl .../>}` check at `page.tsx:518` already skips the review control for it), and its `claim_type` is the human-readable string `"full_text_passage"` (the existing `{hit.claim_type.replace(/_/g, " ")}` badge at `page.tsx:506` already renders it as "full text passage"). The `source` field is added to the type for completeness and for any future code that wants to branch on it explicitly, but no current rendering logic depends on it.

- [ ] **Step 1: Add the field to the type**

In `frontend/lib/qaApi.ts`, update `QuoteHit` (~line 3):

```typescript
export type QuoteHit = {
  evidence_id?: string | null;
  paper_id: string;
  paper_title: string;
  paper_source: string;
  claim_type: string;
  text: string;
  section: string | null;
  confidence: string;
  score: number;
  source?: "claim" | "passage";
};
```

- [ ] **Step 2: Type-check the frontend**

Run: `cd frontend && npm run typecheck` (or `npx tsc --noEmit` if there's no dedicated script — check `frontend/package.json`'s `scripts` first)
Expected: no new type errors.

- [ ] **Step 3: Manually verify in the browser**

Start the frontend dev server, open `/ask`, ask a question against a paper that has full-text chunks (requires Task 3's pipeline to have been run locally at least once), and confirm:
- A passage hit renders with badge text "full text passage" and no evidence-review control.
- A claim hit renders exactly as before.

- [ ] **Step 4: Commit**

```bash
git add frontend/lib/qaApi.ts
git commit -m "feat(frontend): add source field to QuoteHit type"
```

---

## After all tasks

Run the full backend suite once to confirm nothing elsewhere regressed:

```bash
pytest tests/ -x -q
```

Then, to actually populate chunk data locally for manual testing: run `rb-fulltext-fetch` (if not already run) followed by `rb-fulltext-chunk`.
