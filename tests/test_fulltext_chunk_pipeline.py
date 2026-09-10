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
