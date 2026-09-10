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
