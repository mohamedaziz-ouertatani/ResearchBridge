"""Embedding pipeline: select un-embedded papers -> build input text -> embed -> persist.

Idempotent: skips papers that already have an embeddings row for this
(embedding_type, model_name) pair, so re-running is safe and cheap.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from researchbridge.db.models import Embedding, EmbeddingRun, Paper
from researchbridge.embedding.base import Embedder
from researchbridge.retrieval.text import document_text

logger = logging.getLogger(__name__)

EMBEDDING_TYPE = "title_abstract"
BATCH_SIZE = 32


def reset_embedding_data(session: Session, model_name: str) -> int:
    """Delete every existing (embedding_type, model_name) embedding so the
    next run re-embeds the whole corpus from scratch (the "force re-embed"
    path). Scoped to this model_name only - other models' embeddings, if
    any ever coexist, are untouched. No downstream table references
    embeddings.id, so this is a plain delete with no cascade to worry
    about, unlike reset_extraction_data.
    """
    result = session.execute(
        delete(Embedding).where(Embedding.embedding_type == EMBEDDING_TYPE, Embedding.model_name == model_name)
    )
    session.commit()
    return result.rowcount


class EmbeddingPipeline:
    def __init__(self, embedder: Embedder, session_factory: sessionmaker[Session]) -> None:
        self.embedder = embedder
        self.session_factory = session_factory

    def run(self, limit: int | None = None) -> str:
        session = self.session_factory()
        run = EmbeddingRun(model_name=self.embedder.model_name, status="running")
        session.add(run)
        session.commit()
        run_id = str(run.id)

        try:
            papers = self._select_unembedded_papers(session, limit)

            for batch in _chunks(papers, BATCH_SIZE):
                texts: list[str] = []
                embeddable: list[Paper] = []
                for paper in batch:
                    text = document_text(paper)
                    if text is None:
                        run.papers_skipped += 1  # defensive: title is required by ingestion validation,
                        continue  # so this should be ~0 in practice

                    texts.append(text)
                    embeddable.append(paper)

                if texts:
                    vectors = self.embedder.embed_texts(texts)
                    for paper, vector in zip(embeddable, vectors, strict=True):
                        session.add(
                            Embedding(
                                paper_id=paper.id,
                                embedding_type=EMBEDDING_TYPE,
                                model_name=self.embedder.model_name,
                                vector=vector,
                            )
                        )
                        run.papers_processed += 1

                session.commit()

            run.status = "completed"
            run.finished_at = datetime.now(UTC)
            session.commit()

        except Exception as exc:  # noqa: BLE001 - run must record failure, not crash silently
            logger.exception("Embedding run %s failed", run_id)
            run.status = "failed"
            run.error_summary = str(exc)[:2000]
            run.finished_at = datetime.now(UTC)
            session.commit()
            raise
        finally:
            session.close()

        return run_id

    def _select_unembedded_papers(self, session: Session, limit: int | None) -> list[Paper]:
        already_embedded = select(Embedding.paper_id).where(
            Embedding.embedding_type == EMBEDDING_TYPE,
            Embedding.model_name == self.embedder.model_name,
        )
        query = select(Paper).where(Paper.id.notin_(already_embedded))
        if limit is not None:
            query = query.limit(limit)
        return list(session.execute(query).scalars())


def _chunks(items: list[Paper], size: int) -> Iterator[list[Paper]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
