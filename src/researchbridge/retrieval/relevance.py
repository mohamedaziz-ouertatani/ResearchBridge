"""Builds a query set with relevance judgments from the Sec 25 benchmark.

Each benchmark paper's own research_question becomes a query, and that
paper is the query's one known-relevant document (self-retrieval: can a
retriever find the source paper given a paraphrase of what it investigates,
written independently from the paper's own title/abstract text).

This is a real evaluation, not a placebo - the query text comes from
research_question/problem, never from the title or abstract a retriever
actually indexes, so a hit means the retriever bridged a genuine wording
gap. But it is a weaker signal than a judged set with multiple relevant
documents per query: Recall@K and nDCG@K degenerate to a 0/1 hit indicator
per query with only one relevant document. Sec 27's "use manually judged
relevance labels" describes a richer judgment process than this - this is
the pragmatic version, reusing annotation work that already exists rather
than commissioning a second annotation effort.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.benchmark.store import Annotation, load_all
from researchbridge.db.models import Paper


@dataclass
class QueryJudgment:
    query: str
    label: str
    """A short identifier for this query, for logging/debugging only - never
    read by the evaluation harness itself. For self-retrieval it's the
    source paper's arxiv id; topical_queries.py sets it to a query slug
    instead, since a topical query has no single source paper."""
    relevant_ids: set[uuid.UUID]


def _query_text(annotation: Annotation) -> str | None:
    text = annotation.fields.get("research_question") or annotation.fields.get("problem") or ""
    return text.strip() or None


def build_query_set(session: Session, benchmark_dir: Path) -> tuple[list[QueryJudgment], list[str]]:
    """Returns (judgments, skipped_source_ids).

    A paper is skipped if it has no usable query text, or if its source
    paper isn't in the corpus (e.g. the annotation sample predates a
    re-ingest) - loud skips, not silent gaps, per the project's ingestion
    reliability standard (Sec 8).
    """
    annotations = load_all(benchmark_dir)

    source_ids = [a.source_id for a in annotations]
    rows = session.execute(
        select(Paper.id, Paper.source_id).where(Paper.source == "arxiv", Paper.source_id.in_(source_ids))
    ).all()
    paper_id_by_source_id = {source_id: paper_id for paper_id, source_id in rows}

    judgments: list[QueryJudgment] = []
    skipped: list[str] = []
    for annotation in annotations:
        query = _query_text(annotation)
        paper_id = paper_id_by_source_id.get(annotation.source_id)
        if query is None or paper_id is None:
            skipped.append(annotation.source_id)
            continue
        judgments.append(QueryJudgment(query=query, label=annotation.source_id, relevant_ids={paper_id}))

    return judgments, skipped
