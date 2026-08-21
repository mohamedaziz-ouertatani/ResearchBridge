"""Loads the Sec 27 topical query set (benchmark/retrieval_queries.yaml).

This exists because relevance.py's self-retrieval queries (each benchmark
paper's own research_question, judged against itself) turned out to
saturate at a perfect 1.0 for every baseline - see relevance.py's
docstring. Testing why (documented in the commit that added this module)
showed the cause isn't query wording or candidate-pool size: it's that a
query GENERATED FROM a paper always retains enough of that paper's own
phrasing to stay a near-unique fingerprint for it, almost regardless of
how the query field or pool is chosen. That's a known structural weakness
of self-retrieval IR benchmarks - the query and target share generation
lineage, which inflates scores independent of actual retrieval quality.

The fix here is queries with no single source paper at all: each one in
retrieval_queries.yaml was written first, before looking at any specific
paper, then judged against pooled candidates from all four retrieval
baselines (top 12 each - standard IR pooling, so judgment isn't biased
toward whichever method happened to be checked first). Multiple relevant
documents per query, unlike the self-retrieval set's exactly one - that's
what lets Recall@K and nDCG@K measure something real here instead of
degenerating to a 0/1 hit indicator.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.db.models import Paper
from researchbridge.retrieval.relevance import QueryJudgment

DEFAULT_QUERIES_PATH = Path("benchmark/retrieval_queries.yaml")


def build_topical_query_set(
    session: Session, path: Path = DEFAULT_QUERIES_PATH
) -> tuple[list[QueryJudgment], list[str]]:
    """Returns (judgments, skipped_source_ids) - same shape as relevance.build_query_set.

    A judged source_id is skipped (not silently dropped) if it no longer
    resolves to a corpus paper - loud skips, per the project's ingestion
    reliability standard (Sec 8).
    """
    if not path.exists():
        return [], []

    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_queries = document.get("queries") or []

    all_source_ids = {sid for q in raw_queries for sid in q.get("relevant_source_ids", [])}
    rows = session.execute(
        select(Paper.id, Paper.source_id).where(Paper.source == "arxiv", Paper.source_id.in_(all_source_ids))
    ).all()
    paper_id_by_source_id = {source_id: paper_id for paper_id, source_id in rows}

    judgments: list[QueryJudgment] = []
    skipped: list[str] = []
    for i, q in enumerate(raw_queries):
        query_text = (q.get("query") or "").strip()
        relevant_ids: set = set()
        for source_id in q.get("relevant_source_ids") or []:
            paper_id = paper_id_by_source_id.get(source_id)
            if paper_id is None:
                skipped.append(source_id)
                continue
            relevant_ids.add(paper_id)

        if query_text and relevant_ids:
            judgments.append(QueryJudgment(query=query_text, label=f"topical:{i:02d}", relevant_ids=relevant_ids))

    return judgments, skipped
