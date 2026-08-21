"""The text representation a paper offers to any retrieval method.

Shared by every baseline in Sec 27 (TF-IDF, BM25, embeddings, hybrid) and
by the Week 6 embedding pipeline, so "does semantic retrieval outperform
lexical retrieval" is measuring the retrieval method, not an accidental
difference in what text each method got to see.
"""

from __future__ import annotations

from researchbridge.db.models import Paper


def document_text(paper: Paper) -> str | None:
    title = (paper.title or "").strip()
    abstract = (paper.abstract or "").strip()
    text = f"{title}\n\n{abstract}".strip()
    return text or None
