"""Baseline 2 (Sec 27): BM25."""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.db.models import Paper
from researchbridge.retrieval.text import document_text

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Bm25Retriever:
    name = "bm25"

    def __init__(self) -> None:
        self._fitted = False
        self._index = None
        self._papers: list[Paper] = []

    def fit(self, session: Session) -> None:
        from rank_bm25 import BM25Okapi

        papers, tokenized_docs = [], []
        for paper in session.execute(select(Paper)).scalars():
            text = document_text(paper)
            if text is not None:
                papers.append(paper)
                tokenized_docs.append(tokenize(text))

        self._papers = papers
        self._index = BM25Okapi(tokenized_docs) if tokenized_docs else None
        self._fitted = True

    def search(self, query: str, top_k: int) -> list[tuple[Paper, float]]:
        if not self._fitted:
            raise RuntimeError("call fit() before search()")
        if self._index is None:  # fit() ran, but the corpus was empty
            return []

        scores = self._index.get_scores(tokenize(query))
        ranked = sorted(zip(self._papers, scores), key=lambda pair: pair[1], reverse=True)
        return [(paper, float(score)) for paper, score in ranked[:top_k] if score > 0]
