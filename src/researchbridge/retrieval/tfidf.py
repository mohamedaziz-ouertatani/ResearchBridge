"""Baseline 1 (Sec 27): TF-IDF + cosine similarity."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.db.models import Paper
from researchbridge.retrieval.text import document_text


class TfidfRetriever:
    name = "tfidf"

    def __init__(self) -> None:
        self._vectorizer = None
        self._matrix = None
        self._papers: list[Paper] = []

    def fit(self, session: Session) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        papers, texts = [], []
        for paper in session.execute(select(Paper)).scalars():
            text = document_text(paper)
            if text is not None:
                papers.append(paper)
                texts.append(text)

        self._papers = papers
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(texts) if texts else None

    def search(self, query: str, top_k: int) -> list[tuple[Paper, float]]:
        from sklearn.metrics.pairwise import linear_kernel

        if self._vectorizer is None:
            raise RuntimeError("call fit() before search()")
        if self._matrix is None:  # empty corpus
            return []

        query_vector = self._vectorizer.transform([query])
        scores = linear_kernel(query_vector, self._matrix).ravel()  # cosine similarity: both sides are L2-normalized by TfidfVectorizer
        return _top_k(self._papers, scores, top_k)


def _top_k(papers: list[Paper], scores, top_k: int) -> list[tuple[Paper, float]]:
    ranked = sorted(zip(papers, scores), key=lambda pair: pair[1], reverse=True)
    return [(paper, float(score)) for paper, score in ranked[:top_k] if score > 0]
