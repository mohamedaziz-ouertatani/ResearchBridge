"""Deterministic keyword extraction for external market/patent search queries.

No invented terms - every keyword is text the user actually wrote, matching
this codebase's grounding discipline elsewhere (novelty/applications/gap all
only ever surface text that was actually retrieved). Uses sklearn's
CountVectorizer purely as a frequency counter over the single input document
- no training, no external corpus, no IDF weighting needed since there is
only ever one document per call.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer


def extract_keywords(title: str | None, raw_text: str, max_keywords: int = 8) -> list[str]:
    """Top max_keywords most frequent non-stopword unigrams/bigrams from
    title + raw_text. Returns [] if nothing survives stopword removal (e.g.
    empty, whitespace-only, or symbol-only input) - callers treat that as
    "nothing to search," not an error."""
    combined = f"{title or ''} {raw_text}".strip()
    if not combined:
        return []

    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        counts = vectorizer.fit_transform([combined])
    except ValueError:
        # sklearn raises ValueError when every token is a stopword or the
        # vocabulary is otherwise empty (e.g. symbol-only input) - treat
        # the same as "no keywords found", not a crash.
        return []

    terms = vectorizer.get_feature_names_out()
    frequencies = counts.toarray()[0]
    ranked = sorted(zip(terms, frequencies, strict=True), key=lambda pair: (-pair[1], pair[0]))
    return [term for term, _count in ranked[:max_keywords]]
