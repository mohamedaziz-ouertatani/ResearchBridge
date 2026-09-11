"""Embedder interface: provider-agnostic text -> vector embedding.

Mirrors extraction.base.Extractor for the same reason: production uses a
free/local model (see model.py, per the blueprint's Free/Open-Source First
constraint), while tests inject a fast fake so pipeline mechanics can be
verified without loading the real model every run.
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    model_name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalized embedding vector per input text, same order."""
        ...


class CachingEmbedder:
    """Wraps an Embedder with an in-memory, per-instance cache keyed by exact
    text.

    build_assessment() re-derives overlapping sets of claim texts for
    several independent sub-assessments (dimension coverage, cross-paper
    gap clustering, applications matching all separately re-embed claims
    the others already embedded) - each call went straight to the
    (CPU-bound) model with zero memoization. Wrapping the embedder once per
    build_assessment() call and threading the wrapper through instead of
    the raw embedder collapses those duplicate encodes into one, without
    touching any of that logic (see the perf pass, 2026-09-11). Scoped to a
    single instance/call, not process-wide: a persistent cache would need
    invalidation the moment corpus text changes, which this sidesteps
    entirely by just not outliving one build_assessment() call.
    """

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self.model_name = inner.model_name
        self._cache: dict[str, list[float]] = {}

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        uncached = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if uncached:
            for text, vector in zip(uncached, self._inner.embed_texts(uncached), strict=True):
                self._cache[text] = vector
        return [self._cache[t] for t in texts]
