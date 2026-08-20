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
