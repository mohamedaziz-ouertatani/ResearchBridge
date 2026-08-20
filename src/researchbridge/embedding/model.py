"""Free/local embedding model - sentence-transformers, no paid API involved.

Per the blueprint's Free/Open-Source First constraint (Builder Context),
this is the default and only production Embedder: a small open-source
model that runs on CPU, no API key or network call required at inference
time (only once, to download the model on first use - cached afterward).
"""

from __future__ import annotations

import os

MODEL_NAME = "all-MiniLM-L6-v2"

# Windows: torch and numpy/scipy each bundle their own OpenMP runtime
# (libiomp5md.dll), and loading both in one process crashes without this.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class SentenceTransformerEmbedder:
    model_name = MODEL_NAME

    def __init__(self) -> None:
        self._model = None  # lazy-loaded: importing/loading sentence-transformers is not free time-wise

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()
