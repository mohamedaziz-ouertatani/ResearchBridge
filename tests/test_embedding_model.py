"""Sanity checks against the real sentence-transformers model.

Downloads the model on first run (~80MB), cached afterward by
sentence-transformers/huggingface_hub. Slower than the fake-embedder-based
pipeline/search tests, but this is the only place that verifies the real
model actually behaves as the rest of the code assumes (correct dimension,
normalized output, semantically meaningful distances).
"""

from __future__ import annotations

import math

from researchbridge.db.models import EMBEDDING_DIM
from researchbridge.embedding.model import MODEL_NAME, SentenceTransformerEmbedder


def test_model_name_matches_configured_embedding_dim() -> None:
    assert MODEL_NAME == "all-MiniLM-L6-v2"  # the model EMBEDDING_DIM=384 is calibrated for


def test_embed_texts_returns_correct_dimension_and_normalized_vectors() -> None:
    embedder = SentenceTransformerEmbedder()
    vectors = embedder.embed_texts(["A sentence about machine learning."])

    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIM

    norm = math.sqrt(sum(x * x for x in vectors[0]))
    assert math.isclose(norm, 1.0, abs_tol=1e-4)


def test_embed_texts_empty_list_returns_empty() -> None:
    embedder = SentenceTransformerEmbedder()
    assert embedder.embed_texts([]) == []


def test_similar_texts_embed_closer_than_dissimilar_texts() -> None:
    embedder = SentenceTransformerEmbedder()
    anchor, similar, different = embedder.embed_texts(
        [
            "Deep neural networks for image classification.",
            "Convolutional neural networks used for classifying images.",
            "A recipe for baking sourdough bread at home.",
        ]
    )

    def cosine_distance(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return 1 - dot  # vectors are already unit-normalized

    distance_to_similar = cosine_distance(anchor, similar)
    distance_to_different = cosine_distance(anchor, different)

    assert distance_to_similar < distance_to_different
