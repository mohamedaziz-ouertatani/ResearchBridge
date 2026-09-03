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


# --- long-text chunking (fix for silent 256-token truncation) ---------------
# Found investigating "check extraction/embedding for weaknesses": all-
# MiniLM-L6-v2 has max_seq_length=256 and sentence-transformers silently
# truncates anything longer, with no error or warning - proven live against
# the real corpus (65% of papers' title+abstract text exceeds 256 tokens).
# embed_texts() now chunks + weighted-mean-pools any text over the limit,
# staying on the same model so no similarity threshold elsewhere in the
# codebase needs recalibrating - see model.py's own docstring for why a
# longer-context model swap was rejected instead.


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_a_short_text_is_unaffected_by_the_chunking_path() -> None:
    # regression guard: a text within max_seq_length must take the exact
    # same single-pass code path as before this fix, byte-for-byte
    embedder = SentenceTransformerEmbedder()
    text = "A study of federated learning for fraud detection."

    via_embed_texts = embedder.embed_texts([text])[0]
    via_raw_model = embedder.model.encode(
        [text], normalize_embeddings=True, convert_to_numpy=True
    )[0].tolist()

    assert via_embed_texts == via_raw_model


def test_a_long_texts_own_vector_is_normalized() -> None:
    embedder = SentenceTransformerEmbedder()
    long_text = "Deep neural networks for image classification. " * 40  # well over 256 tokens

    [vector] = embedder.embed_texts([long_text])

    assert len(vector) == EMBEDDING_DIM
    norm = math.sqrt(sum(x * x for x in vector))
    assert math.isclose(norm, 1.0, abs_tol=1e-4)


def test_content_beyond_the_old_256_token_cutoff_now_affects_the_embedding() -> None:
    # the exact proof of the bug this fix closes: before it, embedding a
    # long text was IDENTICAL (cosine similarity 1.0000) to embedding just
    # its first 256 tokens, because sentence-transformers silently
    # truncated the rest. Two texts sharing an identical first ~250 tokens
    # but diverging after that must no longer embed identically.
    embedder = SentenceTransformerEmbedder()
    shared_prefix = "Deep neural networks for image classification. " * 30
    diverges_early = shared_prefix + "A recipe for baking sourdough bread at home. " * 10
    diverges_late = shared_prefix + "A recipe for baking sourdough bread at home. " * 10 + (
        "This paper additionally proposes a novel transformer architecture for real-time "
        "medical image segmentation, evaluated on three independent clinical datasets."
    )

    v_early, v_late = embedder.embed_texts([diverges_early, diverges_late])

    assert _cosine(v_early, v_late) < 0.999


def test_a_tail_only_query_matches_the_chunked_embedding_better_than_the_truncated_one() -> None:
    # real-world payoff of the fix, not just a synthetic invariant: a query
    # built only from a long text's tail (content the old truncate-only
    # behavior could never see) must be at least as close to the new
    # chunked embedding as to the old truncated-only one
    embedder = SentenceTransformerEmbedder()
    tail_content = (
        "In the final section, we introduce a novel differentially private aggregation "
        "scheme for cross-silo federated learning under adversarial dropout, and evaluate "
        "it against three baselines on a held-out clinical cohort."
    )
    long_text = "Deep neural networks for image classification. " * 40 + tail_content

    [chunked_vector] = embedder.embed_texts([long_text])
    truncated_vector = embedder.model.encode(
        [long_text], normalize_embeddings=True, convert_to_numpy=True
    )[0].tolist()
    [tail_query_vector] = embedder.embed_texts([tail_content])

    assert _cosine(tail_query_vector, chunked_vector) > _cosine(tail_query_vector, truncated_vector)
