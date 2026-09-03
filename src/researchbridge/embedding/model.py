"""Free/local embedding model - sentence-transformers, no paid API involved.

Per the blueprint's Free/Open-Source First constraint (Builder Context),
this is the default and only production Embedder: a small open-source
model that runs on CPU, no API key or network call required at inference
time (only once, to download the model on first use - cached afterward).

Truncation fix (found investigating "check extraction/embedding for
weaknesses"): all-MiniLM-L6-v2 has a hard 256-token max_seq_length, and
sentence-transformers silently truncates anything longer with no error or
warning surfaced anywhere in this codebase - proven live: embedding a
500-word text produced an embedding IDENTICAL (cosine similarity
1.0000) to embedding just its first 256 tokens. Measured against the
real corpus (n=3000 papers): title+abstract text exceeds 256 tokens for
65% of papers (median 292 tokens, p90=463, max=1587) - meaning most of
the corpus was being indexed on little more than its title and the first
sentence or two of its abstract, with everything else invisible to every
downstream similarity computation (retrieval, novelty, gap detection,
dimension coverage, applications matching all sit on this one embedder).

Fixed by chunking + weighted-mean-pooling ONLY the texts that actually
exceed the limit, staying on the SAME model rather than switching to a
longer-context one: a model swap would double the token budget (e.g.
BAAI/bge-small-en-v1.5's 512, still not enough to cover the p99=903
tail) but invalidates every similarity threshold already calibrated
against this exact model's score distribution throughout the codebase
(applications.py's OWN_TASK_OVERLAP_THRESHOLD, coverage.py's
DIMENSION_MATCH_SIMILARITY, novelty.py's NEAR/FAR_DISTANCE, semantic.py's
MIN_SIMILARITY, extraction/evaluation.py's DEFAULT_SIMILARITY_THRESHOLD,
gaps/cluster.py and gaps/signals.py's thresholds - a different model's
"0.5 cosine similarity" is not the same signal) and requires a full
corpus re-embed - a separate, much larger undertaking than this fix.
Chunking within the same model changes nothing about the embedding space
itself, so every existing threshold stays valid; it only changes what a
too-long text's OWN vector looks like. A text within the limit (the
overwhelming majority of extraction candidates, claim texts, and
dimension labels, which are single sentences) takes the exact same
single-pass code path as before this fix - zero behavior change there.

The chunking approach follows the standard "weighted mean of per-chunk
embeddings" recipe for long-document embedding with a short-context
model (each chunk is separately embedded and L2-normalized, then
averaged weighted by its own token count so a short trailing chunk
doesn't outweigh a full one, then the pooled vector is renormalized to
unit length) - not a novel scheme invented here.
"""

from __future__ import annotations

import os

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

# Windows: torch and numpy/scipy each bundle their own OpenMP runtime
# (libiomp5md.dll), and loading both in one process crashes without this.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Reserve room for the 2 special tokens (CLS/SEP) the tokenizer adds on
# top of content tokens - verified live: 254 content tokens + 2 specials
# = 256 (max_seq_length) exactly, so a chunk built from the full budget
# would itself get silently truncated by 2 tokens when re-encoded.
_SPECIAL_TOKEN_BUDGET = 2


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

        max_tokens = self.model.max_seq_length
        tokenizer = self.model.tokenizer
        token_counts = [len(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
        long_indices = {i for i, n in enumerate(token_counts) if n > max_tokens}

        vectors: list[list[float] | None] = [None] * len(texts)

        # the common case (every text within the limit) takes exactly the
        # same single batched-encode path as before this fix
        short_indices = [i for i in range(len(texts)) if i not in long_indices]
        if short_indices:
            short_vectors = self.model.encode(
                [texts[i] for i in short_indices], normalize_embeddings=True, convert_to_numpy=True
            )
            for i, vector in zip(short_indices, short_vectors, strict=True):
                vectors[i] = vector.tolist()

        for i in long_indices:
            vectors[i] = self._embed_long_text(texts[i], tokenizer, max_tokens)

        return vectors  # type: ignore[return-value]

    def _embed_long_text(self, text: str, tokenizer, max_tokens: int) -> list[float]:
        """Weighted-mean-pools per-chunk embeddings for a text longer than
        max_tokens - see the module docstring for why this is the fix
        instead of a longer-context model swap."""
        content_budget = max_tokens - _SPECIAL_TOKEN_BUDGET
        ids = tokenizer.encode(text, add_special_tokens=False)
        windows = [ids[i : i + content_budget] for i in range(0, len(ids), content_budget)]
        chunk_texts = [tokenizer.decode(window, skip_special_tokens=True) for window in windows]

        chunk_vectors = self.model.encode(chunk_texts, normalize_embeddings=True, convert_to_numpy=True)
        weights = np.array([len(window) for window in windows], dtype=float)
        pooled = (chunk_vectors * weights[:, None]).sum(axis=0) / weights.sum()

        norm = np.linalg.norm(pooled)
        return (pooled / norm).tolist() if norm else pooled.tolist()
