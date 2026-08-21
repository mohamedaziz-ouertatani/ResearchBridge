"""Improvement over the heuristic baseline (Sec 26): embedding-similarity
sentence selection instead of fixed cue phrases.

Still extractive, not generative - for each Sec 28 field, embed every
sentence in the abstract and pick whichever is closest to a description of
that field, taking the sentence itself as the claim. That keeps the same
verbatim-grounding guarantee the heuristic extractor has (the pipeline's
grounding check in extraction/pipeline.py requires an exact substring of
the abstract), which a generative model answering in its own words could
not offer without weakening Sec 15's "no Grounding Illusion" rule - not a
call to make quietly while chasing a better score.

The heuristic baseline's weakest fields (research_question, applications:
0 recall) are exactly where fixed strings are the wrong tool - real
abstracts phrase a research question or an application in enormously
varied ways, and no small phrase list will catch most of them. Semantic
similarity doesn't need the wording to match, only the meaning to.

Uses the Week 6 embedding model (SentenceTransformerEmbedder / all-
MiniLM-L6-v2) - no new model, no new dependency, and it's already fast
enough for this: one batched embed_texts() call per paper covers every
sentence and every field description at once.
"""

from __future__ import annotations

from researchbridge.db.models import Paper
from researchbridge.embedding.base import Embedder
from researchbridge.extraction.base import ClaimCandidate
from researchbridge.extraction.sentences import split_sentences

SEMANTIC_MODEL_VERSION = "semantic-v1"

# One anchor description per Sec 28 field - the sentence most similar to
# this description (by cosine similarity, since Embedder vectors are
# L2-normalized) becomes that field's candidate.
_FIELD_QUERIES: dict[str, str] = {
    "problem": "The problem or challenge that motivates this research.",
    "research_question": "The research question this paper investigates.",
    "method": "The method, approach, or technique this paper proposes.",
    "dataset": "The dataset or data source used in this work.",
    "main_contribution": "The main contribution or novel finding of this paper.",
    "results": "The results or findings reported in this paper.",
    "limitations": "A limitation or weakness acknowledged in this paper.",
    "applications": "A real-world application this work supports.",
}

# Calibrated against this model's actual score distribution, not guessed:
# all-MiniLM-L6-v2 scores a short instructional anchor ("The problem or
# challenge...") much lower against real sentence prose than intuition
# suggests - measured across the real 40-paper benchmark, best-match
# similarity per (paper, field) runs from 0.02 to 0.58, median 0.22, with
# only 10% clearing 0.34. A first attempt at 0.30 sat near the 90th
# percentile and made the extractor abstain on nearly everything. These
# numbers come from the raw similarity distribution alone, never from
# ground-truth match/no-match labels - reading the instrument's own scale,
# not tuning a threshold against the benchmark it's evaluated on.
MIN_SIMILARITY = 0.15
MEDIUM_CONFIDENCE_SIMILARITY = 0.28


class SemanticExtractor:
    extraction_method = "semantic"
    model_version = SEMANTIC_MODEL_VERSION

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def extract(self, paper: Paper) -> list[ClaimCandidate]:
        if not paper.abstract or not paper.abstract.strip():
            return []

        sentences = split_sentences(paper.abstract)
        if not sentences:
            return []

        fields = list(_FIELD_QUERIES)
        vectors = self.embedder.embed_texts(sentences + [_FIELD_QUERIES[f] for f in fields])
        sentence_vectors, field_vectors = vectors[: len(sentences)], vectors[len(sentences) :]

        candidates: list[ClaimCandidate] = []
        for field, field_vector in zip(fields, field_vectors, strict=True):
            best_sentence, best_similarity = _best_match(sentences, sentence_vectors, field_vector)
            if best_similarity < MIN_SIMILARITY:
                continue
            confidence = "medium" if best_similarity >= MEDIUM_CONFIDENCE_SIMILARITY else "low"
            candidates.append(ClaimCandidate(field, best_sentence, best_sentence, confidence))

        return candidates


def _best_match(
    sentences: list[str], sentence_vectors: list[list[float]], field_vector: list[float]
) -> tuple[str, float]:
    best_sentence, best_similarity = sentences[0], -1.0
    for sentence, vector in zip(sentences, sentence_vectors, strict=True):
        similarity = sum(a * b for a, b in zip(vector, field_vector, strict=True))
        if similarity > best_similarity:
            best_sentence, best_similarity = sentence, similarity
    return best_sentence, best_similarity
