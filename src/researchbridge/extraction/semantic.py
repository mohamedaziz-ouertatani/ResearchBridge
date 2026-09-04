"""Improvement over the heuristic baseline (Sec 26): embedding-similarity
sentence selection instead of fixed cue phrases, over the abstract or a
full-text section when one exists (Sec 46).

Still extractive, not generative - for each Sec 28 field, embed every
candidate sentence and pick whichever is closest to a description of that
field, taking the sentence itself as the claim. That keeps the same
verbatim-grounding guarantee the heuristic extractor has (the pipeline's
grounding check in extraction/pipeline.py requires an exact substring of
whatever section produced it), which a generative model answering in its
own words could not offer without weakening Sec 15's "no Grounding
Illusion" rule - not a call to make quietly while chasing a better score.

The heuristic baseline's weakest fields (research_question, applications:
0 recall) are exactly where fixed strings are the wrong tool - real
abstracts phrase a research question or an application in enormously
varied ways, and no small phrase list will catch most of them. Semantic
similarity doesn't need the wording to match, only the meaning to.

Uses the Week 6 embedding model (SentenceTransformerEmbedder / all-
MiniLM-L6-v2) - no new model, no new dependency, and it's already fast
enough for this: one batched embed_texts() call per candidate pool covers
every sentence and every field description in that pool at once.

Each sentence proposes only its own single best-matching field WITHIN ITS
OWN CANDIDATE POOL (a real bug this fixed: on short, jargon-dense
abstracts, all-MiniLM-L6-v2 can rate a fluent-but-content-free sentence -
e.g. "This report has been updated for posterity..." - higher against
several field anchors than the sentence that actually describes the
method, because it rewards grammatically ordinary prose over jargon-heavy
technical prose almost independent of topical relevance. Picking each
field's best sentence independently let that one attractive-but-wrong
sentence get duplicated across four different fields on a real paper.
Requiring each sentence to serve at most one field - whichever it matches
best - means a field with no sentence that actually prefers it abstains
instead of inheriting someone else's leftover.

"Within its own candidate pool" matters once full text is involved: two
fields that resolve to the SAME full-text section (e.g. method and
dataset both -> methods/experiments) share a real collision risk and must
be matched together; two fields resolving to different sections (or one
resolving to a section and another falling back to the abstract) have
disjoint sentence pools and can never collide with each other. extract()
groups fields by their exact resolved pool before running the matching -
see _match_pool.
"""

from __future__ import annotations

from collections import defaultdict

from researchbridge.db.models import Paper
from researchbridge.embedding.base import Embedder
from researchbridge.extraction.base import ClaimCandidate
from researchbridge.extraction.sections import sentences_for_field
from researchbridge.extraction.sentences import split_sentences

SEMANTIC_MODEL_VERSION = "semantic-v3"

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
    "research_gap": "A gap or open question the paper says remains for future work.",
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
# not tuning a threshold against the benchmark it's evaluated on. Reused
# unchanged for full-text sentences (Sec 46) - re-tuning for that
# distribution is future work once real full-text extraction data exists.
MIN_SIMILARITY = 0.15
MEDIUM_CONFIDENCE_SIMILARITY = 0.28


class SemanticExtractor:
    extraction_method = "semantic"
    model_version = SEMANTIC_MODEL_VERSION

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]:
        abstract_sentences = (
            split_sentences(paper.abstract) if paper.abstract and paper.abstract.strip() else []
        )

        # Group fields by the EXACT pool they'll draw from - same pool
        # means real collision risk, different pools can never collide.
        pool_fields: dict[str, list[str]] = defaultdict(list)
        pool_sentences: dict[str, list[str]] = {}
        pool_section: dict[str, str | None] = {}

        for field in _FIELD_QUERIES:
            full_text_pairs = sentences_for_field(field, sections)
            if full_text_pairs:
                section_name = full_text_pairs[0][1]  # one sentences_for_field call = one section
                pool_key = f"fulltext:{section_name}"
                pool_sentences.setdefault(pool_key, [s for s, _ in full_text_pairs])
                pool_section[pool_key] = section_name
            else:
                pool_key = "abstract"
                pool_sentences.setdefault(pool_key, abstract_sentences)
                pool_section[pool_key] = None
            pool_fields[pool_key].append(field)

        candidates: list[ClaimCandidate] = []
        for pool_key, fields in pool_fields.items():
            candidates.extend(self._match_pool(fields, pool_sentences[pool_key], pool_section[pool_key]))
        return candidates

    def _match_pool(
        self, fields: list[str], sentences: list[str], section_name: str | None
    ) -> list[ClaimCandidate]:
        if not sentences:
            return []

        vectors = self.embedder.embed_texts(sentences + [_FIELD_QUERIES[f] for f in fields])
        sentence_vectors, field_vectors = vectors[: len(sentences)], vectors[len(sentences) :]

        # each sentence's own best-matching field (within this pool only),
        # and its score there
        proposals: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for sentence, svec in zip(sentences, sentence_vectors, strict=True):
            similarities = {
                field: sum(a * b for a, b in zip(svec, fvec, strict=True))
                for field, fvec in zip(fields, field_vectors, strict=True)
            }
            best_field = max(similarities, key=similarities.get)
            proposals[best_field].append((sentence, similarities[best_field]))

        candidates: list[ClaimCandidate] = []
        for field in fields:
            suitors = proposals.get(field)
            if not suitors:
                continue
            sentence, similarity = max(suitors, key=lambda pair: pair[1])
            if similarity < MIN_SIMILARITY:
                continue
            confidence = "medium" if similarity >= MEDIUM_CONFIDENCE_SIMILARITY else "low"
            if section_name is not None:
                candidates.append(ClaimCandidate(field, sentence, sentence, confidence, section=section_name))
            else:
                candidates.append(ClaimCandidate(field, sentence, sentence, confidence))
        return candidates
