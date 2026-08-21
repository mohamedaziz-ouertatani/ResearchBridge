"""Baseline Extractor (Sec 28/29): cue-phrase sentence matching over the abstract.

Deliberately naive - the extraction analogue of the retrieval package's
TF-IDF/BM25 baselines (see retrieval/tfidf.py, bm25.py). The point of a
baseline is to be simple enough to establish a floor that a real method
(a local LLM, later) has to beat, not to be a good extractor. Precision
here is expected to be mediocre; that's the honest starting number Sec 28's
evaluation is meant to produce, not a defect to paper over.

For each Sec 28 field, split the abstract into sentences and take the
first one matching an ordered list of cue phrases. No cue match -> no
candidate for that field, never a fabricated guess (same "leave it blank"
rule used throughout the benchmark annotations themselves - see
benchmark/annotation_template.py). "problem" is the one exception: most
abstracts open by stating the problem/motivation before any of the other
cue phrases appear, so it falls back to the first sentence at low
confidence when no stronger cue phrase matches.

Every candidate's evidence_quote is the matched sentence verbatim, so it
is grounded in the abstract by construction - the pipeline's grounding
check (extraction/pipeline.py::_quote_is_grounded) will always pass.
"""

from __future__ import annotations

import re

from researchbridge.db.models import Paper
from researchbridge.extraction.base import ClaimCandidate

HEURISTIC_MODEL_VERSION = "cue-phrase-v1"

# Ordered by specificity: first matching phrase wins per field, so a more
# distinctive phrase (e.g. "our contribution") should sit before a vaguer
# one (e.g. "we show") that could just as easily belong to another field.
_CUE_PHRASES: dict[str, list[str]] = {
    "research_question": [
        "we ask whether", "we investigate whether", "this raises the question",
        "we study whether", "we study how", "the question of", "whether it is possible",
    ],
    "method": [
        "we propose", "we present", "we introduce", "our approach", "our method",
        "we develop", "we design", "we build",
    ],
    "main_contribution": [
        "our contribution", "our main contribution", "to the best of our knowledge",
        "for the first time", "we are the first", "we show that", "we demonstrate that",
    ],
    "dataset": [
        "we collect", "we release", "we construct a dataset", "trained on", "evaluated on",
        "benchmark dataset", "our dataset",
    ],
    "results": [
        "results show", "our results", "experiments show", "experimental results",
        "we achieve", "outperforms", "improves over",
    ],
    "limitations": [
        "however,", "a limitation", "does not", "fails to", "remains challenging",
        "is limited to", "future work",
    ],
    "applications": [
        "can be applied to", "is applicable to", "useful for", "in applications such as",
        "real-world applications",
    ],
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


class HeuristicExtractor:
    extraction_method = "heuristic"
    model_version = HEURISTIC_MODEL_VERSION

    def extract(self, paper: Paper) -> list[ClaimCandidate]:
        if not paper.abstract or not paper.abstract.strip():
            return []

        sentences = split_sentences(paper.abstract)
        candidates: list[ClaimCandidate] = []
        used_sentences: set[str] = set()

        for field, phrases in _CUE_PHRASES.items():
            sentence = _first_matching_sentence(sentences, phrases)
            if sentence is not None:
                candidates.append(ClaimCandidate(field, sentence, sentence, confidence="medium"))
                used_sentences.add(sentence)

        # skip the fallback if the opening sentence was already claimed by a
        # real cue-phrase match above - relabeling it "problem" too would be
        # a duplicate, and quite possibly a mislabel (a method sentence isn't
        # the problem statement just because it happens to open the abstract)
        if sentences and sentences[0] not in used_sentences:
            candidates.append(ClaimCandidate("problem", sentences[0], sentences[0], confidence="low"))

        return candidates


def _first_matching_sentence(sentences: list[str], phrases: list[str]) -> str | None:
    for phrase in phrases:
        for sentence in sentences:
            if phrase in sentence.lower():
                return sentence
    return None
