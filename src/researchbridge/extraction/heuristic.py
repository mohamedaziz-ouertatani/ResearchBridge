"""Baseline Extractor (Sec 28/29): cue-phrase sentence matching over the
abstract, or a full-text section when one exists and has a match (Sec 46).

Deliberately naive - the extraction analogue of the retrieval package's
TF-IDF/BM25 baselines (see retrieval/tfidf.py, bm25.py). The point of a
baseline is to be simple enough to establish a floor that a real method
(a local LLM, later) has to beat, not to be a good extractor. Precision
here is expected to be mediocre; that's the honest starting number Sec 28's
evaluation is meant to produce, not a defect to paper over.

For each Sec 28 field, first try sentences_for_field(field, sections) (the
paper's own most-likely full-text section for that field, per
FIELD_SECTIONS priority order) - a cue-phrase hit there gets
confidence="high" (Sec 15: "explicit statement in a named section"). If
that's empty (no full text, or no section recognized), fall back to
searching paper.abstract exactly as before Sec 46, confidence="medium". No
cue match anywhere -> no candidate for that field, never a fabricated
guess. "problem" is the one exception: most abstracts open by stating the
problem/motivation before any of the other cue phrases appear, so it falls
back to the first abstract sentence at low confidence when no stronger cue
phrase matches - this fallback is abstract-only by design (see
FIELD_SECTIONS's docstring), full text is never consulted for it.

Every candidate's evidence_quote is a verbatim sentence from wherever it
was matched, so it is grounded by construction - the pipeline's grounding
check (extraction/pipeline.py::_quote_is_grounded) will always pass.
"""

from __future__ import annotations

from researchbridge.db.models import Paper
from researchbridge.extraction.base import ClaimCandidate
from researchbridge.extraction.sections import sentences_for_field
from researchbridge.extraction.sentences import split_sentences

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
        "is limited to",
    ],
    "applications": [
        "can be applied to", "is applicable to", "useful for", "in applications such as",
        "real-world applications",
    ],
    # Sec 32's "explicit gaps": a gap the paper states remains open, distinct
    # from limitations (a weakness of the current work) - "future work" used
    # to sit under limitations, which conflated "what's wrong with this"
    # with "what's left to do next".
    "research_gap": [
        "future work", "in future work", "we leave", "remains an open",
        "remains open", "an open question", "open problem", "yet to be explored",
        "we plan to", "future research", "further exploration",
    ],
}

class HeuristicExtractor:
    extraction_method = "heuristic"
    model_version = HEURISTIC_MODEL_VERSION

    def extract(self, paper: Paper, sections: dict[str, str]) -> list[ClaimCandidate]:
        abstract_sentences = (
            split_sentences(paper.abstract) if paper.abstract and paper.abstract.strip() else []
        )

        candidates: list[ClaimCandidate] = []
        used_abstract_sentences: set[str] = set()

        for field, phrases in _CUE_PHRASES.items():
            full_text_pairs = sentences_for_field(field, sections)
            if full_text_pairs:
                match = _first_matching_pair(full_text_pairs, phrases)
                if match is not None:
                    sentence, section_name = match
                    candidates.append(
                        ClaimCandidate(field, sentence, sentence, confidence="high", section=section_name)
                    )
                    continue  # full text produced a candidate - it wins over the abstract (Sec 46)

            sentence = _first_matching_sentence(abstract_sentences, phrases)
            if sentence is not None:
                candidates.append(ClaimCandidate(field, sentence, sentence, confidence="medium"))
                used_abstract_sentences.add(sentence)

        # skip the fallback if the opening sentence was already claimed by a
        # real cue-phrase match above - relabeling it "problem" too would be
        # a duplicate, and quite possibly a mislabel (a method sentence isn't
        # the problem statement just because it happens to open the abstract)
        if abstract_sentences and abstract_sentences[0] not in used_abstract_sentences:
            candidates.append(
                ClaimCandidate("problem", abstract_sentences[0], abstract_sentences[0], confidence="low")
            )

        return candidates


def _first_matching_sentence(sentences: list[str], phrases: list[str]) -> str | None:
    for phrase in phrases:
        for sentence in sentences:
            if phrase in sentence.lower():
                return sentence
    return None


def _first_matching_pair(pairs: list[tuple[str, str]], phrases: list[str]) -> tuple[str, str] | None:
    for phrase in phrases:
        for sentence, section_name in pairs:
            if phrase in sentence.lower():
                return sentence, section_name
    return None
