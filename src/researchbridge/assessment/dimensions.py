"""Deterministic idea-dimension extraction (RAKE-style keyword extraction).

An "idea dimension" is a short noun phrase drawn VERBATIM from the input
text - never invented, paraphrased, or looked up in a fixed vocabulary.
See docs/superpowers/plans/2026-08-29-dimension-aware-assessment-coverage.md
("What 'idea dimension' means here") for why RAKE was chosen over reusing
HybridExtractor, free-form n-gram+embedding clustering, or a fixed keyword
list: it's deterministic, explainable without per-domain tuning, and needs
no new dependency.

Standard RAKE (Rose et al. 2010): split text into candidate phrases at
stopword/punctuation boundaries, score each word by degree(word)/
frequency(word) (co-occurrence with other content words vs. how often it
appears alone), score each phrase as the sum of its words' scores, rank
phrases by score, and drop phrases that are pure substrings of an
already-kept higher-scoring phrase (so "fraud" doesn't survive alongside
"financial fraud").

Deliberately NOT tuned against any specific idea's expected output - the
stopword list is a generic, mid-sized English function-word list, not
hand-picked for the fraud/federated-learning worked example.

Spot-checked against several real ideas (plan Task 12), not just the one
worked example: the fraud/federated-learning idea split cleanly into 7
sensible dimensions (privacy-preserving federated learning system,
detecting financial fraud, multiple banks, sharing raw transaction data,
robust, concept drift, highly imbalanced fraud classes); "real-time fraud
detection with graph transformers" split into 2; an LLM-agent-planning
idea split into 4. No filler phrases like "we propose" or "this paper"
showed up as a dimension in any of these - the stopword list holds up
without per-example tuning.

One real limitation found: a short phrase with no internal stopwords at
all (e.g. "quantum-assisted cat chess strategy optimization") stays a
single, overly broad candidate, since RAKE only splits at stopword/
punctuation boundaries. See coverage.py's DIMENSION_MATCH_SIMILARITY
docstring for how that showed up downstream (a spurious "established"
coverage reading) and why it's left as a known limitation rather than
patched by a length cap that would risk splitting genuinely meaningful
short technical phrases elsewhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_PHRASE_WORDS = 5

_STOPWORDS = frozenset(
    """
    a an the and or but if then than so because of to in on for with without
    by at from as is are was were be been being this that these those it
    its we our you your they their them he she his her him us me can
    could should would
    will shall may might must do does did not no nor while across using use
    via into onto up down over under between among each all any some such
    which what who whom whose when where why how also yet both either
    neither more most much many several own same other only just about
    again further once here there all any both each few more most other
    some such nor too very s t don now build remain remaining provide
    providing based upon
    """.split()
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")

# A gap between two consecutive word tokens counts as a phrase boundary
# (same as a stopword - see _candidate_phrases) if it contains any
# character besides whitespace. _TOKEN_RE itself never captures punctuation,
# so without this a run of otherwise-unrelated words separated only by a
# citation marker or a comma - never by a stopword - silently fuses into
# one nonsensical candidate phrase: "We propose a method (see [1]) that,
# unlike..." merges "method" and "see" into "method see", since the
# bracketed "[1])" simply vanishes from the token stream and no stopword
# sits between them. This restores the module's own documented RAKE
# behavior ("split text into candidate phrases at stopword/punctuation
# boundaries") - the code previously implemented only the stopword half.
_PUNCTUATION_GAP_RE = re.compile(r"[^\w\s-]")


@dataclass
class IdeaDimension:
    label: str


def extract_dimensions(text: str, max_dimensions: int = 8) -> list[IdeaDimension]:
    if not text or not text.strip():
        return []

    candidates = _candidate_phrases(text)
    if not candidates:
        return []

    word_score = _score_words(candidates)
    scored = [(phrase, sum(word_score[w.lower()] for w in phrase.split())) for phrase in candidates]

    # stable order: highest score first, ties broken by first appearance
    order_index = {phrase: i for i, phrase in enumerate(candidates)}
    scored.sort(key=lambda item: (-item[1], order_index[item[0]]))

    selected: list[str] = []
    for phrase, _score in scored:
        lowered = phrase.lower()
        if any(lowered in kept.lower() or kept.lower() in lowered for kept in selected):
            continue
        selected.append(phrase)
        if len(selected) >= max_dimensions:
            break

    # re-order by first appearance in the text for a readable, stable report
    selected.sort(key=lambda phrase: order_index[phrase])
    return [IdeaDimension(label=phrase) for phrase in selected]


def _candidate_phrases(text: str) -> list[str]:
    phrases: list[str] = []
    current: list[str] = []
    pos = 0
    for match in _TOKEN_RE.finditer(text):
        gap = text[pos : match.start()]
        pos = match.end()
        if current and _PUNCTUATION_GAP_RE.search(gap):
            phrases.extend(_split_to_max_length(current))
            current = []
        token = match.group()
        if token.lower() in _STOPWORDS:
            if current:
                phrases.extend(_split_to_max_length(current))
                current = []
            continue
        current.append(token)
    if current:
        phrases.extend(_split_to_max_length(current))
    # de-duplicate identical phrases (case-insensitive), keep first occurrence
    seen: set[str] = set()
    unique: list[str] = []
    for phrase in phrases:
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            unique.append(phrase)
    return unique


def _split_to_max_length(run: list[str]) -> list[str]:
    """A stopword-bounded run longer than MAX_PHRASE_WORDS is still one
    candidate phrase (RAKE doesn't cap phrase length) - but very long runs
    (rare; usually a sentence with no stopwords at all) are chopped into
    fixed windows so one candidate can't swallow half the input."""
    if len(run) <= MAX_PHRASE_WORDS:
        return [" ".join(run)]
    return [" ".join(run[i : i + MAX_PHRASE_WORDS]) for i in range(0, len(run), MAX_PHRASE_WORDS)]


def _score_words(candidates: list[str]) -> dict[str, float]:
    frequency: dict[str, int] = {}
    degree: dict[str, int] = {}
    for phrase in candidates:
        words = phrase.lower().split()
        length = len(words)
        for word in words:
            frequency[word] = frequency.get(word, 0) + 1
            degree[word] = degree.get(word, 0) + length
    return {word: degree[word] / frequency[word] for word in frequency}
