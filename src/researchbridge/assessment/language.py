"""Non-Latin-script heuristic used to caveat assessments of non-English ideas.

This project's embedder (all-MiniLM-L6-v2, see coverage.py's docstring) is
not meaningfully multilingual, and the corpus itself is overwhelmingly
English. A manual test against an Arabic idea ("a new method for using
neural networks to detect fake news in Arabic text") retrieved an unrelated
Arabic paper on first-order formal logic as its closest match - the
retrieval was clustering on "same script, mentions AI" rather than real
topical similarity, so the resulting novelty/feasibility verdicts looked
confidently wrong rather than visibly uncertain.

No language-detection dependency added for this - script-mismatch is a
narrow, cheap signal that catches the failure mode actually observed
(non-Latin-script ideas), without pulling in a model or library for a
one-line heuristic. Non-English ideas written in Latin script (French,
Spanish, German, ...) aren't flagged - the embedder's subword vocabulary
overlaps enough with English there that this specific failure mode wasn't
observed for them.
"""

from __future__ import annotations

import unicodedata

_MIN_LETTERS = 20
_LATIN_FRACTION_THRESHOLD = 0.5


def _is_latin_letter(ch: str) -> bool:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False
    return name.startswith("LATIN")


def is_likely_non_latin_script(text: str) -> bool:
    """True when the idea text is long enough to judge and is majority
    non-Latin-script - the corpus/embedder combination this project uses is
    least reliable there. Short inputs are left unflagged rather than
    guessed at from too little signal."""
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < _MIN_LETTERS:
        return False
    latin = sum(1 for ch in letters if _is_latin_letter(ch))
    return (latin / len(letters)) < _LATIN_FRACTION_THRESHOLD
