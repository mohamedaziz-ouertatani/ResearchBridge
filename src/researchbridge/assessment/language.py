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
one-line heuristic.

The original version of this module stopped there, on the reasoning that
Latin-script non-English ideas (French, Spanish, German, ...) overlap
enough with English in the embedder's subword vocabulary that the failure
mode "wasn't observed for them". Live testing on 2026-09-09 observed it:
a French dermoscopy idea retrieved Spanish-language papers as its closest
matches and drove the dimension extractor to invent the non-word
"Entrenchement" - the same confidently-wrong shape the Arabic case
showed. is_likely_non_english() below extends the caveat to those,
using a function-word frequency comparison rather than a language-ID
dependency, staying within this module's original "cheap heuristic, no
new model" constraint.
"""

from __future__ import annotations

import re
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


# Function words are the cheapest available language signal: they are
# high-frequency, closed-class, and largely disjoint between these
# languages, so a short abstract carries plenty of them. Content words
# are deliberately ignored - technical vocabulary ("convolutional",
# "dermoscopic") is near-identical across European languages and would
# only add noise.
_ENGLISH_FUNCTION_WORDS = frozenset(
    """a an the and or but if then than that this these those of in on at to for from with without
    by as is are was were be been being we our us it its their they them he she his her you your
    not no can could may might must shall should will would have has had do does did which who whom
    where when while into over under between among during about against through above below
    """.split()
)

_OTHER_LANGUAGE_FUNCTION_WORDS = frozenset(
    """
    nous vous ils elles notre nos votre vos leur leurs une des les du de la le et ou mais si donc
    dans sur pour avec sans par chez entre sous vers depuis pendant afin ainsi cette ces cet est
    sont etait etaient etre avoir avons avez ont plus moins tres aussi meme quel quelle quels
    nosotros nuestro nuestra nuestros los las el ella ellos ellas un una unos unas y o pero si
    porque para con sin por sobre entre desde hasta durante entonces este esta estos estas ese esa
    es son era eran ser estar tiene tienen muy tambien mismo cual cuales entre entrenadas partir
    wir uns unser unsere ihr ihre der die das des dem den ein eine einen einem einer und oder aber
    wenn dann als dass diese dieser dieses jener durch fuer mit ohne bei zwischen unter ueber nach
    ist sind war waren sein haben hat hatte werden wird sehr auch selbst welche
    noi nostro nostra nostri gli lo la le un uno una del della dei delle e o ma se perche per con
    senza su tra fra questo questa questi queste quello quella sono era erano essere avere molto
    nos nosso nossa nossos umas uns uma os as do da dos das em no na nos nas ao aos pelo pela
    """.split()
)

_MIN_FUNCTION_WORDS = 4

# Arabizi (Latin-script chat transliteration of Arabic, common in informal
# Tunisian/Maghrebi writing) has no standard orthography, so a word list
# the way _OTHER_LANGUAGE_FUNCTION_WORDS covers French/Spanish/etc. isn't
# practical. What IS consistent across writers is reusing digits that look
# like the Arabic letter they replace (3=ain, 7=Ha, 9=qaf, ...) inside an
# otherwise-lowercase word, e.g. "na3mel" (I do), "eb3ath" (send). That
# digit-between-letters shape is what code-switched idea text (English
# function words plus Arabizi content words) slips past the function-word
# check above: the English words are real English, so english>=other and
# the text reads as plain English even though most of it isn't.
#
# Requiring lowercase on BOTH sides of the digit is what keeps this from
# firing on chemical formulas/model names, which conventionally put an
# uppercase element symbol or letter right after the digit (H2O, Pd2Cl2,
# GPT2) rather than another lowercase letter.
_ARABIZI_TOKEN_PATTERN = re.compile(r"[a-z][2356789][a-z]")
_MIN_ARABIZI_TOKENS = 2


def _has_arabizi_chat_numerals(text: str) -> bool:
    matches = sum(1 for word in text.split() if _ARABIZI_TOKEN_PATTERN.search(word))
    return matches >= _MIN_ARABIZI_TOKENS


def is_likely_non_english(text: str) -> bool:
    """True when the idea text is probably not English, by any of three
    signals: a majority non-Latin script, Latin script whose function
    words look more like another European language than English, or
    Arabizi chat-numeral spelling code-switched into otherwise-English
    text (see _has_arabizi_chat_numerals's own docstring - found live
    2026-09-09: neither of the other two signals catches this case, since
    the script is Latin and the English words present are genuinely
    English).

    Deliberately conservative. It requires _MIN_FUNCTION_WORDS matches
    before judging by that signal at all, and demands a strict majority,
    so an English idea dense with technical terms and light on function
    words stays unflagged rather than being guessed at from too little
    signal - the same "don't guess from thin evidence" stance
    is_likely_non_latin_script takes with its own length floor."""
    if is_likely_non_latin_script(text):
        return True

    if _has_arabizi_chat_numerals(text):
        return True

    words = [w.strip(".,;:!?()[]\"'").casefold() for w in text.split()]
    english = sum(1 for w in words if w in _ENGLISH_FUNCTION_WORDS)
    other = sum(1 for w in words if w in _OTHER_LANGUAGE_FUNCTION_WORDS)
    if english + other < _MIN_FUNCTION_WORDS:
        return False
    return other > english
