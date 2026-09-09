"""Rejects candidate evidence quotes that are shape-malformed - too short,
cut off mid-sentence/mid-word, or non-content boilerplate (copyright lines,
author/correspondence blocks, bare section headings) - independent of
extraction/validation.py's claim-TYPE language check and
extraction/pipeline.py's exact-substring grounding check, neither of which
inspects a quote's own shape. A truncated PDF-line-wrap fragment like "We
propose DU-" is a valid grounded substring and reads like real method
language, so it passes both of those checks trivially; this module is the
one place that asks "is this actually a complete, real sentence" before a
candidate quote is accepted.

Applied by extraction/heuristic.py and extraction/semantic.py at the point
a candidate sentence is chosen, before it becomes a ClaimCandidate - a
rejected sentence is simply skipped in favor of the next-best candidate (or
no candidate at all), never patched or truncated further, matching this
package's "no candidate is better than a fabricated/malformed one" rule.
"""

from __future__ import annotations

import re

MIN_QUOTE_TOKENS = 4

_TERMINAL_PUNCTUATION_RE = re.compile(r'[.!?][)\]"”’\']*\s*$')
_HYPHEN_LINE_WRAP_RE = re.compile(r"[A-Za-z]-\s*$")
_COPYRIGHT_RE = re.compile(r"©|\ball rights reserved\b|\bpublisher copyright\b", re.IGNORECASE)
_CORRESPONDENCE_RE = re.compile(r"\bcorrespondence to\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_VERB_HINT_RE = re.compile(
    r"\b(is|are|was|were|be|been|has|have|had|do|does|did|can|could|will|would|"
    r"propose[sd]?|present[sed]?|show[sn]?|introduce[sd]?|use[sd]?|find[s]?|found|"
    r"achieve[sd]?|demonstrate[sd]?|report[sed]?|study|studies|studied|investigate[sd]?)\b",
    re.IGNORECASE,
)
_NUMBERED_HEADING_RE = re.compile(r"^\s*\d+(\.\d+)*\.?\s*$")

# Abbreviations whose trailing period is NOT a sentence end. A quote
# ending on one of these was cut mid-sentence by the sentence splitter,
# but still satisfies _TERMINAL_PUNCTUATION_RE above and so used to pass.
# Found live 2026-09-09 in a real report, where the risks section showed
# "However, YOLO26 achieved significantly superior specificity (96.1% vs."
# as a standalone risk of the user's idea.
# A sentence that ends on its FIRST enumerated item was cut before the
# list it promised. Found live 2026-09-09: a research gap read "...
# motivates two concrete directions for future work: (i)." - terminal
# punctuation and balanced parentheses meant every other check passed.
#
# Requires a colon earlier in the text, so a sentence that merely ends on a
# parenthetical - "...retrieval-augmented generation (RAG)." - is untouched;
# it is the promise of a list followed by nothing that marks the cut. The
# marker itself is matched loosely (any 1-3 characters) because papers
# enumerate with ASCII "i"/"1"/"a" and with mathematical-italic codepoints
# alike.
_CUT_AT_ENUMERATION_RE = re.compile(r":\s*\S{0,40}?\(.{1,3}\)\s*\.?\s*$", re.DOTALL)

_TRAILING_ABBREVIATION_RE = re.compile(
    r"(?:^|[\s(\[])(?:"
    r"vs|cf|resp|approx|ca|etc|e\.g|i\.e|et\s+al|Fig|Figs|Eq|Eqs|Ref|Refs|Sec|Tab|No|Nos|"
    r"Dr|Prof|Mr|Mrs|Ms|St|Inc|Ltd|Co|Vol|pp|al"
    r")\.$",
    re.IGNORECASE,
)


def looks_truncated(text: str) -> bool:
    """True if `text` was clearly cut mid-sentence.

    Narrower on purpose than is_acceptable_quote below, which also
    enforces a minimum length, demands terminal punctuation, and screens
    out boilerplate/headings. This predicate is for text that has ALREADY
    passed the extraction gate and is about to be rendered as a standalone
    quote, so re-imposing the full extraction policy there would drop good
    material.

    Deliberately does NOT treat "no terminal punctuation" as a cut. That
    rule belongs at extraction time, where it helps pick a well-formed
    sentence out of many candidates. Applied at render time it is
    destructive: 15.1% of this corpus's 5,370 stored research_gap claims
    are complete sentences whose final period was lost in extraction
    ("Recommendations are provided for future research and institutional
    integration"), measured 2026-09-09, and rejecting them discards real
    signal. A dangling bracket, a trailing abbreviation or a cut
    enumeration are structural evidence of a cut; missing punctuation is
    not. Those structural signals reject 1.4% of the same claims.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if _HYPHEN_LINE_WRAP_RE.search(stripped):
        return True
    if _TRAILING_ABBREVIATION_RE.search(stripped):
        return True
    if _CUT_AT_ENUMERATION_RE.search(stripped):
        return True
    # An opened bracket that never closes means the sentence was cut
    # inside it, even when a period happens to follow.
    for opener, closer in (("(", ")"), ("[", "]")):
        if stripped.count(opener) > stripped.count(closer):
            return True
    return False


def is_acceptable_quote(text: str) -> bool:
    """True if `text` looks like a real, complete sentence worth showing as
    evidence - false if it's a fragment, a boilerplate/metadata line, or a
    bare heading. Never mutates `text`; callers reject the candidate
    outright rather than trying to repair it."""
    stripped = text.strip()
    if not stripped:
        return False

    tokens = stripped.split()
    if len(tokens) < MIN_QUOTE_TOKENS:
        return False

    if looks_truncated(stripped):
        return False

    # Extraction-time only, deliberately NOT part of looks_truncated - see
    # that function's docstring on why render-time filtering must not use
    # this rule.
    if not _TERMINAL_PUNCTUATION_RE.search(stripped):
        return False

    if _COPYRIGHT_RE.search(stripped) or _CORRESPONDENCE_RE.search(stripped):
        return False

    # An email address is most of a short line -> author/contact block, not
    # a sentence someone would cite as evidence. Long sentences that merely
    # happen to mention an email in passing are not rejected by this alone.
    email_match = _EMAIL_RE.search(stripped)
    if email_match and len(email_match.group()) > 0.3 * len(stripped):
        return False

    # A bare numbered section marker ("8.1.") with nothing else.
    if _NUMBERED_HEADING_RE.match(stripped):
        return False

    if looks_like_heading(stripped):
        return False

    return True


# Function words a title legitimately leaves lowercase ("Limitations AND
# Future Work"). They are excluded from the capitalized-word ratio below
# rather than counted as evidence of running prose - counting them is what
# let "Limitations and Future Work." through as a quotable sentence, since
# one lowercase word out of four drops the ratio to 0.75.
_TITLE_LOWERCASE_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "nor", "of", "on",
     "or", "the", "to", "up", "via", "with", "versus", "vs"}
)

# A leading section number ("3.2", "9.") before the heading text itself.
_LEADING_SECTION_NUMBER_RE = re.compile(r"^\s*\d+(\.\d+)*\.?\s+")

MAX_HEADING_WORDS = 12


def looks_like_heading(text: str) -> bool:
    """True if `text` is a section heading rather than a sentence.

    Title-Case-with-no-verb, applied per line so stacked headings are
    caught too. Found live 2026-09-09: an assessment reported its research
    gap as "Limitations and Future Work / Limited Distance Metrics.", two
    headings and no gap statement, because extraction had captured a
    section boundary rather than prose.

    Every line must look like a heading for the whole text to count as
    one, so a heading followed by a real sentence is NOT rejected - that
    text still contains substance worth quoting.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    for line in lines:
        line = _LEADING_SECTION_NUMBER_RE.sub("", line)
        words = re.findall(r"[A-Za-z]+", line)
        # A long line is prose even if heavily capitalized; a heading is short.
        if not words or len(words) > MAX_HEADING_WORDS:
            return False
        if _VERB_HINT_RE.search(line):
            return False
        significant = [w for w in words if w.lower() not in _TITLE_LOWERCASE_WORDS]
        if not significant:
            return False
        capitalized = sum(1 for w in significant if w[0].isupper())
        if capitalized / len(significant) < 0.8:
            return False

    return True
