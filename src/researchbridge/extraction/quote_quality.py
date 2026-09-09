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

MIN_QUOTE_TOKENS = 6

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

    if _HYPHEN_LINE_WRAP_RE.search(stripped):
        return False

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

    # Title-Case-with-no-verb heuristic for a bare section heading: most
    # words capitalized, and no recognizable verb anywhere in the line.
    words = [w for w in re.findall(r"[A-Za-z]+", stripped) if w]
    if words:
        capitalized = sum(1 for w in words if w[0].isupper())
        if capitalized / len(words) >= 0.8 and not _VERB_HINT_RE.search(stripped):
            return False

    return True
