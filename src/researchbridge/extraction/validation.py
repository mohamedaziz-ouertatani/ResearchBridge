"""Claim-type semantic validation (blueprint Sec 34).

The pipeline's existing grounding check (pipeline.py::_quote_is_grounded)
only proves an evidence quote is not fabricated - that it really appears
in the paper. It says nothing about whether the quote actually expresses
the semantic category (claim_type) it was filed under. Both extractors
label a claim by whichever fixed phrase/anchor happened to match best
(heuristic.py's cue-phrase list, semantic.py's embedding-similarity
argmax against one anchor sentence per field) and neither checks that the
winning sentence actually reads like that category. semantic.py's own
docstring documents the concrete failure mode: a jargon-dense results
sentence ("achieves 94% AUC") can win an unrelated field's anchor on a
short abstract precisely because embedding similarity rewards
grammatically ordinary prose almost independent of topical relevance -
this is how a results sentence ends up stored as a "research_gap".

This module is the second, independent gate the pipeline runs right
after grounding, in extraction/pipeline.py::_process_paper: lexical/regex
signal detection over the claim text itself, not an LLM call and not a
relabeling. A claim is accepted only if the language it actually contains
is consistent with its claimed type; otherwise it is rejected (logged to
extraction_errors, exactly like an ungrounded quote) rather than silently
persisted or silently relabeled - the report must never present a
misclassified claim as if it were correctly classified.

Deliberately narrow in scope, matching the reported failure modes:
VALIDATABLE_CLAIM_TYPES covers only the six fields the bug report named
(research_gap, applications, limitations, results, method, problem).
research_question/dataset/main_contribution (and the stub extractor's
"contribution") have no reported misclassification and no natural
lexicon to check against - inventing a weak one just to say "we validate
everything" would trade real false negatives now for imagined ones
later. Those types are accepted unconditionally: unvalidated, not
"validated ok".

Deliberately conservative in direction, per Sec 8's research-gap rule
("results/achievements must never be accepted as research gaps simply
because the upstream extractor labeled them that way"): every rule here
is a REQUIRE-the-right-language check, not a set of ad hoc keyword
blocklists layered onto extraction - it asks the same question every
validated field: does the evidence text actually contain the kind of
language that field's category requires, in a paper's own words?
"""

from __future__ import annotations

import re
from dataclasses import dataclass

VALIDATABLE_CLAIM_TYPES = frozenset(
    {"problem", "method", "results", "limitations", "research_gap", "applications"}
)

# Metric/achievement language - shared by "results" (required) and used as
# a disqualifying signal for "problem" (a sentence reporting a benchmark
# number is not a problem statement, however it got labeled).
_RESULT_METRIC_RE = re.compile(
    r"\d+(\.\d+)?\s*%"
    r"|\bAUC\b|\bF1[- ]?(score)?\b|\bROC(-|\s)?AUC\b"
    r"|\bp\s*[<>=]\s*0?\.\d+"
    r"|\baccuracy of\b|\bprecision of\b|\brecall of\b",
    re.IGNORECASE,
)
_ACHIEVEMENT_VERB_RE = re.compile(
    r"\boutperform(s|ed)?\b"
    r"|\bachiev(e|es|ed)\b"
    r"|\bimprove(s|d)? (over|upon)\b"
    r"|\bsurpass(es|ed)?\b"
    r"|\bstate[- ]of[- ]the[- ]art\b"
    r"|\bresults show\b|\bexperimental results\b|\bexperiments show\b|\bour results\b",
    re.IGNORECASE,
)

# "Explicit gap" language (Sec 32): the paper itself says something is
# unresolved, missing, or left for later - as distinct from a limitation
# (a weakness of THIS work) or a result (something already achieved).
# Split into a strong (unambiguous) tier and a weak (ambiguous) tier: a
# real production example caught the weak tier's failure mode directly -
# "closes 15.4% of the gap between X and Y" and "...paving the way for
# future research" are both purely results sentences that happen to
# contain "gap between"/"future research" as boilerplate, not a stated
# unresolved problem. The weak tier only counts when there's no
# competing metric/achievement signal in the same text; the strong tier
# is trusted either way, since "remains an open problem"/"future work"
# is not something a results sentence says in passing.
_STRONG_GAP_LANGUAGE_RE = re.compile(
    r"\bfuture work\b|\bwe leave\b|\bwe plan to\b|\bfurther exploration\b"
    r"|\bremains? (an? )?open\b|\bopen (question|problem)\b"
    r"|\byet to be\b|\bhas not been\b|\bhave not been\b|\bno existing\b"
    r"|\black(s|ing)?( of)?\b|\bunexplored\b|\bunderexplored\b"
    r"|\bstill unknown\b|\bunaddressed\b|\bnot yet been\b"
    r"|\bremains (unclear|unresolved|unknown)\b|\bunresolved\b",
    re.IGNORECASE,
)
_WEAK_GAP_LANGUAGE_RE = re.compile(r"\bfuture research\b|\bgap (in|between)\b", re.IGNORECASE)

# Real application/use-context language - a stated context this work is
# useful *in*, not a generic claim of usefulness (Sec 5's failure mode)
# and not a method/architecture/result sentence. Voice-agnostic: real
# abstracts write this in first person ("can be applied to"), the
# benchmark's hand-annotated ground truth in third person ("the authors
# suggest this can be used for", "targets autonomous driving") - both
# describe the same thing, a named context this work is useful *in*.
_APPLICATION_LANGUAGE_RE = re.compile(
    r"\b(can|could) be (applied|used) (to|for|in)\b|\bis applicable to\b"
    r"|\bapplications? (such as|include|in|to|for)\b|\breal-world applications?\b"
    r"|\bdeployed (in|to|for)\b|\bused (in|for|to) [a-z]"
    r"|\b(applied|applicable|useful|targets?|targeting) (to|for|in) [a-z]"
    r"|\bin (clinical|industrial|practical|real-world) (practice|settings?|use|contexts?)\b"
    r"|\bpractical use\b",
    re.IGNORECASE,
)
# "useful for X" only counts as an application when X names something
# concrete, not a vague catch-all ("future studies", "future work",
# "further research", "general purposes") - otherwise almost any
# throwaway sentence would qualify.
_VAGUE_USEFULNESS_RE = re.compile(
    r"\buseful for (future (studies|work|research)|general purposes|further research)\b",
    re.IGNORECASE,
)

# Voice-agnostic, same reason as applications above: abstracts say "we
# propose"/"our approach", the benchmark's third-person ground truth says
# "the authors propose"/"the paper introduces" - the verb/noun carries the
# signal, not the pronoun.
_METHOD_LANGUAGE_RE = re.compile(
    r"\bpropose(s|d)?\b|\bpresent(s|ed)?\b|\bintroduce(s|d)?\b|\bapproach\b|\bmethod\b"
    r"|\bdevelop(s|ed)?\b|\bdesign(s|ed)?\b|\bbuild(s)?\b|\bbuilt\b|\bbased on\b"
    r"|\btrained (using|on|with)\b|\barchitecture\b|\bframework\b|\balgorithm\b"
    r"|\btechnique\b|\bleverages?\b|\butilizes?\b",
    re.IGNORECASE,
)

_LIMITATION_LANGUAGE_RE = re.compile(
    r"\bhowever,?\b|\ba limitation\b|\bdoes not\b|\bfails? to\b|\bremains? challenging\b"
    r"|\bis limited to\b|\bweakness(es)?\b|\blimitation(s)? of\b|\bshortcoming(s)?\b|\bdrawback(s)?\b"
    r"|\bcannot\b|\bunable to\b|\bdisadvantage(s)?\b|\bimperfect\b|\bexcludes?\b"
    r"|\bfocuses? only on\b|\bbias(ed)?\b|\brestricted to\b|\bonly (works|works well|applies) (on|to|for)\b"
    r"|\bstruggles? (with|to)\b|\bnot generalize\b|\bdifficult(y|ies)\b to (compare|reproduce|evaluate)\b",
    re.IGNORECASE,
)

_PROBLEM_LANGUAGE_RE = re.compile(
    r"\bchallenge\b|\bproblem (of|is)\b|\bdifficult(y|ies)\b|\bissue (of|is)\b|\btask of\b"
    r"|\bmotivated by\b|\bremains? (a|an) (challenge|open)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    reason: str | None = None
    tier: str | None = None
    """"strong" or "weak" for an accepted research_gap claim (which regex
    tier matched) - None for every other claim_type, and None for any
    rejected claim. Persisted downstream (extracted_claims.validation_tier)
    so gap clustering can require an unambiguous anchor claim rather than
    treating "future research" boilerplate the same as "remains unresolved"."""


def _has_result_signal(text: str) -> bool:
    return bool(_RESULT_METRIC_RE.search(text) or _ACHIEVEMENT_VERB_RE.search(text))


def _has_application_signal(text: str) -> bool:
    if _VAGUE_USEFULNESS_RE.search(text):
        return False
    return bool(_APPLICATION_LANGUAGE_RE.search(text))


def validate_claim_type(claim_type: str, text: str) -> ValidationResult:
    """Does `text` actually express `claim_type`, by lexical signal - see
    module docstring for why this is regex/lexicon-based rather than an
    LLM call or a relabeling step. Types outside VALIDATABLE_CLAIM_TYPES
    are always accepted (unvalidated, not "confirmed correct")."""
    if claim_type not in VALIDATABLE_CLAIM_TYPES:
        return ValidationResult(is_valid=True)

    if claim_type == "research_gap":
        if _STRONG_GAP_LANGUAGE_RE.search(text):
            return ValidationResult(True, tier="strong")
        if _WEAK_GAP_LANGUAGE_RE.search(text) and not _has_result_signal(text):
            return ValidationResult(True, tier="weak")
        return ValidationResult(
            False, "no unresolved-problem/future-work language found; not a stated research gap"
        )

    if claim_type == "applications":
        if not _has_application_signal(text):
            return ValidationResult(
                False, "no concrete application/use-context language found; reads as method, result, or generic text"
            )
        return ValidationResult(True)

    if claim_type == "results":
        if not _has_result_signal(text):
            return ValidationResult(
                False, "no metric or achievement language found; does not read as a results statement"
            )
        return ValidationResult(True)

    if claim_type == "limitations":
        if not _LIMITATION_LANGUAGE_RE.search(text):
            return ValidationResult(False, "no limitation/weakness language found")
        return ValidationResult(True)

    if claim_type == "method":
        if not _METHOD_LANGUAGE_RE.search(text):
            return ValidationResult(False, "no method/approach language found")
        return ValidationResult(True)

    if claim_type == "problem":
        if _has_result_signal(text) and not _PROBLEM_LANGUAGE_RE.search(text):
            return ValidationResult(False, "reads as a results statement, not a problem statement")
        return ValidationResult(True)

    return ValidationResult(True)  # pragma: no cover - unreachable, VALIDATABLE_CLAIM_TYPES is exhaustive above
