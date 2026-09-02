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

# Real application/use-context language - Gate 1 of a two-gate design (see
# docs/superpowers/specs/2026-09-02-applications-evidence-grounding-design.md).
# This gate asks a narrow, context-free question: does this sentence, read
# on its own, describe genuine deployment/use-case language with a
# concrete, named external complement? It deliberately does NOT try to
# detect whether the claim restates the SAME PAPER's own task - that
# requires comparing against the paper's other claims, which this
# function has no access to (it validates one candidate sentence at a
# time). That check is Gate 2, in assessment/applications.py, at
# assessment time, where the paper's other claims are available.
#
# Two real production false positives motivated the redesign of the old,
# single-regex version of this check:
#   1. "...useful for predicting performance level of students." - the
#      old _APPLICATION_LANGUAGE_RE's `useful (to|for|in) [a-z]` clause
#      accepted ANY following word, so a bare restatement of the paper's
#      own predictive task (no actor, no institution, no downstream
#      action, nothing beyond the task's own object) passed.
#   2. "...software applications for technology-oriented learning." - the
#      old regex's `applications? (such as|include|in|to|for)` matched the
#      literal substring "applications for" regardless of whether
#      "application(s)" meant "use-case" or "computer program."

# Strip the ordinary "software/mobile/web/computer application(s)" noun
# sense before matching anything else, so it can never itself satisfy the
# "applications... for/to/in" clause below.
_TECH_NOUN_APPLICATION_RE = re.compile(
    r"\b(software|mobile|web|desktop|computer|online)\s+applications?\b", re.IGNORECASE
)

# Each alternative captures its own complement (the text after the
# deployment verb) into a distinctly-named group, so the code below can
# find whichever one matched and inspect it.
_DEPLOYMENT_CLAUSE_RE = re.compile(
    r"\b(?:can|could) be (?:applied|used|deployed)\s+(?:to|for|in)\s+(?P<c1>[^.;]+)"
    r"|\bis applicable\s+(?:to|in)\s+(?P<c2>[^.;]+)"
    r"|\bapplications?\s+(?:such as|include|in|to|for)\s+(?P<c3>[^.;]+)"
    r"|\breal-world applications?\s+(?:in|for)\s+(?P<c4>[^.;]+)"
    r"|\bdeployed\s+(?:in|to|for|by)\s+(?P<c5>[^.;]+)"
    r"|\bused\s+(?:in|for|to|by)\s+(?P<c6>[^.;]+)"
    r"|\b(?:applied|applicable|targets?|targeting)\s+(?:to|for|in)\s+(?P<c7>[^.;]+)"
    r"|\buseful\s+(?:to|for|in)\s+(?P<c8>[^.;]+)",
    re.IGNORECASE,
)

# Named human/institutional actors or deployment settings - if the
# deployment verb's complement names one of these, the claim is naming
# something beyond the paper's own task.
_ACTOR_SETTING_RE = re.compile(
    r"\buniversit(y|ies)\b|\bschools?\b|\bhospitals?\b|\bclinics?\b|\bclinicians?\b"
    r"|\bphysicians?\b|\bdoctors?\b|\bnurses?\b|\bbanks?\b|\bbanking\b"
    r"|\bfinancial institutions?\b|\bcompliance teams?\b|\bregulators?\b"
    r"|\bpolicymakers?\b|\binstructors?\b|\bteachers?\b|\beducators?\b"
    r"|\badvisors?\b|\bcounselors?\b|\bpractitioners?\b|\bindustry\b"
    r"|\borganizations?\b|\bcompanies\b|\bbusinesses?\b|\benterprises?\b"
    r"|\bgovernments?\b|\bagenc(y|ies)\b|\bdecision[- ]makers?\b|\bstakeholders?\b"
    r"|\b\w+(ologists?|icians?)\b"
    r"|\bin (clinical|industrial|educational|practical|real-world) (practice|settings?|use|contexts?)\b"
    r"|\bat scale\b|\bin the field\b|\bin practice\b",
    re.IGNORECASE,
)

# Downstream actions distinct from the paper's own predictive/detection/
# classification verb - a human or institutional response taken as a
# RESULT of the system's output, not the system's own computation.
_DOWNSTREAM_ACTION_RE = re.compile(
    r"\binterventions?\b|\btriage\b|\bmanual review\b|\bcounsel(l)?ing\b"
    r"|\btreatment plan(ning)?\b|\bresource allocation\b|\bpolicy( ?making)?\b"
    r"|\bremediation\b|\bprioriti[sz](e|ation|ing)\b|\bdecision support\b"
    r"|\brisk mitigation\b|\bearly (intervention|warning)\b|\bscreening\b"
    r"|\breferrals?\b|\bflagg?ing\b|\balert(ing)?\b",
    re.IGNORECASE,
)

# A generic structural fallback: the complement contains ITS OWN
# qualifying "in X"/"for X"/"at X"/"by X" phrase beyond the deployment
# verb's own preposition - e.g. "fraud detection IN financial
# transactions", "drug discovery pipelines IN pharmaceutical R&D". Deliberately
# excludes "on" (method/training language routinely says "trained on X").
_QUALIFYING_CONTEXT_RE = re.compile(r"\b(in|for|at|by|within|across|among)\s+[a-z]", re.IGNORECASE)

# Known-vague qualifiers that would otherwise satisfy _QUALIFYING_CONTEXT_RE
# without naming anything concrete - "in general" contains "in general[a-z]"
# but names no real context.
_VAGUE_QUALIFIER_RE = re.compile(
    r"\bin general\b|\bfor future (work|studies|research)\b|\bin the future\b"
    r"|\bfor general purposes\b|\bfor further research\b|\bin future\b",
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
    tier matched); "strong" for an accepted applications claim (there is
    no "weak" applications tier - hedged/generic language is rejected
    outright, not accepted-but-flagged); None for every other claim_type,
    and None for any rejected claim. Persisted downstream
    (extracted_claims.validation_tier)."""


def _has_result_signal(text: str) -> bool:
    return bool(_RESULT_METRIC_RE.search(text) or _ACHIEVEMENT_VERB_RE.search(text))


def _has_application_signal(text: str) -> bool:
    masked = _TECH_NOUN_APPLICATION_RE.sub(" ", text)
    match = _DEPLOYMENT_CLAUSE_RE.search(masked)
    if not match:
        return False
    complement = next(g for g in match.groupdict().values() if g)
    if _ACTOR_SETTING_RE.search(complement) or _DOWNSTREAM_ACTION_RE.search(complement):
        return True
    if _VAGUE_QUALIFIER_RE.search(complement):
        return False
    return bool(_QUALIFYING_CONTEXT_RE.search(complement))


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
                False,
                "no concrete deployment context (actor, institution, downstream action, or "
                "named external setting) found; reads as method, result, restated task, or "
                "generic/vague text",
            )
        return ValidationResult(True, tier="strong")

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
