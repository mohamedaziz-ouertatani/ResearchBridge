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
#
# The strong tier's own trust-it-either-way stance is exactly why
# "open (question|problem)" needs its own narrower guard (2026-09-04
# investigation, found live-testing the assessment pipeline with real
# ideas): it matched "six existing OPEN QUESTION ANSWERING datasets" in a
# real paper describing its own benchmark contribution - a standard NLP
# task name ("open [-domain/-book/-ended] question answering"), not a
# stated gap, and the strong tier has no competing-signal check to catch
# it. Verified against the whole corpus before narrowing anything: of 169
# real "open question"/"open problem" occurrences, the word immediately
# following was "answering" in exactly the 3 that were this false
# positive (2 identical claims, "Visual Open Question Answering (Visual
# OpenQA)" and "open question answering datasets") and never for any
# other following word sampled ("in", "whether", "is", "posed", "of",
# "to", ...) - all genuine gap language. Narrow negative lookahead rather
# than a broader exclusion, to avoid guessing past what the data showed.
_STRONG_GAP_LANGUAGE_RE = re.compile(
    r"\bfuture work\b|\bwe leave\b|\bwe plan to\b|\bfurther exploration\b"
    r"|\bremains? (an? )?open\b|\bopen (question|problem)\b(?!\s+answering)"
    r"|\byet to be\b|\bhas not been\b|\bhave not been\b|\bno existing\b"
    r"|\black(s|ing)?( of)?\b|\bunexplored\b|\bunderexplored\b"
    r"|\bstill unknown\b|\bunaddressed\b|\bnot yet been\b"
    r"|\bremains (unclear|unresolved|unknown)\b|\bunresolved\b",
    re.IGNORECASE,
)
_WEAK_GAP_LANGUAGE_RE = re.compile(r"\bfuture research\b|\bgap (in|between)\b", re.IGNORECASE)

# A second, distinct false-positive shape for "gap (in|between)", found in a
# real ResearchAssessment export (2026-09-03 investigation): "continuously
# computes the gap between 'Average Daily Demand' and 'Instantaneous
# Demand'... to fill the gaps" is describing the paper's own TECHNICAL
# METRIC (a numeric difference the method computes), not a gap in the
# literature - and it has no competing metric/achievement language
# (_has_result_signal doesn't fire: no percentage, no AUC/F1, no
# "outperforms"), so it slipped through where the sibling "closes 15.4% of
# the gap between..." case is already caught. Same "gap between" phrase,
# different failure mode: there the giveaway was a co-occurring metric
# (a RESULT), here it's a co-occurring computation VERB (a METHOD
# description) - "computes/calculates/measures/determines THE gap"
# reads as "our system quantifies a difference," never as "the field
# hasn't addressed X." Deliberately narrow (four verbs, immediately
# governing "gap") rather than a broad STEM-word blocklist, to avoid
# rejecting a genuine "a gap exists between the accuracy we compute and
# what is achievable" - that phrasing doesn't match this shape.
_TECHNICAL_GAP_METRIC_RE = re.compile(
    r"\b(?:comput(?:e|es|ed|ing)|calculat(?:e|es|ed|ing)|measur(?:e|es|ed|ing)|determin(?:e|es|ed|ing))"
    r"\s+(?:the\s+)?gap\b",
    re.IGNORECASE,
)

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
#
# A first version of this rewrite required EVERY deployment-verb match to
# clear an actor/downstream-action/qualifying-context gate. Re-running the
# extraction benchmark's claim-type-validation check against the real,
# hand-annotated ground truth (rb-extract-evaluate) surfaced real
# regressions that weren't in the two reported bugs or the hand-picked
# test cases, each traced to a specific bug rather than accepted as
# "appropriately conservative":
#   - .search() only inspects the FIRST regex match in the text, so a
#     sentence with an early deployment phrase that fails the gate (e.g.
#     "used in modern integrated circuits") never got a chance to try a
#     later, genuinely qualifying one in the same sentence ("used AS
#     solid-state synapses IN neuromorphic computing circuits").
#   - a captured complement group can lose the very preposition the
#     actor-setting regex needs: "useful IN practice" - the "in" is
#     consumed by the verb clause's own non-capturing group, so the
#     complement text starts at "practice", and "\bin practice\b" can
#     never match against text that no longer contains "in".
#   - the old `[^.;]+` complement boundary truncated at the first literal
#     period, including one inside an abbreviation like "e.g." mid-
#     sentence, cutting off real qualifying context that came after it.
#   - a single verb-strength tiering ("applicable to"/"can be applied to"
#     always accepted, "useful for"/"used to" always gated) is also wrong:
#     "can be applied to predicting customer churn" is exactly as much a
#     bare task-restatement as the reported "useful for predicting..."
#     bug - the verb doesn't determine whether the complement names
#     something real. What actually distinguished the genuine real-corpus
#     cases from the restatements was ENUMERATION - "applications such as
#     A, B, C" or "applicable to Spotify, YouTube, ..." lists multiple
#     concrete items, and a bare task restatement is never phrased as a
#     list. So enumeration ("such as"/"including") is a third acceptable
#     signal alongside actor/institution and downstream-action - checked
#     the same way, against every deployment-verb match uniformly,
#     regardless of which specific verb introduced it.
# See _has_application_signal below for how these are fixed: matched with
# finditer() (not .search()) so every occurrence in the text gets a
# chance, checked against each match's own full span (not a captured
# sub-group) so a consumed preposition or enumeration marker is still
# visible, over text with common abbreviation periods normalized away
# first.

# Strip the ordinary "software/mobile/web/computer application(s)" noun
# sense before matching anything else, so it can never itself satisfy the
# "applications... for/to/in" clause below.
_TECH_NOUN_APPLICATION_RE = re.compile(
    r"\b(software|mobile|web|desktop|computer|online)\s+applications?\b", re.IGNORECASE
)

# "e.g." / "i.e." / "etc." mid-sentence would otherwise truncate a
# complement-boundary search at their own internal period.
_ABBREVIATION_RE = re.compile(r"\b(e\.g|i\.e|etc)\.", re.IGNORECASE)

# Every recognized deployment/use-case verb construction, voice-agnostic
# between first-person abstract prose ("can be applied to") and third-
# person ground-truth annotation ("is applicable to X" / "applicable to
# X"). No tiering by verb - see the module docstring above for why that
# was tried and rejected. Each alternative requires at least one
# following word so an empty/truncated complement never matches.
# Fix B (2026-09-03 investigation): "support commuters in selecting
# alternative routes" and similar genuine deployment claims were false
# negatives - "support" was not a recognized deployment verb at all.
# Added as a grammatical SHAPE ("support <actor> in <downstream action>-
# ing"), not a curated actor/action word list: the actor and the gerund
# complement are matched by word class ([a-z]+), so this recognizes any
# "support X in Y-ing" construction rather than a fixed vocabulary of
# actors ("commuters") or actions ("selecting routes"). See
# _DOWNSTREAM_ACTION_RE below for the matching self-sufficiency check.
#
# Fix C (2026-09-04, found building a claim-revalidation backfill): the
# "applications... such as X" alternative above requires the enumeration
# trigger immediately after "application(s)" - it never matches "the
# real-world applications OF KRL, such as language modeling, question
# answering..." (a genuine enumerated-applications survey sentence) or
# even the much more common "...numerous real-world applications, SUCH
# AS robotics, autonomous vehicles..." (a comma between "applications"
# and "such as" - completely ordinary written English, not an edge
# case). Both were silently rejected as "no deployment context found."
# Verified against the whole corpus before writing this (not guessed):
# sampled every currently-rejected applications claim containing "such
# as"/"including" - the majority are genuinely NOT applications
# ("algorithms including SVM, XGBoost...", "tools such as Hadoop..." -
# enumerating the paper's own METHOD components, correctly rejected, and
# specifically NOT matched by this new alternative since it requires
# "application(s)" to appear immediately before the trigger or
# immediately after one of a small closed set of prepositions
# (of/for/in) - "algorithms including X" has no "application(s)" word
# anywhere nearby, so this alternative simply never fires for it. The
# intervening noun phrase after of/for/in is bounded to 30 characters so
# this can't reach across an unrelated later "such as" clause in a long
# sentence.
#
# Fix D (2026-09-04, sampling a broader set of rejected candidates than
# Fix C looked at): "deployed" only recognized (in|to|for|by) as its
# following preposition - real corpus cases like "deployed THROUGH
# collaboration between WeBank and Extreme Vision" and "deployed VIA a
# WeChat Mini Program" never matched the verb clause at all, so the
# actor/institution/downstream-action check inside it never even ran,
# even though both name real companies/platforms. Added "via"/"through" -
# unlike "to", neither naturally introduces a gerund describing the
# system's own task ("deployed to predicting X" is not idiomatic English
# the way "applied to predicting X" is), so this doesn't reopen the
# original bare-task-restatement bug - verified with
# test_deployed_via_a_bare_task_restatement_is_still_rejected.
_DEPLOYMENT_CLAUSE_RE = re.compile(
    r"\b(?:can|could) be (?:applied|used|deployed)\s+(?:to|for|in)\s+[a-z]"
    r"|\b(?:is\s+)?applicable\s+(?:to|in)\s+[a-z]"
    r"|\bapplications?\s+(?:such as|include|in|to|for)\s+[a-z]"
    r"|\bapplications?\s*(?:,\s*|\s+(?:of|for|in)\s+[A-Za-z0-9][\w\-' ]{0,30}?,\s*)(?:such as|includes?|including)\s+[a-z]"
    r"|\breal-world applications?\s+(?:in|for)\s+[a-z]"
    r"|\bdeployed\s+(?:in|to|for|by|via|through)\s+[a-z]"
    r"|\bused\s+(?:in|for|to|by|as)\s+[a-z]"
    r"|\b(?:applied|applicable|targets?|targeting)\s+(?:to|for|in)\s+[a-z]"
    r"|\buseful\s+(?:to|for|in)\s+[a-z]"
    r"|\bsupport(?:s|ed|ing)?\s+[a-z]+\s+in\s+[a-z]",
    re.IGNORECASE,
)

# A multi-item list marker - a real restatement of the paper's own task
# is never phrased as a list of examples.
_ENUMERATION_RE = re.compile(r"\bsuch as\b|\bincluding\b", re.IGNORECASE)

# A deployment-clause span is bounded at the next real sentence break -
# after abbreviation periods have already been normalized away, so an
# "e.g."/"i.e." inside the span doesn't cut it short.
_SPAN_BOUNDARY_RE = re.compile(r"[.;]")

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
    r"|\bgovernments?\b|\bagenc(y|ies)\b|\bauthorit(y|ies)\b|\bdecision[- ]makers?\b|\bstakeholders?\b"
    r"|\b\w+(ologists?|icians?)\b"
    r"|\bin (clinical|industrial|educational|practical|real-world) (practice|settings?|use|contexts?)\b"
    r"|\bat scale\b|\bin the field\b|\bin practice\b",
    re.IGNORECASE,
)

# Downstream actions distinct from the paper's own predictive/detection/
# classification verb - a human or institutional response taken as a
# RESULT of the system's output, not the system's own computation.
# The "support X in Y-ing" alternative is Fix B's shape, mirrored here so
# it's self-sufficient the same way enumeration is below (Sec 97
# comment): supporting a named actor IN performing an action is, by its
# own grammar, both an actor reference and a downstream action - it does
# not need a separate curated actor/action word to also match.
#
# The "inform(s/ed/ing) ... decision(s)" alternative is the same kind of
# grammatical SHAPE, added 2026-09-04 investigating applications'
# unusually high (97%) rejection rate across the corpus: "zLend is
# deployed in production, informing real lending decisions via third-
# party API integrations" is unambiguous real-world deployment language -
# a system that INFORMS decisions is providing decision support in
# substance - but the curated phrase "decision support" doesn't literally
# appear, only its meaning. Matches "inform(s/ed/ing) <=3 words>
# decision(s)" rather than a fixed vocabulary, same reasoning as "support
# X in Y-ing" above.
_DOWNSTREAM_ACTION_RE = re.compile(
    r"\binterventions?\b|\btriage\b|\bmanual review\b|\bcounsel(l)?ing\b"
    r"|\btreatment plan(ning)?\b|\bresource allocation\b|\bpolicy( ?making)?\b"
    r"|\bremediation\b|\bprioriti[sz](e|ation|ing)\b|\bdecision support\b"
    r"|\brisk mitigation\b|\bearly (intervention|warning)\b|\bscreening\b"
    r"|\breferrals?\b|\bflagg?ing\b|\balert(ing)?\b"
    r"|\bsupport(?:s|ed|ing)?\s+[a-z]+\s+in\s+[a-z]+ing\b"
    r"|\binform(?:s|ed|ing)?\s+(?:[a-z]+\s+){0,3}decisions?\b",
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
#
# Fix C (2026-09-03 investigation, closing one of the two residual Gate 1
# false positives noted in assessment/applications.py's
# OWN_TASK_OVERLAP_THRESHOLD comment): "for a given user" / "for an
# individual user" and similar generic-user phrasing satisfied this fallback
# ("for a g[iven user]" reads as "for" + a following word) without naming
# any real actor, institution, or setting - it is a restatement of the
# system's OWN subject ("a user" of the system being described), the same
# failure mode "in general" already guards against, just with a noun phrase
# instead of an adverb. Unlike the companion "in QPE"/bare-acronym false
# positive documented alongside this one, this fix is a closed, low-risk
# addition: it only ever REMOVES acceptance for a specific vague phrasing,
# and cannot be confused with a genuine named actor the way rejecting bare
# acronyms would reject real institutions like "NHS"/"NASA" that also
# happen to be all-caps (verified this collision is real: "...for NHS."
# and "...for QPE." are accepted through the exact same code path, so a
# bare-acronym exclusion can't discriminate them without corpus-level
# tuning this investigation didn't have access to run - left unaddressed,
# same as the module already documents).
_VAGUE_QUALIFIER_RE = re.compile(
    r"\bin general\b|\bfor future (work|studies|research)\b|\bin the future\b"
    r"|\bfor general purposes\b|\bfor further research\b|\bin future\b"
    r"|\bfor (?:(?:a|an|the|any)\s+)?(?:given|individual|specific|particular|single) users?\b",
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
    normalized = _ABBREVIATION_RE.sub(r"\1", text)
    masked = _TECH_NOUN_APPLICATION_RE.sub(" ", normalized)

    for match in _DEPLOYMENT_CLAUSE_RE.finditer(masked):
        boundary_match = _SPAN_BOUNDARY_RE.search(masked, match.end())
        boundary = boundary_match.start() if boundary_match else len(masked)
        full_span = masked[match.start() : boundary]
        complement = masked[match.end() : boundary]

        if _ACTOR_SETTING_RE.search(full_span) or _DOWNSTREAM_ACTION_RE.search(full_span):
            return True
        if _ENUMERATION_RE.search(full_span):
            return True
        if _VAGUE_QUALIFIER_RE.search(complement):
            continue
        if _QUALIFYING_CONTEXT_RE.search(complement):
            return True

    return False


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
        if (
            _WEAK_GAP_LANGUAGE_RE.search(text)
            and not _has_result_signal(text)
            and not _TECHNICAL_GAP_METRIC_RE.search(text)
        ):
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
