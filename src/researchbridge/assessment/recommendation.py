"""Final recommendation + confidence (blueprint Sec 42/49).

Deterministic, no LLM: aggregates novelty_level, whether a research gap
was found, and technical_feasibility_level into one top-line judgment.
Deliberately does not fold risks_and_limitations/potential_applications
into the decision - those are informational context for a human reader,
not additional signals this rule table has been designed or checked
against.

Sec 42 requires a trustworthy system to be able to conclude "insufficient
evidence" - it must not always force a recommendation. That's the first
check here, not an afterthought.

This is a first-pass heuristic, not a validated model - same status as
novelty.py's/feasibility.py's distance thresholds when first written.
Spot-check it against real assessments before trusting it the way those
thresholds were checked (and one of them, feasibility's, was wrong on
first try and had to be tightened after real-corpus testing). Revisit
this rule table the same way once real assessments exist to check it
against.

Confidence reflects how many of the three underlying signals were
actually assessed WITH STRENGTH, not merely present - not a claim about
how likely the recommendation is to be right:
- high: all three signals are strong
- medium: exactly two are strong
- low: zero or one is strong

"Strong" is deliberately a stricter bar than "assessed" (which still
drives the RECOMMENDATION category below, unchanged):
- novelty: assessed at all (novelty_level != "not_assessed") - no
  further gradient exists to draw on without inventing one.
- research gap: found AND research_gap_is_strong is True (both distance-
  close AND the gap text itself isn't generic boilerplate - see gap.py's
  is_closely_grounded/is_strongly_stated docstrings).
- feasibility: technical_feasibility_level == "high" specifically, not
  "medium" - feasibility.py's own docstring already defines "medium" as
  single-source grounding and "high" as 2+ independent sources, so this
  reuses an existing distinction rather than inventing one.

Found live-testing real ideas (2026-09-04): a fraud-detection assessment
reported "HIGH PRIORITY / high confidence" backed by a single-source
"medium" feasibility grounding and a research-gap sentence that was pure
"future research" boilerplate (extraction/validation.py's weak tier) -
every signal technically had SOME value, so assessed_count was 3/3,
but none of the three was actually strong. A reader has no way to tell
"high confidence, solidly grounded" from "high confidence, thin on every
axis" under the old formula. The RECOMMENDATION category is unchanged by
this - "medium" feasibility and any found gap still count toward HIGH
PRIORITY exactly as before; only what counts as a STRONG signal for
confidence purposes changed.

Recommendation, checked in this order:
1. INSUFFICIENT EVIDENCE - nothing was assessed at all.
2. HIGH PRIORITY - novelty medium/high, a research gap was found, AND
   feasibility medium/high: a real unmet need with documented technical
   grounding to build on.
3. LOW PRIORITY - novelty is low: existing literature already closely
   covers this idea, regardless of the other signals.
4. REQUIRES HUMAN REVIEW - only one of the three signals was assessed;
   too little to weigh against each other.
5. MEDIUM PRIORITY - everything else: some signal exists, but not enough
   to clear the HIGH PRIORITY bar.
"""

from __future__ import annotations

from dataclasses import dataclass

_ASSESSED_NOVELTY_LEVELS = ("medium", "high")


@dataclass
class RecommendationResult:
    recommendation: str
    confidence: str  # high | medium | low
    reasoning: str


def assess_recommendation(
    novelty_level: str,
    research_gap_text: str | None,
    research_gap_is_strong: bool,
    technical_feasibility_level: str,
) -> RecommendationResult:
    novelty_assessed = novelty_level != "not_assessed"
    gap_found = research_gap_text is not None
    feasibility_assessed = technical_feasibility_level != "not_assessed"
    assessed_count = sum([novelty_assessed, gap_found, feasibility_assessed])

    if assessed_count == 0:
        recommendation = "INSUFFICIENT EVIDENCE"
    elif (
        novelty_level in _ASSESSED_NOVELTY_LEVELS
        and gap_found
        and technical_feasibility_level in ("medium", "high")
    ):
        recommendation = "HIGH PRIORITY"
    elif novelty_level == "low":
        recommendation = "LOW PRIORITY"
    elif assessed_count == 1:
        recommendation = "REQUIRES HUMAN REVIEW"
    else:
        recommendation = "MEDIUM PRIORITY"

    # Confidence: a stricter bar than "assessed" - see module docstring.
    gap_is_strong = gap_found and research_gap_is_strong
    feasibility_is_strong = technical_feasibility_level == "high"
    strong_count = sum([novelty_assessed, gap_is_strong, feasibility_is_strong])
    if strong_count == 3:
        confidence = "high"
    elif strong_count == 2:
        confidence = "medium"
    else:
        confidence = "low"

    reasoning = _build_reasoning(
        novelty_level=novelty_level,
        gap_found=gap_found,
        gap_is_strong=gap_is_strong,
        technical_feasibility_level=technical_feasibility_level,
        recommendation=recommendation,
        confidence=confidence,
    )

    return RecommendationResult(recommendation=recommendation, confidence=confidence, reasoning=reasoning)


def _build_reasoning(
    *,
    novelty_level: str,
    gap_found: bool,
    gap_is_strong: bool,
    technical_feasibility_level: str,
    recommendation: str,
    confidence: str,
) -> str:
    novelty_line = (
        f"Novelty signal: {novelty_level}"
        if novelty_level != "not_assessed"
        else "Novelty signal: not assessed (insufficient retrieved evidence)"
    )
    if not gap_found:
        gap_line = "Research gap evidence: none found or not assessed"
    elif gap_is_strong:
        gap_line = "Research gap evidence: a gap was found in the retrieved literature (strongly grounded)"
    else:
        gap_line = (
            "Research gap evidence: a gap was found in the retrieved literature, but the "
            "source is only loosely related or the gap statement itself is generic - "
            "treated as present but not strong for confidence purposes"
        )
    feasibility_line = (
        f"Technical feasibility grounding: {technical_feasibility_level}"
        if technical_feasibility_level != "not_assessed"
        else "Technical feasibility grounding: not assessed (insufficient retrieved evidence)"
    )
    return (
        f"{novelty_line}\n{gap_line}\n{feasibility_line}\n\n"
        f"Recommendation: {recommendation}\nConfidence: {confidence}"
    )
