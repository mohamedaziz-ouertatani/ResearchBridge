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
actually assessed (novelty != "not_assessed", a research gap was found,
feasibility != "not_assessed") - not a claim about how likely the
recommendation is to be right:
- high: all three assessed
- medium: exactly two assessed
- low: zero or one assessed

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

    if assessed_count == 3:
        confidence = "high"
    elif assessed_count == 2:
        confidence = "medium"
    else:
        confidence = "low"

    reasoning = _build_reasoning(
        novelty_level=novelty_level,
        gap_found=gap_found,
        technical_feasibility_level=technical_feasibility_level,
        recommendation=recommendation,
        confidence=confidence,
    )

    return RecommendationResult(recommendation=recommendation, confidence=confidence, reasoning=reasoning)


def _build_reasoning(
    *, novelty_level: str, gap_found: bool, technical_feasibility_level: str, recommendation: str, confidence: str
) -> str:
    novelty_line = (
        f"Novelty signal: {novelty_level}"
        if novelty_level != "not_assessed"
        else "Novelty signal: not assessed (insufficient retrieved evidence)"
    )
    gap_line = (
        "Research gap evidence: a gap was found in the retrieved literature"
        if gap_found
        else "Research gap evidence: none found or not assessed"
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
