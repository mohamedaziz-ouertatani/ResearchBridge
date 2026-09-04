from __future__ import annotations

import uuid

from researchbridge.assessment.existing_solutions import build_existing_solutions
from researchbridge.assessment.novelty import FAR_DISTANCE

NEAR = 0.1  # well within FAR_DISTANCE - any in-gate distance would do
FAR = FAR_DISTANCE + 0.1  # comfortably beyond the gate


def _uid() -> uuid.UUID:
    return uuid.uuid4()


def test_empty_input_yields_no_result() -> None:
    result = build_existing_solutions([])
    assert result.text is None
    assert result.evidence_ids == []


def test_groups_claims_by_question_not_by_paper() -> None:
    problem_ev, method_ev, limitation_ev = _uid(), _uid(), _uid()
    papers = [
        (
            "Paper A",
            NEAR,
            [
                ("problem", "Coordination across regions is expensive.", problem_ev),
                ("method", "We propose a lock-free queue.", method_ev),
                ("limitations", "However, it does not scale past 64 threads.", limitation_ev),
            ],
        )
    ]

    result = build_existing_solutions(papers)

    assert result.text is not None
    assert "Problems already addressed" in result.text
    assert "Existing approaches / methods used" in result.text
    assert "Limitations that remain" in result.text
    assert result.text.index("Problems already addressed") < result.text.index("Existing approaches")
    assert result.text.index("Existing approaches") < result.text.index("Limitations that remain")
    assert set(result.evidence_ids) == {problem_ev, method_ev, limitation_ev}


def test_research_gap_and_applications_claims_are_excluded() -> None:
    # those already have their own dedicated report sections
    # (assess_research_gap / assess_applications) - repeating them here
    # would double-report the same claim under two headings
    papers = [
        (
            "Paper A",
            NEAR,
            [
                ("research_gap", "Extending this remains an open problem.", _uid()),
                ("applications", "This can be applied to real-time systems.", _uid()),
            ],
        )
    ]

    result = build_existing_solutions(papers)

    assert result.text is None
    assert result.evidence_ids == []


def test_results_claims_are_excluded() -> None:
    papers = [("Paper A", NEAR, [("results", "Our results show a 3x speedup.", _uid())])]

    result = build_existing_solutions(papers)

    assert result.text is None


def test_multiple_papers_are_merged_within_each_section() -> None:
    papers = [
        ("Paper A", NEAR, [("problem", "Problem A text.", _uid())]),
        ("Paper B", NEAR, [("problem", "Problem B text.", _uid())]),
    ]

    result = build_existing_solutions(papers)

    assert result.text is not None
    assert '"Paper A": Problem A text.' in result.text
    assert '"Paper B": Problem B text.' in result.text


def test_main_contribution_claims_are_not_classified_as_problems() -> None:
    # a paper's own contribution ("we show that X improves Y") describes
    # what the paper CONTRIBUTES, not a problem it addresses - putting it
    # under "Problems already addressed" was a section-mapping bug
    papers = [("Paper A", NEAR, [("main_contribution", "We show that our method improves accuracy by 12%.", _uid())])]

    result = build_existing_solutions(papers)

    assert result.text is None or "Problems already addressed" not in result.text


def test_papers_beyond_far_distance_contribute_nothing() -> None:
    papers = [("Off-topic Paper", FAR, [("problem", "A generic scene-setting sentence.", _uid())])]

    result = build_existing_solutions(papers)

    assert result.text is None
    assert result.evidence_ids == []


def test_in_gate_papers_still_contribute_when_mixed_with_out_of_gate_papers() -> None:
    near_ev, far_ev = _uid(), _uid()
    papers = [
        ("Relevant Paper", NEAR, [("method", "We propose a graph attention mechanism.", near_ev)]),
        ("Off-topic Paper", FAR, [("method", "We propose an unrelated method.", far_ev)]),
    ]

    result = build_existing_solutions(papers)

    assert result.text is not None
    assert '"Relevant Paper": We propose a graph attention mechanism.' in result.text
    assert "Off-topic Paper" not in result.text
    assert result.evidence_ids == [near_ev]


def test_distance_exactly_at_far_distance_is_still_in_gate() -> None:
    # the gate is <=, matching every sibling section's own boundary
    ev = _uid()
    papers = [("Boundary Paper", FAR_DISTANCE, [("problem", "Right at the edge.", ev)])]

    result = build_existing_solutions(papers)

    assert result.text is not None
    assert result.evidence_ids == [ev]


def test_distance_just_beyond_far_distance_is_excluded() -> None:
    import math

    just_beyond = math.nextafter(FAR_DISTANCE, math.inf)
    papers = [("Just Beyond Paper", just_beyond, [("problem", "Just past the edge.", _uid())])]

    result = build_existing_solutions(papers)

    assert result.text is None


def test_claim_extraction_and_evidence_text_stay_verbatim_for_in_gate_papers() -> None:
    # this fix only decides WHICH papers contribute, never rewrites or
    # summarizes what a contributing paper's own claim says - same "no
    # Grounding Illusion" guarantee as before
    ev = _uid()
    verbatim_text = "We propose a lock-free queue for high throughput systems."
    papers = [("Paper A", NEAR, [("method", verbatim_text, ev)])]

    result = build_existing_solutions(papers)

    assert f'- "Paper A": {verbatim_text}' in result.text
    assert result.evidence_ids == [ev]
