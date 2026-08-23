from __future__ import annotations

import uuid

from researchbridge.assessment.existing_solutions import build_existing_solutions


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
    papers = [("Paper A", [("results", "Our results show a 3x speedup.", _uid())])]

    result = build_existing_solutions(papers)

    assert result.text is None


def test_multiple_papers_are_merged_within_each_section() -> None:
    papers = [
        ("Paper A", [("problem", "Problem A text.", _uid())]),
        ("Paper B", [("problem", "Problem B text.", _uid())]),
    ]

    result = build_existing_solutions(papers)

    assert result.text is not None
    assert '"Paper A": Problem A text.' in result.text
    assert '"Paper B": Problem B text.' in result.text
