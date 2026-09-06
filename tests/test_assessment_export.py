from __future__ import annotations

import io
import uuid

from researchbridge.api.schemas import (
    AnalysisClaimOut,
    AssessmentEvidenceOut,
    ResearchAssessmentOut,
    ResearchInputOut,
)
from researchbridge.assessment.export import build_docx, build_markdown, build_pdf, build_report_sections

RESEARCH_INPUT_ID = uuid.uuid4()
ASSESSMENT_ID = uuid.uuid4()
PAPER_ID = uuid.uuid4()


def _assessment(**overrides) -> ResearchAssessmentOut:
    defaults = dict(
        id=ASSESSMENT_ID,
        research_input=ResearchInputOut(
            id=RESEARCH_INPUT_ID,
            input_type="idea",
            raw_text="graph transformers for fraud detection",
            title=None,
            matched_paper_id=None,
        ),
        status="completed",
        retrieved_paper_ids=[str(PAPER_ID)],
        comparison_summary="Paper Title\n- method: a graph attention mechanism",
        novelty_level="medium",
        novelty_reasoning="Moderately related to the closest retrieved paper.",
        research_gap_text="no real-time evaluation exists",
        research_gap_source="input_specific",
        candidate_gap_id=None,
        potential_applications=[
            {"application": "real-time payment fraud screening", "source_paper": "Paper Title", "paper_id": str(PAPER_ID)}
        ],
        potential_applications_status="found",
        technical_feasibility_level="medium",
        technical_feasibility_reasoning="A graph attention mechanism was described.",
        potential_opportunities=None,
        risks_and_limitations="- Paper Title: evaluated only on offline datasets",
        recommendation="Proceed with caution",
        confidence="medium",
        human_reviewed=False,
        evidence=[
            AssessmentEvidenceOut(
                role="comparison", paper_id=PAPER_ID, paper_title="Paper Title",
                text="a graph attention mechanism", section=None,
            ),
            AssessmentEvidenceOut(
                role="risk", paper_id=PAPER_ID, paper_title="Paper Title",
                text="evaluated only on offline datasets", section="Limitations",
            ),
        ],
        claims=[],
    )
    defaults.update(overrides)
    return ResearchAssessmentOut(**defaults)


def _unassessed_assessment() -> ResearchAssessmentOut:
    return _assessment(
        comparison_summary=None,
        novelty_reasoning="Nothing in the corpus is close enough to judge novelty from.",
        research_gap_text=None,
        research_gap_source=None,
        potential_applications=None,
        potential_applications_status="not_assessed",
        technical_feasibility_reasoning="Nothing close enough to ground a feasibility judgement.",
        risks_and_limitations=None,
        evidence=[],
    )


def _docx_text(data: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs)


def _pdf_text(data: bytes) -> str:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def test_build_docx_contains_recommendation_and_input_text() -> None:
    text = _docx_text(build_docx(_assessment()))

    assert "Proceed with caution" in text
    assert "graph transformers for fraud detection" in text


def test_build_docx_includes_evidence_passages() -> None:
    text = _docx_text(build_docx(_assessment()))

    assert "evaluated only on offline datasets" in text
    assert "Paper Title" in text


def _comparison_claim() -> AnalysisClaimOut:
    return AnalysisClaimOut(
        id=uuid.uuid4(),
        claim_type="fact",
        claim_text="Paper Title\n- method: a graph attention mechanism",
        confidence="medium",
        status="pending",
    )


def test_build_report_sections_matches_a_claim_by_exact_text() -> None:
    assessment = _assessment(claims=[_comparison_claim()])

    sections = build_report_sections(assessment)

    existing_solutions = next(s for s in sections if s.label == "Existing solutions")
    assert existing_solutions.claim is not None
    assert existing_solutions.claim.claim_type == "fact"
    novelty = next(s for s in sections if s.label == "Novelty assessment")
    assert novelty.claim is None  # no claim in the list matches novelty_reasoning's text


def test_build_docx_includes_claim_type_and_confidence() -> None:
    # _docx_eyebrow uppercases every label it renders (Word "eyebrow" style)
    text = _docx_text(build_docx(_assessment(claims=[_comparison_claim()])))

    assert "FACT" in text
    assert "CONFIDENCE: MEDIUM" in text


def test_build_pdf_includes_claim_type_and_confidence() -> None:
    text = _pdf_text(build_pdf(_assessment(claims=[_comparison_claim()])))

    assert "fact" in text
    assert "confidence: medium" in text


def test_build_docx_marks_unassessed_fields_with_reasoning() -> None:
    text = _docx_text(build_docx(_unassessed_assessment()))

    assert "No retrieved paper had extracted claims to compare against" in text
    assert "No gap was found" in text
    # potential_applications=None here means "not assessed" (no relevant
    # papers retrieved at all) - distinct from the "no_evidence" ([]) case,
    # which keeps the older "No retrieved paper stated an application"
    # wording - see test_export_distinguishes_applications_not_assessed_from_no_evidence
    assert "No relevant paper was retrieved for this input" in text


def test_build_pdf_contains_recommendation_and_input_text() -> None:
    text = _pdf_text(build_pdf(_assessment()))

    assert "Proceed with caution" in text
    assert "graph transformers for fraud detection" in text


def test_build_pdf_includes_evidence_passages() -> None:
    text = _pdf_text(build_pdf(_assessment()))

    assert "evaluated only on offline datasets" in text
    assert "Paper Title" in text


def test_build_pdf_marks_unassessed_fields_with_reasoning() -> None:
    text = _pdf_text(build_pdf(_unassessed_assessment()))

    assert "No retrieved paper had extracted claims to compare against" in text
    assert "No gap was found" in text
    assert "No relevant paper was retrieved for this input" in text


def test_export_distinguishes_not_assessed_gap_from_checked_no_gap_found() -> None:
    not_assessed = _assessment(research_gap_text=None, research_gap_source="no_relevant_evidence")
    not_found = _assessment(research_gap_text=None, research_gap_source="checked_no_gap_found")

    sections_not_assessed = build_report_sections(not_assessed)
    sections_not_found = build_report_sections(not_found)

    gap_section_a = next(s for s in sections_not_assessed if s.label == "Research gap")
    gap_section_b = next(s for s in sections_not_found if s.label == "Research gap")

    assert gap_section_a.unassessed_reason != gap_section_b.unassessed_reason
    assert "insufficient" in gap_section_a.unassessed_reason.lower()
    assert "no gap" in gap_section_b.unassessed_reason.lower() or "none" in gap_section_b.unassessed_reason.lower()


def test_export_distinguishes_applications_not_assessed_from_no_evidence() -> None:
    not_assessed = _assessment(potential_applications=None, potential_applications_status="not_assessed")
    no_evidence = _assessment(potential_applications=[], potential_applications_status="no_evidence")

    sections_not_assessed = build_report_sections(not_assessed)
    sections_no_evidence = build_report_sections(no_evidence)

    app_section_a = next(s for s in sections_not_assessed if s.label == "Potential applications")
    app_section_b = next(s for s in sections_no_evidence if s.label == "Potential applications")

    assert app_section_a.unassessed_reason != app_section_b.unassessed_reason


def _md_text(data: bytes) -> str:
    return data.decode("utf-8")


def test_build_markdown_contains_recommendation_and_input_text() -> None:
    text = _md_text(build_markdown(_assessment()))

    assert "Proceed with caution" in text
    assert "graph transformers for fraud detection" in text


def test_build_markdown_includes_evidence_passages_as_blockquotes() -> None:
    text = _md_text(build_markdown(_assessment()))

    assert '> "evaluated only on offline datasets"' in text
    assert "Paper Title" in text


def test_build_markdown_marks_unassessed_fields_with_reasoning() -> None:
    text = _md_text(build_markdown(_unassessed_assessment()))

    assert "No retrieved paper had extracted claims to compare against" in text
    assert "No gap was found" in text
    assert "No relevant paper was retrieved for this input" in text


def test_build_markdown_includes_claim_type_and_confidence() -> None:
    text = _md_text(build_markdown(_assessment(claims=[_comparison_claim()])))

    assert "fact" in text
    assert "confidence: medium" in text


def test_build_markdown_is_valid_utf8_bytes() -> None:
    data = build_markdown(_assessment())

    assert isinstance(data, bytes)
    data.decode("utf-8")  # raises if not valid UTF-8


def test_md_escape_neutralizes_inline_markup_characters() -> None:
    from researchbridge.assessment.export import _md_escape

    assert _md_escape("*bold* _italic_ [link](url) `code` back\\slash") == (
        "\\*bold\\* \\_italic\\_ \\[link\\](url) \\`code\\` back\\\\slash"
    )


def test_md_escape_only_guards_line_starting_markers_not_mid_sentence_punctuation() -> None:
    from researchbridge.assessment.export import _md_escape

    # a hyphen or period mid-sentence must NOT be escaped - only doing so
    # at line start (where it could trigger a list/heading) keeps normal
    # prose readable instead of buried in backslashes
    assert _md_escape("state-of-the-art results. Solid work.") == "state-of-the-art results. Solid work."
    assert _md_escape("- a leading bullet-like line") == "\\- a leading bullet-like line"
    assert _md_escape("# not a heading") == "\\# not a heading"
    assert _md_escape("1. not a list") == "\\1. not a list"


def test_build_markdown_escapes_body_text_containing_markdown_syntax() -> None:
    text = _md_text(
        build_markdown(_assessment(risks_and_limitations="- Paper Title: *fabricated* claims [dangerous](url)"))
    )

    # the escaped form should appear verbatim; the raw unescaped form should not
    assert "\\- Paper Title: \\*fabricated\\* claims \\[dangerous\\](url)" in text
    assert "\n- Paper Title: *fabricated* claims [dangerous](url)\n" not in text
