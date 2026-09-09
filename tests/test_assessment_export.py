from __future__ import annotations

import io
import uuid

import pytest

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
        potential_opportunities_status="not_assessed",
        risks_and_limitations="- Paper Title: evaluated only on offline datasets",
        recommendation="Proceed with caution",
        confidence="medium",
        human_reviewed=False,
        evidence=[
            AssessmentEvidenceOut(
                role="comparison", evidence_id=uuid.uuid4(), paper_id=PAPER_ID, paper_title="Paper Title",
                text="a graph attention mechanism", section=None,
            ),
            AssessmentEvidenceOut(
                role="risk", evidence_id=uuid.uuid4(), paper_id=PAPER_ID, paper_title="Paper Title",
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


def test_unrecognized_applications_status_raises_instead_of_masking() -> None:
    assessment = _assessment(potential_applications=None, potential_applications_status="some_new_status_value")
    with pytest.raises(ValueError, match="some_new_status_value"):
        build_report_sections(assessment)


def test_opportunities_unavailable_status_shows_retry_message_not_permanent_refusal() -> None:
    assessment = _assessment(potential_opportunities=None, potential_opportunities_status="unavailable")
    sections = build_report_sections(assessment)
    opportunities = next(s for s in sections if s.label == "Product / technology opportunities")
    assert "temporarily unavailable" in opportunities.unassessed_reason.lower()


def test_opportunities_not_assessed_status_shows_no_grounds_message() -> None:
    assessment = _assessment(potential_opportunities=None, potential_opportunities_status="not_assessed")
    sections = build_report_sections(assessment)
    opportunities = next(s for s in sections if s.label == "Product / technology opportunities")
    assert "inventing a claim" in opportunities.unassessed_reason.lower()


def test_existing_solutions_evidence_excludes_quotes_already_in_body() -> None:
    """comparison_summary already embeds each claim's text as a short
    blockquote; the same evidence rows must not also appear in
    section.evidence, or every quote renders twice."""
    assessment = _assessment(
        comparison_summary='Problems already addressed\n- "Paper Title": a graph attention mechanism',
        evidence=[
            AssessmentEvidenceOut(
                role="comparison", evidence_id=uuid.uuid4(), paper_id=PAPER_ID, paper_title="Paper Title",
                text="a graph attention mechanism", section=None,
            ),
        ],
    )
    sections = build_report_sections(assessment)
    existing_solutions = next(s for s in sections if s.label == "Existing solutions")
    assert existing_solutions.evidence == []


def test_role_evidence_dedups_identical_quotes() -> None:
    """Two distinct evidence rows with identical text (e.g. the same claim
    cited for two different dimensions) must collapse to one in the
    rendered list."""
    duplicate_text = "We introduce PPML-Omics, a federated learning framework."
    assessment = _assessment(
        novelty_reasoning="Some overlap exists.",
        evidence=[
            AssessmentEvidenceOut(
                role="novelty", evidence_id=uuid.uuid4(), paper_id=PAPER_ID, paper_title="Paper Title",
                text=duplicate_text, section=None,
            ),
            AssessmentEvidenceOut(
                role="novelty", evidence_id=uuid.uuid4(), paper_id=PAPER_ID, paper_title="Paper Title",
                text=duplicate_text, section=None,
            ),
        ],
    )
    sections = build_report_sections(assessment)
    novelty = next(s for s in sections if s.label == "Novelty assessment")
    assert len(novelty.evidence) == 1


def test_build_markdown_escapes_body_text_containing_markdown_syntax() -> None:
    text = _md_text(
        build_markdown(_assessment(risks_and_limitations="- Paper Title: *fabricated* claims [dangerous](url)"))
    )

    # the escaped form should appear verbatim; the raw unescaped form should not
    assert "\\- Paper Title: \\*fabricated\\* claims \\[dangerous\\](url)" in text
    assert "\n- Paper Title: *fabricated* claims [dangerous](url)\n" not in text


def test_out_of_corpus_assessment_carries_a_banner_in_the_markdown_report() -> None:
    """An idea with no relevant neighbours must say so at the top. Found
    live 2026-09-09: a 13th-century manuscript idea produced a research gap
    and a risk quoted from an Arabic OCR benchmark, and nothing in the
    report told the reader the idea fell outside this corpus."""
    md = build_markdown(_assessment(corpus_coverage_status="out_of_corpus")).decode("utf-8")

    assert "outside" in md.lower()
    banner_position = md.lower().index("outside")
    assert banner_position < md.index("01")


def test_in_corpus_assessment_has_no_out_of_corpus_banner() -> None:
    md = build_markdown(_assessment(corpus_coverage_status="in_corpus")).decode("utf-8")

    assert "falls outside" not in md.lower()


def test_evidence_quote_tile_counts_distinct_quotes_only() -> None:
    """The tile read len(assessment.evidence), which counts the same
    sentence once per stored row. Found live 2026-09-09: one report
    advertised "evidence quotes: 41" while showing 6 distinct quotes."""
    repeated = "a graph attention mechanism"
    assessment = _assessment(
        evidence=[
            AssessmentEvidenceOut(
                role="comparison", evidence_id=uuid.uuid4(), paper_id=PAPER_ID,
                paper_title="Paper Title", text=repeated, section=None,
            )
            for _ in range(5)
        ]
    )

    md = build_markdown(assessment).decode("utf-8")

    assert "**evidence quotes**: 1" in md


def test_out_of_corpus_banner_appears_in_the_docx_report() -> None:
    from docx import Document

    data = build_docx(_assessment(corpus_coverage_status="out_of_corpus"))
    document = Document(io.BytesIO(data))
    text = "\n".join(p.text for p in document.paragraphs)

    assert "falls outside the corpus" in text


def test_out_of_corpus_banner_appears_in_the_pdf_report() -> None:
    data = build_pdf(_assessment(corpus_coverage_status="out_of_corpus"))

    assert data.startswith(b"%PDF")
    assert len(data) > 0


def test_no_application_evidence_message_names_the_corpus_coverage_cause() -> None:
    """"No retrieved paper stated an application" reads like a fact about
    these specific papers. The real cause is structural: only 2% of the
    corpus (1,619 of 82,596 papers as of 2026-09-09) carries an
    applications claim at all, so this field is empty for almost every
    idea. A reader deciding whether to trust the gap should know which of
    the two they are looking at."""
    sections = build_report_sections(
        _assessment(potential_applications=[], potential_applications_status="no_evidence")
    )
    applications = next(s for s in sections if s.label == "Potential applications")

    assert applications.unassessed_reason is not None
    assert "extraction coverage" in applications.unassessed_reason.lower()


def test_opportunities_not_assessed_message_names_its_dependency_on_applications() -> None:
    sections = build_report_sections(
        _assessment(potential_opportunities=None, potential_opportunities_status="not_assessed")
    )
    opportunities = next(s for s in sections if s.label == "Product / technology opportunities")

    assert opportunities.unassessed_reason is not None
    assert "application" in opportunities.unassessed_reason.lower()


def test_non_english_input_carries_a_banner_above_the_first_section() -> None:
    """The language caveat used to exist only as the first sentence of
    novelty_reasoning's prose, where it read as commentary on the novelty
    number rather than a warning about the whole reading."""
    md = build_markdown(_assessment(input_language_caveat=True)).decode("utf-8")

    assert "does not appear to be written in English" in md
    assert md.index("does not appear to be written in English") < md.index("01")


def test_english_input_has_no_language_banner() -> None:
    md = build_markdown(_assessment(input_language_caveat=False)).decode("utf-8")

    assert "does not appear to be written in English" not in md


def test_both_reliability_banners_render_together() -> None:
    md = build_markdown(
        _assessment(corpus_coverage_status="out_of_corpus", input_language_caveat=True)
    ).decode("utf-8")

    assert "falls outside the corpus" in md
    assert "does not appear to be written in English" in md


def test_stats_tiles_show_retrieval_distances_and_corpus_coverage() -> None:
    md = build_markdown(
        _assessment(nearest_distance=0.0068, mean_distance=0.4123, corpus_coverage_status="in_corpus")
    ).decode("utf-8")

    assert "**nearest / mean distance**: 0.007 / 0.412" in md
    assert "**corpus coverage**: in corpus" in md


def test_stats_tiles_do_not_repeat_the_header_line_fields() -> None:
    # confidence and human-reviewed are already on the header line; the
    # tiles used to print both again
    md = build_markdown(_assessment()).decode("utf-8")

    assert md.count("confidence") == 1
    assert "**human reviewed**" not in md


def test_missing_distances_render_as_a_dash_not_a_crash() -> None:
    md = build_markdown(_assessment(nearest_distance=None, mean_distance=None)).decode("utf-8")

    assert "**nearest / mean distance**: —" in md


def test_limitations_section_is_not_labelled_as_risks_of_the_idea() -> None:
    """Every line is a limitation a RETRIEVED PAPER stated about its own
    work. Calling that "risks" invited readers to take it as an assessment
    of the submitted idea."""
    labels = [s.label for s in build_report_sections(_assessment())]

    assert "Limitations reported in related work" in labels
    assert "Risks / limitations" not in labels
