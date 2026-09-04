from __future__ import annotations

from researchbridge.extraction.base import FIELD_SECTIONS, ClaimCandidate


def test_claim_candidate_defaults_section_to_abstract() -> None:
    candidate = ClaimCandidate(claim_type="method", claim_text="x", evidence_quote="x", confidence="medium")
    assert candidate.section == "abstract"


def test_claim_candidate_accepts_an_explicit_section() -> None:
    candidate = ClaimCandidate(
        claim_type="method", claim_text="x", evidence_quote="x", confidence="high", section="methods"
    )
    assert candidate.section == "methods"


def test_field_sections_covers_every_cue_phrase_field() -> None:
    # every field HeuristicExtractor's _CUE_PHRASES covers (all Sec 28
    # fields except "problem", which stays abstract-only by design) must
    # have a FIELD_SECTIONS entry, or it would silently never get a
    # full-text match at all
    expected_fields = {
        "research_question", "method", "dataset", "main_contribution",
        "results", "limitations", "applications", "research_gap",
    }
    assert expected_fields.issubset(FIELD_SECTIONS.keys())


def test_field_sections_values_are_nonempty_tuples_of_known_section_names() -> None:
    known_sections = {
        "abstract", "introduction", "related_work", "methods", "experiments",
        "results", "discussion", "limitations", "conclusion", "acknowledgments",
        "references", "body",
    }
    for field, section_tuple in FIELD_SECTIONS.items():
        assert len(section_tuple) > 0, f"{field} has an empty section tuple"
        assert set(section_tuple).issubset(known_sections), f"{field} references an unknown section"
