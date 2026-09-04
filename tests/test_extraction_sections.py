from __future__ import annotations

from researchbridge.extraction.sections import sentences_for_field


def test_returns_sentences_from_the_first_present_target_section() -> None:
    sections = {"discussion": "First sentence here. Second one too.", "conclusion": "A conclusion sentence."}

    result = sentences_for_field("limitations", sections)

    assert result == [("First sentence here.", "discussion"), ("Second one too.", "discussion")]


def test_falls_through_to_the_next_target_section_when_the_first_is_absent() -> None:
    sections = {"conclusion": "A conclusion sentence."}

    result = sentences_for_field("limitations", sections)

    assert result == [("A conclusion sentence.", "conclusion")]


def test_empty_list_when_no_target_section_is_present() -> None:
    assert sentences_for_field("limitations", {"references": "Some citation text."}) == []


def test_empty_list_for_a_field_with_no_field_sections_entry() -> None:
    assert sentences_for_field("problem", {"introduction": "Some text."}) == []


def test_empty_list_when_sections_is_empty() -> None:
    assert sentences_for_field("method", {}) == []


def test_skips_a_present_but_blank_section_and_falls_through() -> None:
    sections = {"discussion": "   ", "limitations": "Real limitation text."}

    result = sentences_for_field("limitations", sections)

    assert result == [("Real limitation text.", "limitations")]
