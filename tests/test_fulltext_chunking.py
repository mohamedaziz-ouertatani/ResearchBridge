from __future__ import annotations

from researchbridge.fulltext.chunking import split_paragraphs


def test_splits_on_blank_lines():
    text = (
        "This is the first paragraph with clearly more than enough words to stand alone.\n\n"
        "This is the second paragraph, which also has plenty of words in it to stand alone."
    )
    result = split_paragraphs(text)
    assert result == [
        "This is the first paragraph with clearly more than enough words to stand alone.",
        "This is the second paragraph, which also has plenty of words in it to stand alone.",
    ]


def test_merges_short_leading_paragraph_into_the_next_one():
    text = "Fig. 1.\n\nThis is a real paragraph with more than ten words describing the figure above."
    result = split_paragraphs(text)
    assert result == [
        "Fig. 1. This is a real paragraph with more than ten words describing the figure above."
    ]


def test_merges_short_trailing_paragraph_into_the_previous_one():
    text = "This is a real paragraph with more than ten words describing something important.\n\nThe end."
    result = split_paragraphs(text)
    assert result == [
        "This is a real paragraph with more than ten words describing something important. The end."
    ]


def test_lone_short_paragraph_is_kept_as_its_own_chunk():
    result = split_paragraphs("Too short.")
    assert result == ["Too short."]


def test_empty_section_returns_no_paragraphs():
    assert split_paragraphs("") == []
    assert split_paragraphs("   \n\n  ") == []


def test_collapses_runs_of_multiple_blank_lines():
    text = (
        "Paragraph one has clearly more than enough words to count on its own merits here.\n\n\n\n"
        "Paragraph two also has clearly more than enough words to count on its own merits here."
    )
    result = split_paragraphs(text)
    assert len(result) == 2
