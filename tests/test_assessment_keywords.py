from __future__ import annotations

from researchbridge.assessment.keywords import extract_keywords


def test_extracts_top_terms_from_title_and_text() -> None:
    keywords = extract_keywords(
        title="Automated irrigation control",
        raw_text="A low-cost soil moisture sensor network for automated irrigation control "
        "in smallholder farms. The system uses soil moisture readings to trigger irrigation.",
        max_keywords=8,
    )

    assert "soil moisture" in keywords
    assert "irrigation" in " ".join(keywords)
    assert len(keywords) <= 8


def test_excludes_stopwords() -> None:
    keywords = extract_keywords(title=None, raw_text="the a an of and or but soil moisture sensor")

    assert "the" not in keywords
    assert "and" not in keywords


def test_empty_text_returns_empty_list() -> None:
    assert extract_keywords(title=None, raw_text="") == []


def test_symbol_only_text_returns_empty_list() -> None:
    assert extract_keywords(title=None, raw_text="!!! ... ---") == []


def test_none_title_does_not_error() -> None:
    keywords = extract_keywords(title=None, raw_text="soil moisture sensor network irrigation control")

    assert keywords != []


def test_is_deterministic() -> None:
    args = {"title": "Irrigation", "raw_text": "soil moisture sensor network for irrigation control"}
    assert extract_keywords(**args) == extract_keywords(**args)


def test_respects_max_keywords() -> None:
    keywords = extract_keywords(
        title="Irrigation",
        raw_text="soil moisture sensor network automated irrigation control valve pump reservoir "
        "farm crop yield water conservation drought",
        max_keywords=3,
    )

    assert len(keywords) <= 3
