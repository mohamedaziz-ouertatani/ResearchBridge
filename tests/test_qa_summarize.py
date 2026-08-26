from __future__ import annotations

import uuid

import pytest

from researchbridge.api.schemas import QuoteHitOut
from researchbridge.qa.summarize import build_prompt, extract_citations


def _hit(text: str, paper_title: str = "Some Paper") -> QuoteHitOut:
    return QuoteHitOut(
        paper_id=uuid.uuid4(),
        paper_title=paper_title,
        paper_source="arxiv",
        claim_type="limitations",
        text=text,
        section=None,
        confidence="medium",
        score=0.9,
    )


def test_build_prompt_numbers_hits_in_order() -> None:
    hits = [_hit("first quote", "Paper A"), _hit("second quote", "Paper B")]

    system_prompt, user_prompt = build_prompt("what are the limitations?", hits)

    assert "only use" in system_prompt.lower() or "only" in system_prompt.lower()
    assert '[1] "first quote" — Paper A' in user_prompt
    assert '[2] "second quote" — Paper B' in user_prompt
    assert "what are the limitations?" in user_prompt


def test_extract_citations_returns_unique_numbers_in_order_of_appearance() -> None:
    text = "Models struggle offline [2]. This was also noted elsewhere [1][2]."

    citations = extract_citations(text, hit_count=2)

    assert citations == [2, 1]


def test_extract_citations_returns_empty_list_when_no_citations_present() -> None:
    citations = extract_citations("The quotes don't address this question.", hit_count=3)

    assert citations == []


def test_extract_citations_raises_on_out_of_range_citation() -> None:
    with pytest.raises(ValueError, match="out of range"):
        extract_citations("This claims something [5].", hit_count=2)


def test_extract_citations_raises_on_zero_citation() -> None:
    with pytest.raises(ValueError, match="out of range"):
        extract_citations("Cites [0] which isn't a valid index.", hit_count=2)
