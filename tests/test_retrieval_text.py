from __future__ import annotations

import uuid

from researchbridge.db.models import Paper
from researchbridge.retrieval.text import document_text


def _paper(title: str | None, abstract: str | None) -> Paper:
    return Paper(
        id=uuid.uuid4(), source="fake", source_id="x", title=title or "", abstract=abstract,
        raw_metadata={}, ingestion_metadata={},
    )


def test_combines_title_and_abstract() -> None:
    text = document_text(_paper("A Title", "An abstract."))
    assert "A Title" in text
    assert "An abstract." in text


def test_missing_abstract_falls_back_to_title_only() -> None:
    assert document_text(_paper("Only A Title", None)) == "Only A Title"


def test_blank_title_and_abstract_returns_none() -> None:
    assert document_text(_paper("   ", "  ")) is None
