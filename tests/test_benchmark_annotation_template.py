from __future__ import annotations

import uuid
from datetime import date

import yaml

from researchbridge.benchmark.annotation_template import annotation_filename, render_annotation_template
from researchbridge.db.models import Paper


def _paper() -> Paper:
    return Paper(
        id=uuid.uuid4(),
        source="arxiv",
        source_id="2401.01234",
        title='A "Tricky" Title: With Colons & Quotes',
        publication_date=date(2024, 3, 1),
        url="https://arxiv.org/abs/2401.01234",
        raw_metadata={},
        ingestion_metadata={},
    )


def test_filename_is_source_and_source_id() -> None:
    assert annotation_filename(_paper()) == "arxiv_2401.01234.yaml"


def test_rendered_template_is_valid_yaml_with_all_schema_fields() -> None:
    paper = _paper()
    rendered = render_annotation_template(paper, domain="NLP")
    parsed = yaml.safe_load(rendered)

    assert parsed["paper_id"] == str(paper.id)
    assert parsed["source_id"] == "2401.01234"
    assert parsed["title"] == paper.title
    assert parsed["domain"] == "NLP"
    assert parsed["year"] == 2024

    for field in [
        "problem",
        "research_question",
        "method",
        "dataset",
        "main_contribution",
        "results",
        "limitations",
        "applications",
    ]:
        assert parsed[field] == ""

    assert parsed["research_gap"] == {"addressed": "", "remaining": ""}
    assert parsed["key_evidence"] == []


def test_title_with_special_characters_stays_valid_yaml() -> None:
    paper = _paper()
    rendered = render_annotation_template(paper, domain="NLP")
    parsed = yaml.safe_load(rendered)
    assert parsed["title"] == 'A "Tricky" Title: With Colons & Quotes'


def test_missing_publication_date_leaves_year_blank() -> None:
    paper = _paper()
    paper.publication_date = None
    rendered = render_annotation_template(paper, domain="NLP")
    parsed = yaml.safe_load(rendered)
    assert parsed["year"] is None
