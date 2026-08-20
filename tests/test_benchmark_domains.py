from __future__ import annotations

import uuid

from researchbridge.benchmark.domains import DOMAIN_CV, DOMAIN_ML, DOMAIN_NLP, DOMAIN_OTHER, DOMAIN_SYSTEMS, classify_domain
from researchbridge.db.models import Paper


def _paper(raw_metadata: dict) -> Paper:
    return Paper(id=uuid.uuid4(), source="arxiv", source_id="x", title="t", raw_metadata=raw_metadata, ingestion_metadata={})


def test_classifies_by_primary_category() -> None:
    paper = _paper({"primary_category": "cs.CL", "categories": ["cs.CL", "cs.LG"]})
    assert classify_domain(paper) == DOMAIN_NLP


def test_falls_back_to_categories_list_when_primary_unmapped() -> None:
    paper = _paper({"primary_category": "cs.RO", "categories": ["cs.RO", "cs.CV"]})
    assert classify_domain(paper) == DOMAIN_CV


def test_unrecognized_categories_fall_back_to_other() -> None:
    paper = _paper({"primary_category": "cs.RO", "categories": ["cs.RO"]})
    assert classify_domain(paper) == DOMAIN_OTHER


def test_missing_metadata_falls_back_to_other() -> None:
    paper = _paper({})
    assert classify_domain(paper) == DOMAIN_OTHER


def test_systems_and_ml_categories() -> None:
    assert classify_domain(_paper({"primary_category": "cs.DC"})) == DOMAIN_SYSTEMS
    assert classify_domain(_paper({"primary_category": "stat.ML"})) == DOMAIN_ML
