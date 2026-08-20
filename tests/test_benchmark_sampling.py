from __future__ import annotations

import uuid
from datetime import date

import pytest

from researchbridge.benchmark.domains import DOMAIN_CV, DOMAIN_ML, DOMAIN_NLP
from researchbridge.benchmark.sampling import stratified_sample
from researchbridge.db.models import Paper


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    s.close()  # leaving it open would hold a lock and stall the fixture's TRUNCATE teardown


def _paper(session, source_id: str, primary_category: str, year: int) -> Paper:
    paper = Paper(
        id=uuid.uuid4(),
        source="arxiv",
        source_id=source_id,
        title=f"Paper {source_id}",
        publication_date=date(year, 1, 1),
        raw_metadata={"primary_category": primary_category, "categories": [primary_category]},
        ingestion_metadata={},
    )
    session.add(paper)
    return paper


def test_sample_respects_target_counts(session) -> None:
    for i in range(5):
        _paper(session, f"ml-{i}", "cs.LG", 2020 + i)
    for i in range(3):
        _paper(session, f"nlp-{i}", "cs.CL", 2020 + i)
    session.commit()

    sample = stratified_sample(session, targets={DOMAIN_ML: 2, DOMAIN_NLP: 2})

    assert len(sample[DOMAIN_ML]) == 2
    assert len(sample[DOMAIN_NLP]) == 2  # pool smaller than target -> returns everything available


def test_sample_is_deterministic_for_a_fixed_seed(session) -> None:
    for i in range(10):
        _paper(session, f"cv-{i}", "cs.CV", 2015 + i)
    session.commit()

    first = stratified_sample(session, targets={DOMAIN_CV: 4}, seed=7)
    second = stratified_sample(session, targets={DOMAIN_CV: 4}, seed=7)

    assert [p.source_id for p in first[DOMAIN_CV]] == [p.source_id for p in second[DOMAIN_CV]]


def test_sample_spreads_across_years_rather_than_clustering(session) -> None:
    for i in range(20):
        _paper(session, f"cv-{i}", "cs.CV", 2005 + i)
    session.commit()

    sample = stratified_sample(session, targets={DOMAIN_CV: 5})
    years = sorted(p.publication_date.year for p in sample[DOMAIN_CV])

    assert years[0] <= 2007  # near the start of the range
    assert years[-1] >= 2022  # near the end of the range
    assert years[-1] - years[0] >= 15  # spread across most of the 20-year span, not bunched together


def test_empty_domain_returns_empty_list(session) -> None:
    _paper(session, "ml-0", "cs.LG", 2020)
    session.commit()

    sample = stratified_sample(session, targets={DOMAIN_NLP: 3})

    assert sample[DOMAIN_NLP] == []
