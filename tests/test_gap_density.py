"""Tests for compute_gap_density (src/researchbridge/gaps/density.py)."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from researchbridge.db.models import CandidateGap, Paper, PaperCategory
from researchbridge.gaps.density import UNCATEGORIZED, compute_gap_density


@pytest.fixture()
def session(session_factory):
    s = session_factory()
    yield s
    s.execute(text("TRUNCATE TABLE candidate_gap_evidence, candidate_gaps CASCADE"))
    s.commit()
    s.close()


def _add_paper(session, title: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=str(uuid.uuid4()), title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def _add_gap(session, seed_paper: Paper, status: str = "pending") -> CandidateGap:
    gap = CandidateGap(
        id=uuid.uuid4(), seed_paper_id=seed_paper.id, observation="an observation", gap_type="inference",
        status=status, contributing_paper_count=3, similarity_threshold=0.35, detection_method="cluster-v1",
    )
    session.add(gap)
    session.flush()
    return gap


def test_groups_gap_counts_by_seed_papers_category(session) -> None:
    ml_paper = _add_paper(session, "ml paper")
    session.add(PaperCategory(paper_id=ml_paper.id, category="Machine Learning", confidence="high", source="arxiv"))
    robotics_paper = _add_paper(session, "robotics paper")
    session.add(PaperCategory(paper_id=robotics_paper.id, category="Robotics", confidence="high", source="arxiv"))
    session.commit()
    _add_gap(session, ml_paper)
    _add_gap(session, ml_paper)
    _add_gap(session, robotics_paper)
    session.commit()

    buckets = {b.category: b.gap_count for b in compute_gap_density(session)}

    assert buckets == {"Machine Learning": 2, "Robotics": 1}


def test_gap_with_no_paper_category_falls_into_uncategorized_bucket(session) -> None:
    paper = _add_paper(session, "an uncategorized paper")
    _add_gap(session, paper)
    session.commit()

    buckets = {b.category: b.gap_count for b in compute_gap_density(session)}

    assert buckets == {UNCATEGORIZED: 1}


def test_rejected_gaps_are_excluded(session) -> None:
    paper = _add_paper(session, "a paper")
    session.add(PaperCategory(paper_id=paper.id, category="Machine Learning", confidence="high", source="arxiv"))
    session.commit()
    _add_gap(session, paper, status="rejected")
    session.commit()

    buckets = compute_gap_density(session)

    assert buckets == []


def test_gap_counted_once_per_category_even_with_multiple_seed_categories(session) -> None:
    paper = _add_paper(session, "a multi-category paper")
    session.add(PaperCategory(paper_id=paper.id, category="Machine Learning", confidence="high", source="arxiv"))
    session.add(PaperCategory(paper_id=paper.id, category="Robotics", confidence="high", source="arxiv"))
    session.commit()
    _add_gap(session, paper)
    session.commit()

    buckets = {b.category: b.gap_count for b in compute_gap_density(session)}

    assert buckets == {"Machine Learning": 1, "Robotics": 1}
