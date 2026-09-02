from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field

import pytest

from researchbridge.assessment.applications import assess_applications

NEAR = 0.1
FAR = 0.9

_WORD = re.compile(r"[a-z]+")


@dataclass
class WordOverlapEmbedder:
    """Same fake used elsewhere in this suite (test_assessment_gap.py,
    test_assessment_coverage.py): cosine similarity of two texts is
    exactly their fraction of shared distinct words."""

    model_name: str = "word-overlap-fake"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        vocab = sorted({w for t in texts for w in _WORD.findall(t.lower())})
        vectors = []
        for t in texts:
            words = set(_WORD.findall(t.lower()))
            raw = [1.0 if w in words else 0.0 for w in vocab]
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            vectors.append([x / norm for x in raw])
        return vectors


@pytest.fixture()
def embedder() -> WordOverlapEmbedder:
    return WordOverlapEmbedder()


def _claim(claim_type: str, text: str) -> tuple[str, str, uuid.UUID]:
    return (claim_type, text, uuid.uuid4())


def test_returns_none_for_empty_neighborhood(embedder) -> None:
    result = assess_applications([], embedder)
    assert result.applications is None
    assert result.evidence_ids == []


def test_returns_none_when_no_relevant_papers(embedder) -> None:
    paper_id = uuid.uuid4()
    result = assess_applications(
        [(paper_id, "p1", FAR, [_claim("applications", "fraud detection")])], embedder
    )
    assert result.applications is None


def test_returns_empty_list_when_no_applications_claims_exist(embedder) -> None:
    paper_id = uuid.uuid4()
    result = assess_applications(
        [(paper_id, "p1", NEAR, [_claim("limitations", "some limitation")])], embedder
    )
    assert result.applications == []


def test_collects_application_from_a_relevant_paper(embedder) -> None:
    paper_id = uuid.uuid4()
    app_claim = _claim("applications", "real-time payment fraud screening")

    result = assess_applications([(paper_id, "Fraud Paper", NEAR, [app_claim])], embedder)

    assert result.applications is not None
    assert len(result.applications) == 1
    app = result.applications[0]
    assert app.application == "real-time payment fraud screening"
    assert app.source_paper == "Fraud Paper"
    assert app.paper_id == paper_id
    assert result.evidence_ids == [app_claim[2]]


def test_collects_applications_from_multiple_relevant_papers_nearest_first(embedder) -> None:
    near_id, far_id = uuid.uuid4(), uuid.uuid4()
    near_claim = _claim("applications", "near application")
    far_claim = _claim("applications", "far application")

    result = assess_applications(
        [(near_id, "Near Paper", 0.1, [near_claim]), (far_id, "Far Paper", 0.4, [far_claim])], embedder
    )

    assert [a.application for a in result.applications] == ["near application", "far application"]
    assert result.evidence_ids == [near_claim[2], far_claim[2]]


def test_excludes_too_distant_papers(embedder) -> None:
    near_id, far_id = uuid.uuid4(), uuid.uuid4()
    near_claim = _claim("applications", "a real application")
    far_claim = _claim("applications", "an irrelevant application")

    result = assess_applications(
        [(near_id, "Near", NEAR, [near_claim]), (far_id, "Far", FAR, [far_claim])], embedder
    )

    assert len(result.applications) == 1
    assert result.applications[0].application == "a real application"
    assert result.evidence_ids == [near_claim[2]]


def test_status_not_assessed_when_no_relevant_papers(embedder) -> None:
    result = assess_applications([], embedder)
    assert result.status == "not_assessed"
    assert result.applications is None


def test_status_no_evidence_when_relevant_papers_have_no_application_claim(embedder) -> None:
    paper_id = uuid.uuid4()
    result = assess_applications(
        [(paper_id, "p1", NEAR, [_claim("method", "an unrelated method")])], embedder
    )
    assert result.status == "no_evidence"
    assert result.applications == []


def test_status_found_when_an_application_claim_exists(embedder) -> None:
    paper_id = uuid.uuid4()
    result = assess_applications(
        [(paper_id, "p1", NEAR, [_claim("applications", "used in real-time fraud monitoring")])], embedder
    )
    assert result.status == "found"
    assert result.applications is not None
    assert len(result.applications) == 1


def test_own_task_overlap_does_not_reject_a_genuinely_external_application(embedder) -> None:
    # your worked example: high topical relatedness, but the application
    # names an actor (universities) and a downstream action (intervention)
    # the task claim doesn't have
    task = _claim(
        "method", "predicts which students are at risk of failing using grades and attendance data"
    )
    app = _claim(
        "applications",
        "support universities in identifying students at risk of failure early enough for intervention",
    )
    paper_id = uuid.uuid4()

    result = assess_applications([(paper_id, "Student Risk Paper", NEAR, [task, app])], embedder)

    assert result.status == "found"
    assert len(result.applications) == 1
    assert result.applications[0].application == app[1]


def test_own_task_overlap_rejects_a_near_total_paraphrase_of_the_papers_own_task(embedder) -> None:
    # identical text on both sides is a deliberate stand-in for a real
    # near-paraphrase: the coarse word-overlap fake embedder needs near-
    # identical wording to clear OWN_TASK_OVERLAP_THRESHOLD, whereas a
    # real embedder would already show high-but-not-identical similarity
    # for a genuine paraphrase like "university-level performance
    # prediction" vs. "predicting student academic performance" - this
    # test exercises the same rejection code path unambiguously
    restated_text = "predicts which students are at risk of failing using grades and attendance data"
    task = _claim("method", restated_text)
    app = _claim("applications", restated_text)
    paper_id = uuid.uuid4()

    result = assess_applications([(paper_id, "Student Risk Paper", NEAR, [task, app])], embedder)

    assert result.status == "no_evidence"
    assert result.applications == []


def test_own_task_overlap_only_compares_against_the_same_paper(embedder) -> None:
    # paper A's application claim happens to closely paraphrase paper B's
    # task, not its own - must not be rejected for a cross-paper match
    paper_a, paper_b = uuid.uuid4(), uuid.uuid4()
    shared_text = "predicts which students are at risk of failing using grades and attendance data"
    task_b = _claim("method", shared_text)
    app_a = _claim("applications", shared_text)

    result = assess_applications(
        [(paper_a, "Paper A", NEAR, [app_a]), (paper_b, "Paper B", NEAR, [task_b])], embedder
    )

    assert result.status == "found"
    assert len(result.applications) == 1
