from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field

import pytest

from researchbridge.assessment.gap import assess_research_gap
from researchbridge.db.models import CandidateGap, CandidateGapEvidence, Evidence, ExtractedClaim, Paper

_WORD = re.compile(r"[a-z]+")
NEAR = 0.1  # comfortably within the relevance gate AND the closer NEAR_DISTANCE gate
MID = 0.5  # within RELEVANCE_DISTANCE (0.65) but beyond NEAR_DISTANCE (0.35)
FAR = 0.9  # comfortably beyond both


@dataclass
class WordOverlapEmbedder:
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


def _paper(session, source_id: str, title: str = "a paper") -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    return paper


def _claim(session, paper: Paper, claim_type: str, text: str, extraction_method: str = "hybrid") -> uuid.UUID:
    evidence = Evidence(
        paper_id=paper.id, evidence_type=claim_type, section=None, text=text,
        extraction_method=extraction_method, model_version="v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    session.add(
        ExtractedClaim(paper_id=paper.id, claim_type=claim_type, text=text, evidence_id=evidence.id, confidence="medium")
    )
    return evidence.id


def _candidate_gap(session, seed: Paper, evidence_id: uuid.UUID, status: str, observation: str = "obs") -> CandidateGap:
    gap = CandidateGap(
        id=uuid.uuid4(), seed_paper_id=seed.id, observation=observation, gap_type="inference",
        status=status, contributing_paper_count=3, similarity_threshold=0.35, detection_method="cluster-v1",
    )
    session.add(gap)
    session.flush()
    session.add(CandidateGapEvidence(candidate_gap_id=gap.id, evidence_id=evidence_id))
    return gap


def test_returns_nothing_for_empty_neighborhood(session_factory, embedder) -> None:
    session = session_factory()
    result = assess_research_gap(session, [], embedder)
    session.close()
    assert result.source is None
    assert result.text is None
    assert result.candidate_gap_id is None
    assert result.evidence_ids == []


def test_reuses_an_approved_candidate_gap_in_the_neighborhood(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, "seed")
    other = _paper(session, "other")
    evidence_id = _claim(session, other, "limitations", "offline only")
    gap = _candidate_gap(session, seed, evidence_id, status="approved", observation="Recurring: offline only")
    session.commit()

    result = assess_research_gap(session, [(seed.id, NEAR), (other.id, NEAR)], embedder)

    session.close()
    assert result.source == "reused_candidate_gap"
    assert result.candidate_gap_id == gap.id
    assert result.text == "Recurring: offline only"
    assert result.evidence_ids == [evidence_id]


def test_ignores_an_approved_candidate_gap_too_distant_to_be_relevant(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, "seed")
    other = _paper(session, "other")
    evidence_id = _claim(session, other, "limitations", "offline only")
    _candidate_gap(session, seed, evidence_id, status="approved", observation="Recurring: offline only")
    session.commit()

    result = assess_research_gap(session, [(seed.id, FAR), (other.id, FAR)], embedder)

    session.close()
    assert result.source != "reused_candidate_gap"


def test_ignores_pending_candidate_gaps(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, "seed")
    other = _paper(session, "other")
    evidence_id = _claim(session, other, "limitations", "offline only")
    _candidate_gap(session, seed, evidence_id, status="pending")
    session.commit()

    result = assess_research_gap(session, [(seed.id, NEAR), (other.id, NEAR)], embedder)

    session.close()
    assert result.source != "reused_candidate_gap"


def test_ignores_rejected_candidate_gaps(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, "seed")
    other = _paper(session, "other")
    evidence_id = _claim(session, other, "limitations", "offline only")
    _candidate_gap(session, seed, evidence_id, status="rejected")
    session.commit()

    result = assess_research_gap(session, [(seed.id, NEAR), (other.id, NEAR)], embedder)

    session.close()
    assert result.source != "reused_candidate_gap"


def test_falls_back_to_explicit_research_gap_claim(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, "p1", title="Explicit Gap Paper")
    evidence_id = _claim(session, paper, "research_gap", "no real-time evaluation exists")
    session.commit()

    result = assess_research_gap(session, [(paper.id, NEAR)], embedder)

    session.close()
    assert result.source == "input_specific"
    assert result.candidate_gap_id is None
    assert "no real-time evaluation exists" in result.text
    assert "Explicit Gap Paper" in result.text
    assert result.evidence_ids == [evidence_id]


def test_explicit_gap_is_not_labeled_as_inference(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, "p1", title="Explicit Gap Paper")
    _claim(session, paper, "research_gap", "no real-time evaluation exists")
    session.commit()

    result = assess_research_gap(session, [(paper.id, NEAR)], embedder)

    session.close()
    assert "inference" not in result.text.lower()


def test_explicit_gap_excludes_stub_claims(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, "p1")
    _claim(session, paper, "research_gap", "synthetic placeholder", extraction_method="stub")
    session.commit()

    result = assess_research_gap(session, [(paper.id, NEAR)], embedder)

    session.close()
    assert result.source is None


def test_explicit_gap_ignored_when_paper_is_too_distant_to_be_relevant(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, "p1", title="Barely Related Paper")
    _claim(session, paper, "research_gap", "a tangential unrelated claim")
    session.commit()

    result = assess_research_gap(session, [(paper.id, FAR)], embedder)

    session.close()
    assert result.source is None
    assert result.text is None


def test_falls_back_to_inferred_cross_paper_gap(session_factory, embedder) -> None:
    session = session_factory()
    a = _paper(session, "a")
    b = _paper(session, "b")
    c = _paper(session, "c")
    e1 = _claim(session, a, "limitations", "tested only offline in this setup")
    e2 = _claim(session, b, "limitations", "we test the model only offline in our setup")
    e3 = _claim(session, c, "limitations", "testing here happens only offline within this setup")
    session.commit()

    result = assess_research_gap(
        session, [(a.id, NEAR), (b.id, NEAR), (c.id, NEAR)], embedder, min_cluster_size=3, similarity_threshold=0.3
    )

    session.close()
    assert result.source == "input_specific"
    assert result.candidate_gap_id is None
    assert "inference" in result.text.lower()
    assert set(result.evidence_ids) == {e1, e2, e3}


def test_explicit_gap_prefers_the_nearer_paper_over_an_arbitrary_one(session_factory, embedder) -> None:
    session = session_factory()
    far_paper = _paper(session, "far", title="Far Paper")
    near_paper = _paper(session, "near", title="Near Paper")
    _claim(session, far_paper, "research_gap", "a barely related tangent")
    _claim(session, near_paper, "research_gap", "the actually relevant gap")
    session.commit()

    # near_paper listed first (it's what the assessment actually retrieved closest),
    # even though far_paper was created first
    result = assess_research_gap(session, [(near_paper.id, 0.1), (far_paper.id, 0.3)], embedder)

    session.close()
    assert "the actually relevant gap" in result.text
    assert "Near Paper" in result.text


def test_returns_nothing_when_no_pattern_clears_the_threshold(session_factory, embedder) -> None:
    session = session_factory()
    a = _paper(session, "a")
    b = _paper(session, "b")
    _claim(session, a, "limitations", "tested only offline")
    _claim(session, b, "limitations", "requires substantial gpu resources")
    session.commit()

    result = assess_research_gap(
        session, [(a.id, NEAR), (b.id, NEAR)], embedder, min_cluster_size=3, similarity_threshold=0.3
    )

    session.close()
    assert result.source is None
    assert result.text is None
    assert result.evidence_ids == []


def test_returns_nothing_when_every_retrieved_paper_is_too_distant(session_factory, embedder) -> None:
    session = session_factory()
    a = _paper(session, "a")
    b = _paper(session, "b")
    c = _paper(session, "c")
    _claim(session, a, "limitations", "tested only offline in this setup")
    _claim(session, b, "limitations", "we test the model only offline in our setup")
    _claim(session, c, "limitations", "testing here happens only offline within this setup")
    session.commit()

    result = assess_research_gap(
        session, [(a.id, FAR), (b.id, FAR), (c.id, FAR)], embedder, min_cluster_size=3, similarity_threshold=0.3
    )

    session.close()
    assert result.source is None
    assert result.text is None


def test_explicit_gap_is_closely_grounded_when_source_paper_is_near(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, "p1", title="Explicit Gap Paper")
    _claim(session, paper, "research_gap", "no real-time evaluation exists")
    session.commit()

    result = assess_research_gap(session, [(paper.id, NEAR)], embedder)

    session.close()
    assert result.is_closely_grounded is True


def test_explicit_gap_is_not_closely_grounded_when_source_paper_is_only_mid_distance(session_factory, embedder) -> None:
    session = session_factory()
    paper = _paper(session, "p1", title="Explicit Gap Paper")
    _claim(session, paper, "research_gap", "no real-time evaluation exists")
    session.commit()

    result = assess_research_gap(session, [(paper.id, MID)], embedder)

    session.close()
    # still surfaced as the report field (within RELEVANCE_DISTANCE), but not
    # close enough to count as a strong signal for recommendation purposes
    assert result.text is not None
    assert result.is_closely_grounded is False


def test_reused_candidate_gap_is_closely_grounded_when_seed_paper_is_near(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, "seed")
    other = _paper(session, "other")
    evidence_id = _claim(session, other, "limitations", "offline only")
    _candidate_gap(session, seed, evidence_id, status="approved", observation="Recurring: offline only")
    session.commit()

    result = assess_research_gap(session, [(seed.id, NEAR), (other.id, NEAR)], embedder)

    session.close()
    assert result.is_closely_grounded is True


def test_reused_candidate_gap_is_not_closely_grounded_when_seed_paper_is_only_mid_distance(session_factory, embedder) -> None:
    session = session_factory()
    seed = _paper(session, "seed")
    other = _paper(session, "other")
    evidence_id = _claim(session, other, "limitations", "offline only")
    _candidate_gap(session, seed, evidence_id, status="approved", observation="Recurring: offline only")
    session.commit()

    result = assess_research_gap(session, [(seed.id, MID), (other.id, MID)], embedder)

    session.close()
    assert result.text is not None
    assert result.is_closely_grounded is False


def test_inferred_gap_is_closely_grounded_when_at_least_one_member_paper_is_near(session_factory, embedder) -> None:
    session = session_factory()
    a = _paper(session, "a")
    b = _paper(session, "b")
    c = _paper(session, "c")
    _claim(session, a, "limitations", "tested only offline in this setup")
    _claim(session, b, "limitations", "we test the model only offline in our setup")
    _claim(session, c, "limitations", "testing here happens only offline within this setup")
    session.commit()

    result = assess_research_gap(
        session, [(a.id, NEAR), (b.id, MID), (c.id, MID)], embedder, min_cluster_size=3, similarity_threshold=0.3
    )

    session.close()
    assert result.is_closely_grounded is True


def test_inferred_gap_is_not_closely_grounded_when_no_member_paper_is_near(session_factory, embedder) -> None:
    session = session_factory()
    a = _paper(session, "a")
    b = _paper(session, "b")
    c = _paper(session, "c")
    _claim(session, a, "limitations", "tested only offline in this setup")
    _claim(session, b, "limitations", "we test the model only offline in our setup")
    _claim(session, c, "limitations", "testing here happens only offline within this setup")
    session.commit()

    result = assess_research_gap(
        session, [(a.id, MID), (b.id, MID), (c.id, MID)], embedder, min_cluster_size=3, similarity_threshold=0.3
    )

    session.close()
    assert result.text is not None
    assert result.is_closely_grounded is False


def test_no_gap_found_is_not_closely_grounded(session_factory, embedder) -> None:
    result = assess_research_gap(session_factory(), [], embedder)

    assert result.is_closely_grounded is False
