from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select

from researchbridge.db.models import (
    CandidateGap,
    CandidateGapEvidence,
    Evidence,
    ExtractedClaim,
    ExtractionError,
    ExtractionRun,
    Paper,
)
from researchbridge.extraction.base import ClaimCandidate
from researchbridge.extraction.pipeline import ExtractionPipeline, reset_extraction_data
from researchbridge.extraction.stub import STUB_MODEL_VERSION, StubExtractor


def _make_paper(session, source_id: str, abstract: str | None = "A short abstract.") -> Paper:
    paper = Paper(
        id=uuid.uuid4(),
        source="fake",
        source_id=source_id,
        title="A Paper",
        abstract=abstract,
        raw_metadata={},
        ingestion_metadata={},
    )
    session.add(paper)
    session.commit()
    return paper


@dataclass
class FakeExtractor:
    candidates_by_paper_id: dict = field(default_factory=dict)
    extraction_method: str = "fake"
    model_version: str = "fake-v1"
    raises_for: set = field(default_factory=set)

    def extract(self, paper: Paper) -> list[ClaimCandidate]:
        if paper.id in self.raises_for:
            raise RuntimeError("boom")
        return self.candidates_by_paper_id.get(paper.id, [])


def test_stub_extractor_creates_grounded_evidence_and_claim(session_factory) -> None:
    session = session_factory()
    paper = _make_paper(session, "P1", abstract="We propose a new method for X.")
    session.close()

    pipeline = ExtractionPipeline(extractor=StubExtractor(), session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(ExtractionRun, run_id)
        assert run.status == "completed"
        assert run.papers_processed == 1
        assert run.claims_created == 1
        assert run.candidates_rejected == 0

        claim = session.execute(select(ExtractedClaim).where(ExtractedClaim.paper_id == paper.id)).scalar_one()
        assert claim.claim_type == "contribution"
        assert claim.text == "We propose a new method for X."
        assert claim.confidence == "medium"

        evidence = session.get(Evidence, claim.evidence_id)
        assert evidence.text == "We propose a new method for X."
        assert evidence.section == "Abstract"
        assert evidence.extraction_method == "stub"
        assert evidence.model_version == STUB_MODEL_VERSION
        assert evidence.confidence == "medium"
    finally:
        session.close()


def test_paper_without_abstract_produces_no_claims(session_factory) -> None:
    session = session_factory()
    _make_paper(session, "P1", abstract=None)
    session.close()

    pipeline = ExtractionPipeline(extractor=StubExtractor(), session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(ExtractionRun, run_id)
        assert run.papers_processed == 1
        assert run.claims_created == 0
        assert run.candidates_rejected == 0

        assert session.execute(select(ExtractedClaim)).scalars().all() == []
        assert session.execute(select(ExtractionError)).scalars().all() == []
    finally:
        session.close()


def test_extraction_is_idempotent_across_runs(session_factory) -> None:
    session = session_factory()
    _make_paper(session, "P1")
    session.close()

    pipeline = ExtractionPipeline(extractor=StubExtractor(), session_factory=session_factory)
    first_run_id = pipeline.run()
    second_run_id = pipeline.run()

    session = session_factory()
    try:
        first_run = session.get(ExtractionRun, first_run_id)
        second_run = session.get(ExtractionRun, second_run_id)
        assert first_run.papers_processed == 1
        assert first_run.claims_created == 1
        assert second_run.papers_processed == 0  # nothing left to process
        assert second_run.claims_created == 0

        assert len(session.execute(select(ExtractedClaim)).scalars().all()) == 1  # not duplicated
    finally:
        session.close()


def test_ungrounded_quote_is_rejected_and_logged(session_factory) -> None:
    session = session_factory()
    paper = _make_paper(session, "P1", abstract="The real abstract text.")
    session.close()

    extractor = FakeExtractor(
        candidates_by_paper_id={
            paper.id: [
                ClaimCandidate(
                    claim_type="contribution",
                    claim_text="A fabricated claim.",
                    evidence_quote="This text does not appear in the abstract.",
                    confidence="medium",
                )
            ]
        }
    )
    pipeline = ExtractionPipeline(extractor=extractor, session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(ExtractionRun, run_id)
        assert run.status == "completed"
        assert run.claims_created == 0
        assert run.candidates_rejected == 1

        assert session.execute(select(ExtractedClaim)).scalars().all() == []

        errors = session.execute(select(ExtractionError).where(ExtractionError.paper_id == paper.id)).scalars().all()
        assert len(errors) == 1
        assert errors[0].error_type == "ungrounded_quote"
        assert errors[0].extraction_run_id == run.id
    finally:
        session.close()


def test_grounded_but_semantically_mismatched_claim_is_rejected_and_logged(session_factory) -> None:
    # The reported bug: a benchmark-results sentence is verbatim in the
    # abstract (so grounding passes) but does not express a research gap.
    session = session_factory()
    abstract = "Our model achieves 94.2% AUC and an F1 score of 0.89 on the benchmark."
    paper = _make_paper(session, "P1", abstract=abstract)
    session.close()

    extractor = FakeExtractor(
        candidates_by_paper_id={
            paper.id: [
                ClaimCandidate(
                    claim_type="research_gap",
                    claim_text=abstract,
                    evidence_quote=abstract,
                    confidence="medium",
                )
            ]
        }
    )
    pipeline = ExtractionPipeline(extractor=extractor, session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(ExtractionRun, run_id)
        assert run.status == "completed"
        assert run.claims_created == 0
        assert run.candidates_rejected == 1

        assert session.execute(select(ExtractedClaim)).scalars().all() == []

        errors = session.execute(select(ExtractionError).where(ExtractionError.paper_id == paper.id)).scalars().all()
        assert len(errors) == 1
        assert errors[0].error_type == "semantic_type_mismatch"
        assert errors[0].extraction_run_id == run.id
    finally:
        session.close()


def test_grounded_quote_with_extra_whitespace_still_verifies(session_factory) -> None:
    session = session_factory()
    paper = _make_paper(session, "P1", abstract="First sentence.   Second   sentence with gaps.")
    session.close()

    extractor = FakeExtractor(
        candidates_by_paper_id={
            paper.id: [
                ClaimCandidate(
                    claim_type="contribution",
                    claim_text="Second sentence with gaps.",
                    evidence_quote="Second sentence with gaps.",
                    confidence="medium",
                )
            ]
        }
    )
    pipeline = ExtractionPipeline(extractor=extractor, session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(ExtractionRun, run_id)
        assert run.claims_created == 1
        assert run.candidates_rejected == 0
    finally:
        session.close()


def test_extractor_failure_is_logged_and_run_continues(session_factory) -> None:
    session = session_factory()
    bad_paper = _make_paper(session, "P1")
    good_paper = _make_paper(session, "P2", abstract="A fine abstract.")
    session.close()

    extractor = FakeExtractor(raises_for={bad_paper.id})
    # good_paper isn't in candidates_by_paper_id, so it just yields no candidates;
    # what matters here is that bad_paper's exception doesn't crash the run.
    pipeline = ExtractionPipeline(extractor=extractor, session_factory=session_factory)
    run_id = pipeline.run()

    session = session_factory()
    try:
        run = session.get(ExtractionRun, run_id)
        assert run.status == "completed"
        assert run.papers_processed == 2

        errors = (
            session.execute(select(ExtractionError).where(ExtractionError.paper_id == bad_paper.id)).scalars().all()
        )
        assert len(errors) == 1
        assert errors[0].error_type == "extractor_error"
    finally:
        session.close()


def test_force_reset_deletes_prior_extraction_and_allows_reprocessing(session_factory) -> None:
    session = session_factory()
    paper = _make_paper(session, "P1", abstract="We propose a new method for X.")
    session.close()

    pipeline = ExtractionPipeline(extractor=StubExtractor(), session_factory=session_factory)
    pipeline.run()

    session = session_factory()
    try:
        assert session.execute(select(ExtractedClaim)).scalars().all() != []
        deleted = reset_extraction_data(session)
        assert deleted == 1
        assert session.execute(select(ExtractedClaim)).scalars().all() == []
        assert session.execute(select(Evidence)).scalars().all() == []
    finally:
        session.close()

    # A fresh run should reprocess the paper as if it were never extracted.
    second_run_id = pipeline.run()
    session = session_factory()
    try:
        run = session.get(ExtractionRun, second_run_id)
        assert run.papers_processed == 1
        assert run.claims_created == 1
        assert len(session.execute(select(ExtractedClaim).where(ExtractedClaim.paper_id == paper.id)).scalars().all()) == 1
    finally:
        session.close()


def test_force_reset_cascades_through_candidate_gap_evidence(session_factory) -> None:
    # A candidate gap can cite evidence from a paper being re-extracted;
    # reset_extraction_data must clear that reference first or the Evidence
    # delete violates candidate_gap_evidence's foreign key.
    session = session_factory()
    paper = _make_paper(session, "P1", abstract="We propose a new method for X.")
    session.close()

    pipeline = ExtractionPipeline(extractor=StubExtractor(), session_factory=session_factory)
    pipeline.run()

    session = session_factory()
    try:
        evidence = session.execute(select(Evidence)).scalar_one()
        gap = CandidateGap(
            seed_paper_id=paper.id,
            observation="A recurring pattern.",
            contributing_paper_count=3,
            similarity_threshold=0.8,
            detection_method="embedding_cosine",
        )
        session.add(gap)
        session.flush()
        session.add(CandidateGapEvidence(candidate_gap_id=gap.id, evidence_id=evidence.id))
        session.commit()

        deleted = reset_extraction_data(session)
        assert deleted == 1
        assert session.execute(select(CandidateGapEvidence)).scalars().all() == []
        # The candidate gap itself survives - only its evidence link is severed.
        assert session.get(CandidateGap, gap.id) is not None
    finally:
        session.close()


def test_limit_caps_papers_processed(session_factory) -> None:
    session = session_factory()
    _make_paper(session, "P1")
    _make_paper(session, "P2")
    _make_paper(session, "P3")
    session.close()

    pipeline = ExtractionPipeline(extractor=StubExtractor(), session_factory=session_factory)
    run_id = pipeline.run(limit=2)

    session = session_factory()
    try:
        run = session.get(ExtractionRun, run_id)
        assert run.papers_processed == 2
    finally:
        session.close()
