"""Extraction pipeline: select unprocessed papers -> extract -> verify -> persist.

Idempotent: only processes papers with no existing extracted_claims row, so
re-running is safe and (once a costlier extractor is ever used) won't
reprocess/re-bill already-processed papers. Every candidate passes through
two independent gates before anything is persisted, and failing either one
rejects the candidate and logs it to extraction_errors rather than silently
storing or silently relabeling it:

1. Grounding (_quote_is_grounded): the evidence_quote is verified against
   the paper's abstract - the concrete mechanism behind the blueprint's "no
   Grounding Illusion" rule, §15. This proves the quote is not fabricated.

2. Claim-type validation (extraction/validation.py::validate_claim_type):
   grounding alone does not prove the quote actually expresses the claim
   type it was filed under - a quote can be a verbatim, grounded substring
   of the abstract and still be a results sentence mislabeled as a
   research_gap. This is the semantic check for that.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from researchbridge.db.models import (
    Evidence,
    ExtractedClaim,
    ExtractionError,
    ExtractionRun,
    Paper,
)
from researchbridge.extraction.base import Extractor
from researchbridge.extraction.validation import validate_claim_type

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    def __init__(self, extractor: Extractor, session_factory: sessionmaker[Session]) -> None:
        self.extractor = extractor
        self.session_factory = session_factory

    def run(self, limit: int | None = None) -> str:
        session = self.session_factory()
        run = ExtractionRun(extractor_name=self.extractor.extraction_method, status="running")
        session.add(run)
        session.commit()
        run_id = str(run.id)

        try:
            papers = self._select_unprocessed_papers(session, limit)

            for paper in papers:
                self._process_paper(session, run, paper)
                run.papers_processed += 1
                session.commit()

            run.status = "completed"
            run.finished_at = datetime.now(UTC)
            session.commit()

        except Exception as exc:  # noqa: BLE001 - run must record failure, not crash silently
            logger.exception("Extraction run %s failed", run_id)
            run.status = "failed"
            run.error_summary = str(exc)[:2000]
            run.finished_at = datetime.now(UTC)
            session.commit()
            raise
        finally:
            session.close()

        return run_id

    def _process_paper(self, session: Session, run: ExtractionRun, paper: Paper) -> None:
        try:
            candidates = self.extractor.extract(paper)
        except Exception as exc:  # noqa: BLE001 - one extractor failure must not crash the run
            logger.exception("Extractor failed for paper %s", paper.id)
            self._record_error(session, run.id, paper.id, "extractor_error", str(exc)[:2000])
            return

        for candidate in candidates:
            if not _quote_is_grounded(candidate.evidence_quote, paper.abstract):
                self._record_error(
                    session,
                    run.id,
                    paper.id,
                    "ungrounded_quote",
                    f"evidence_quote not found in paper.abstract: {candidate.evidence_quote!r}",
                )
                run.candidates_rejected += 1
                continue

            type_validation = validate_claim_type(candidate.claim_type, candidate.claim_text)
            if not type_validation.is_valid:
                self._record_error(
                    session,
                    run.id,
                    paper.id,
                    "semantic_type_mismatch",
                    f"claim_type={candidate.claim_type!r} rejected: {type_validation.reason} "
                    f"(text: {candidate.claim_text!r})",
                )
                run.candidates_rejected += 1
                continue

            try:
                with session.begin_nested():
                    evidence = Evidence(
                        paper_id=paper.id,
                        evidence_type=candidate.claim_type,
                        section="Abstract",
                        text=candidate.evidence_quote,
                        source_locator="abstract",
                        extraction_method=self.extractor.extraction_method,
                        model_version=self.extractor.model_version,
                        confidence=candidate.confidence,
                    )
                    session.add(evidence)
                    session.flush()  # populate evidence.id for the FK below

                    session.add(
                        ExtractedClaim(
                            paper_id=paper.id,
                            claim_type=candidate.claim_type,
                            text=candidate.claim_text,
                            evidence_id=evidence.id,
                            confidence=candidate.confidence,
                        )
                    )
                run.claims_created += 1
            except IntegrityError as exc:
                self._record_error(session, run.id, paper.id, "persist_error", str(exc)[:2000])

    def _select_unprocessed_papers(self, session: Session, limit: int | None) -> list[Paper]:
        already_extracted = select(ExtractedClaim.paper_id).distinct()
        query = select(Paper).where(Paper.id.notin_(already_extracted))
        if limit is not None:
            query = query.limit(limit)
        return list(session.execute(query).scalars())

    def _record_error(self, session: Session, run_id: Any, paper_id: Any, error_type: str, detail: str) -> None:
        session.add(
            ExtractionError(
                extraction_run_id=run_id,
                paper_id=paper_id,
                error_type=error_type,
                error_detail=detail,
            )
        )


def _quote_is_grounded(quote: str, abstract: str | None) -> bool:
    if not quote or not abstract:
        return False
    return _normalize_whitespace(quote) in _normalize_whitespace(abstract)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
