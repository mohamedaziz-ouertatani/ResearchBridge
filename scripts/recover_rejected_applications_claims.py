"""One-off recovery pass: re-validates every historically-rejected
'applications' candidate logged in extraction_errors against TODAY's
validate_claim_type, and persists the ones that now pass as real
ExtractedClaim/Evidence rows - the same thing the live extraction pipeline
would have done had validation accepted them the first time.

Why this exists: investigating applications' unusually high (97%) corpus-
wide rejection rate found one confirmed, narrow validator miss ("informing
X decisions" as unrecognized decision-support language - see
extraction/validation.py's _DOWNSTREAM_ACTION_RE fix, 2026-09-04). That fix
only affects extraction going forward; every candidate this exact pattern
already rejected is sitting in extraction_errors, never persisted. Rather
than re-ingest/re-extract the whole corpus (expensive, re-fetches nothing
useful - the abstracts are already here), this re-checks only what was
already rejected.

Mirrors extraction/pipeline.py::_process_paper's own two-gate persistence
exactly (evidence shape, grounding re-check) rather than a shortcut path -
see that module for the canonical version this is kept in sync with:
1. Re-run validate_claim_type - skip if still rejected (most will be;
   that's expected, this fix was narrow by design).
2. Re-check grounding against the paper's CURRENT abstract (not skipped
   just because it passed once historically) - the entity-repair backfill
   (2026-09-03) changed some abstracts' text since these candidates were
   first extracted, so a quote grounded then might not be an exact
   substring now, and a fresh quote should never be persisted ungrounded.
3. Skip any (paper_id, text) pair that already has a persisted
   applications claim - a paper can accumulate multiple identical
   extraction_errors rows across separate extraction runs.

confidence is set to "low" for every recovered claim, deliberately not
guessed as "medium": the original candidate's real confidence (which
extractor produced it, heuristic vs semantic - see hybrid.py) isn't
logged in extraction_errors, so this doesn't fabricate a value the
original pipeline never actually assigned to this exact recovery.
evidence.extraction_method/model_version are hardcoded to "hybrid"/
"hybrid-v1" - the only extractor this corpus has ever used (verified
before writing this).

Backs up the full parsed error_detail rows it's about to recover before
writing, same reversibility pattern as this repo's other backfill
scripts.

Usage:
    uv run python scripts/recover_rejected_applications_claims.py            # dry run
    uv run python scripts/recover_rejected_applications_claims.py --apply    # writes
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.config import load_config
from researchbridge.db.models import Evidence, ExtractedClaim, ExtractionError, Paper
from researchbridge.db.session import make_engine
from researchbridge.extraction.validation import validate_claim_type

CLAIM_TYPE = "applications"
RECOVERED_CONFIDENCE = "low"
EXTRACTION_METHOD = "hybrid"
MODEL_VERSION = "hybrid-v1"

_TEXT_MARKER = "(text: "


def _parse_rejected_text(error_detail: str) -> str | None:
    """Extracts the original candidate text from a "claim_type=X rejected:
    ... (text: '...')" error_detail string - the trailing "(text: {text!r})"
    is a real Python repr (see pipeline.py::_process_paper), so it can be
    decoded exactly via ast.literal_eval rather than guessed at with a
    fragile hand-rolled unescaper."""
    marker_index = error_detail.find(_TEXT_MARKER)
    if marker_index == -1 or not error_detail.endswith(")"):
        return None
    repr_text = error_detail[marker_index + len(_TEXT_MARKER) : -1]
    try:
        value = ast.literal_eval(repr_text)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _is_grounded(quote: str, abstract: str | None) -> bool:
    if not quote or not abstract:
        return False
    return _normalize_whitespace(quote) in _normalize_whitespace(abstract)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()

    load_config()
    engine = make_engine(os.environ["DATABASE_URL"])

    with Session(engine) as session:
        rows = session.execute(
            select(ExtractionError.paper_id, ExtractionError.error_detail).where(
                ExtractionError.error_type == "semantic_type_mismatch",
                ExtractionError.error_detail.like(f"claim_type='{CLAIM_TYPE}'%"),
            )
        ).all()

        # dedupe (paper_id, text) - the same rejection can be logged more
        # than once across separate extraction runs
        candidates: dict[tuple[uuid.UUID, str], None] = {}
        unparsed = 0
        for paper_id, error_detail in rows:
            text = _parse_rejected_text(error_detail)
            if text is None:
                unparsed += 1
                continue
            candidates[(paper_id, text)] = None

        print(f"{len(rows)} total rejected '{CLAIM_TYPE}' error rows, {unparsed} unparseable")
        print(f"{len(candidates)} unique (paper, text) candidates after dedup")

        # skip any (paper_id, text) that's already persisted (a prior
        # recovery run, or the candidate was accepted through some other
        # path since)
        existing = {
            (paper_id, text)
            for paper_id, text in session.execute(
                select(ExtractedClaim.paper_id, ExtractedClaim.text).where(ExtractedClaim.claim_type == CLAIM_TYPE)
            ).all()
        }

        paper_ids = {paper_id for paper_id, _text in candidates}
        abstracts = {
            paper_id: abstract
            for paper_id, abstract in session.execute(
                select(Paper.id, Paper.abstract).where(Paper.id.in_(paper_ids))
            ).all()
        }

        to_recover: list[tuple[uuid.UUID, str, str]] = []  # (paper_id, text, tier)
        still_rejected = 0
        no_longer_grounded = 0
        already_persisted = 0

        for paper_id, text in candidates:
            if (paper_id, text) in existing:
                already_persisted += 1
                continue
            result = validate_claim_type(CLAIM_TYPE, text)
            if not result.is_valid:
                still_rejected += 1
                continue
            if not _is_grounded(text, abstracts.get(paper_id)):
                no_longer_grounded += 1
                continue
            to_recover.append((paper_id, text, result.tier or ""))

        print(f"{already_persisted} already persisted (skipped)")
        print(f"{still_rejected} still fail validation under current code (expected - this fix was narrow)")
        print(f"{no_longer_grounded} now ungrounded against the current abstract (skipped)")
        print(f"{len(to_recover)} would be recovered")

        if not to_recover:
            return

        backup_path = Path(
            f"scripts/backfills/recovered_applications_claims_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
        )
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(
            json.dumps(
                [{"paper_id": str(pid), "text": text, "tier": tier} for pid, text, tier in to_recover], indent=2
            )
        )
        print(f"backed up {len(to_recover)} candidates to {backup_path}")

        if not args.apply:
            print("dry run only - pass --apply to write changes")
            return

        for paper_id, text, tier in to_recover:
            evidence = Evidence(
                paper_id=paper_id,
                evidence_type=CLAIM_TYPE,
                section="Abstract",
                text=text,
                source_locator="abstract",
                extraction_method=EXTRACTION_METHOD,
                model_version=MODEL_VERSION,
                confidence=RECOVERED_CONFIDENCE,
            )
            session.add(evidence)
            session.flush()
            session.add(
                ExtractedClaim(
                    paper_id=paper_id,
                    claim_type=CLAIM_TYPE,
                    text=text,
                    evidence_id=evidence.id,
                    confidence=RECOVERED_CONFIDENCE,
                    validation_tier=tier or None,
                )
            )
        session.commit()
        print(f"recovered {len(to_recover)} applications claims")


if __name__ == "__main__":
    main()
