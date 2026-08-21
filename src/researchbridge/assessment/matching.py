"""Uploaded-document corpus matching (blueprint Sec 2A, "Uploaded-Paper Path").

Purely an optimization/signal, never required: when an upload can be
identified as an already-ingested arXiv paper, the assessment can point at
it via ResearchInput.matched_paper_id. Detection only for now - matching
does not change how the assessment itself is built (see
assessment/representation.py, which still runs regardless of a match).

Only arXiv is matched (the only connector this corpus has - see
connectors/arxiv.py). The filename check trusts a bare id.id pattern
because that's this corpus's own naming convention (see benchmark/fulltext/
*.txt); the text check requires an explicit "arXiv:" prefix so a
coincidental digit.digit sequence in prose (a percentage, a page number)
never produces a false match.
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from researchbridge.db.models import Paper

_ARXIV_ID = r"\d{4}\.\d{4,5}"
_FILENAME_ID = re.compile(_ARXIV_ID)
_TEXT_ID = re.compile(rf"arXiv:\s*({_ARXIV_ID})(v\d+)?", re.IGNORECASE)
_TEXT_SEARCH_WINDOW = 2000


def match_uploaded_paper(session: Session, source_filename: str | None, raw_text: str) -> uuid.UUID | None:
    arxiv_id = _extract_from_filename(source_filename) or _extract_from_text(raw_text)
    if arxiv_id is None:
        return None

    return session.execute(
        select(Paper.id).where(Paper.source == "arxiv", Paper.source_id == arxiv_id)
    ).scalar_one_or_none()


def _extract_from_filename(source_filename: str | None) -> str | None:
    if not source_filename:
        return None
    match = _FILENAME_ID.search(source_filename)
    return match.group(0) if match else None


def _extract_from_text(raw_text: str) -> str | None:
    match = _TEXT_ID.search(raw_text[:_TEXT_SEARCH_WINDOW])
    return match.group(1) if match else None
