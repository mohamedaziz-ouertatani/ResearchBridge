"""Fetches full text for CORE papers via the CORE API's own `fullText`
field, not a PDF download.

CORE's public download URLs (core.ac.uk/download/<id>.pdf - what
pdf_url.py's earlier design assumed was directly fetchable) are behind
Cloudflare's bot challenge and return a "Just a moment..." HTML page to
every automated request regardless of authentication - confirmed by
testing both with and without the existing CORE_API_KEY. The CORE API
itself (api.core.ac.uk/v3/outputs/<id>, the same host the connector
already searches against) sometimes carries the paper's full text
directly in its `fullText` field - no PDF, no parsing, just already-clean
text. Coverage is real but sparse: a 15-paper random sample found `fullText`
populated for only ~7% of records, since CORE aggregates many repositories
with wildly differing completeness. A paper without it is a real "nothing
to fetch" outcome, not an error - same bucket as a paper with no PDF URL
for any other source.
"""

from __future__ import annotations

import requests

CORE_OUTPUT_URL = "https://api.core.ac.uk/v3/outputs/{core_id}"
REQUEST_TIMEOUT = 30


def fetch_core_fulltext(source_id: str, api_key: str) -> str | None:
    """Returns the paper's full text, or None if CORE has none for it.

    Raises requests.RequestException on network/HTTP failure - the caller
    (fulltext/pipeline.py) catches this the same way it catches a failed
    PDF fetch for any other source.
    """
    response = requests.get(
        CORE_OUTPUT_URL.format(core_id=source_id),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("fullText") or None
