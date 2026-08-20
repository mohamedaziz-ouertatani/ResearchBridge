"""Field-level normalization applied during the pipeline's normalize stage."""

from __future__ import annotations

import re

_DOI_PREFIX_RE = re.compile(r"^\s*(?:https?://)?(?:dx\.)?doi\.org/", re.IGNORECASE)
_DOI_SCHEME_RE = re.compile(r"^\s*doi:\s*", re.IGNORECASE)


def normalize_doi(raw: str | None) -> str | None:
    """Strip URL/scheme prefixes and lowercase a DOI for consistent matching.

    DOIs are case-insensitive per the DOI Handbook, so lowercasing here means
    two differently-cased renderings of the same DOI compare equal. The
    original, unnormalized value should be kept separately (e.g. in
    raw_metadata) — this function's output is for matching/dedup, not display.
    """
    if raw is None:
        return None

    value = raw.strip()
    if not value:
        return None

    value = _DOI_PREFIX_RE.sub("", value)
    value = _DOI_SCHEME_RE.sub("", value)
    value = value.strip().strip("/")

    return value.lower() or None
