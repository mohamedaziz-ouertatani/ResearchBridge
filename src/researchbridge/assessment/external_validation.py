"""External validation needed (blueprint Sec 21, Layer 3 - External Impact).

Unlike every other assess_* function in this package, this one is NOT
literature-derived and needs no DB access or evidence grounding: Sec 21
is explicit that market potential, economic impact, customer demand,
competitive landscape, and commercialization potential must NOT be
confidently inferred from scientific papers alone. There is no external
market/patent/industry data source integrated into this system (Sec 48
is future work), so the honest answer is always the same structural
disclaimer, not something to compute per-assessment.

If no external evidence exists, Sec 21 requires exactly this shape:

    Market Potential: NOT ASSESSED
    Reason: Insufficient external market evidence.
    Required validation: Market research / industry expert review

This mentions whether potential applications were identified only to
point the reader at what would need validating - it never asserts a
market potential level or any external-impact judgment itself.
"""

from __future__ import annotations

_BASE_TEXT = (
    "Market potential, economic impact, and commercialization viability are "
    "NOT ASSESSED. This system does not integrate external market, patent, "
    "or industry data (see blueprint Sec 21/48) - the literature alone cannot "
    "support a judgment here. Required validation: market research and/or "
    "industry expert review."
)

_APPLICATIONS_SUFFIX = (
    " This applies to the potential applications identified above: their real-world "
    "demand and commercial viability still need independent validation."
)


def assess_external_validation(has_applications: bool) -> str:
    """Deterministic, always the same disclaimer - see module docstring."""
    return _BASE_TEXT + (_APPLICATIONS_SUFFIX if has_applications else "")
