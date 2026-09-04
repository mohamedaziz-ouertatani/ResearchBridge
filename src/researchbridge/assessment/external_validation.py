"""External validation (blueprint Sec 21, Layer 3 - External Impact; Sec 48).

Real patent and market-indicator lookups, scoped to Tunisia, replacing the
previous always-identical disclaimer (see git history for that version).
Sec 21 still governs the framing: market potential, economic impact, and
commercialization viability are never confidently ASSERTED from literature
or from a handful of patent/indicator hits - this stage only ever reports
what real external sources returned, with an explicit "still needs
independent validation" framing, never a market-potential LEVEL judgment.

Fail-closed per source (see docs/superpowers/specs/2026-09-04-market-patent
-data-integration-design.md): an unconfigured, erroring, timing-out, or
empty-keyword-blocked source reverts to disclaimer wording for its own
section only - this function never raises, and the overall assessment
pipeline never fails because of this stage.
"""

from __future__ import annotations

from dataclasses import dataclass

from researchbridge.assessment.keywords import extract_keywords
from researchbridge.connectors.epo_patents import EPOPatentConnector, EPOPatentConnectorError, PatentHit
from researchbridge.connectors.world_bank import IndicatorValue, WorldBankConnector, WorldBankConnectorError

_DISCLAIMER_TEXT = (
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

_INTRO_TEXT = (
    "Market potential, economic impact, and commercialization viability: external evidence "
    "found below (see sources). This does not replace independent market research or industry "
    "expert review."
)

_PATENT_UNCONFIGURED = "Related patents: NOT ASSESSED - patent search unavailable."
_PATENT_ERROR = "Related patents: NOT ASSESSED - patent search failed."
_PATENT_NONE_FOUND = "Related patents (Tunisia): no matching patents found for this idea's terms."
_MARKET_ERROR = "Tunisia economic indicators: NOT ASSESSED - indicator lookup failed."
_MARKET_NONE_FOUND = "Tunisia economic indicators: no data available."


@dataclass
class ExternalValidationResult:
    text: str
    patent_hits: list[PatentHit]
    indicator_values: list[IndicatorValue]


def assess_external_validation(
    title: str | None,
    raw_text: str,
    has_applications: bool,
    patent_connector: EPOPatentConnector | None,
    market_connector: WorldBankConnector | None,
) -> ExternalValidationResult:
    keywords = extract_keywords(title, raw_text)
    if not keywords:
        return ExternalValidationResult(
            text=_DISCLAIMER_TEXT + (_APPLICATIONS_SUFFIX if has_applications else ""),
            patent_hits=[],
            indicator_values=[],
        )

    patent_hits, patent_section = _patent_section(patent_connector, keywords)
    indicator_values, market_section = _market_section(market_connector)

    text = _INTRO_TEXT + "\n\n" + patent_section + "\n\n" + market_section
    if has_applications:
        text += _APPLICATIONS_SUFFIX

    return ExternalValidationResult(text=text, patent_hits=patent_hits, indicator_values=indicator_values)


def _patent_section(connector: EPOPatentConnector | None, keywords: list[str]) -> tuple[list[PatentHit], str]:
    if connector is None:
        return [], _PATENT_UNCONFIGURED
    try:
        hits = connector.search(keywords=keywords, country="TN")
    except EPOPatentConnectorError:
        return [], _PATENT_ERROR
    if not hits:
        return [], _PATENT_NONE_FOUND
    keyword_list = ", ".join(f'"{kw}"' for kw in keywords)
    lines = [f"Related patents (Tunisia, EPO OPS search on {keyword_list}):"]
    for hit in hits:
        applicant = f" (applicant: {hit.applicant})" if hit.applicant else ""
        pub_date = f" - {hit.publication_date.isoformat()}" if hit.publication_date else ""
        lines.append(f"- {hit.publication_number} - \"{hit.title}\"{applicant}{pub_date}")
        if hit.url:
            lines.append(f"  {hit.url}")
    return hits, "\n".join(lines)


def _market_section(connector: WorldBankConnector | None) -> tuple[list[IndicatorValue], str]:
    if connector is None:
        return [], _MARKET_ERROR
    try:
        values = connector.fetch_indicators(country="TN")
    except WorldBankConnectorError:
        return [], _MARKET_ERROR
    if not values:
        return [], _MARKET_NONE_FOUND
    lines = ["Tunisia economic indicators (World Bank, most recent available):"]
    for value in values:
        unit = f" {value.unit}" if value.unit else ""
        lines.append(f"- {value.indicator_name}: {value.value:,.1f}{unit} ({value.year})")
    return values, "\n".join(lines)
