"""Maps a paper to one of the blueprint's stratification buckets (Sec 24).

Classification is by arXiv category only (the corpus's one source in Phase
1). It reads straight from Paper.raw_metadata rather than joining
paper_categories, since the arXiv connector already stores
primary_category/categories there - see connectors/arxiv.py.
"""

from __future__ import annotations

from researchbridge.db.models import Paper

DOMAIN_ML = "Machine Learning"
DOMAIN_NLP = "NLP"
DOMAIN_CV = "Computer Vision"
DOMAIN_SYSTEMS = "Systems"
DOMAIN_OTHER = "General AI / Other"

DEFAULT_TARGETS: dict[str, int] = {
    DOMAIN_ML: 10,
    DOMAIN_NLP: 8,
    DOMAIN_CV: 8,
    DOMAIN_SYSTEMS: 7,
    DOMAIN_OTHER: 7,
}

_CATEGORY_TO_DOMAIN: dict[str, str] = {
    "cs.LG": DOMAIN_ML,
    "stat.ML": DOMAIN_ML,
    "cs.NE": DOMAIN_ML,
    "cs.CL": DOMAIN_NLP,
    "cs.CV": DOMAIN_CV,
    "eess.IV": DOMAIN_CV,
    "cs.DC": DOMAIN_SYSTEMS,
    "cs.OS": DOMAIN_SYSTEMS,
    "cs.NI": DOMAIN_SYSTEMS,
    "cs.DB": DOMAIN_SYSTEMS,
    "cs.SE": DOMAIN_SYSTEMS,
    "cs.AR": DOMAIN_SYSTEMS,
    "cs.PF": DOMAIN_SYSTEMS,
    "cs.DS": DOMAIN_SYSTEMS,
}


def classify_domain(paper: Paper) -> str:
    """Assign a paper to a stratification bucket using its arXiv category."""
    metadata = paper.raw_metadata or {}
    primary = metadata.get("primary_category")
    if primary in _CATEGORY_TO_DOMAIN:
        return _CATEGORY_TO_DOMAIN[primary]

    for category in metadata.get("categories") or []:
        if category in _CATEGORY_TO_DOMAIN:
            return _CATEGORY_TO_DOMAIN[category]

    return DOMAIN_OTHER
