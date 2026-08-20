"""Builds the per-paper annotation file for the Phase 1 benchmark (Sec 25).

Output is hand-written YAML (not yaml.dump) so the field prompts from
Sec 25 can ship as inline comments next to each empty value - the whole
point of the file is to guide a human through filling it in.
"""

from __future__ import annotations

import json

from researchbridge.db.models import Paper


def annotation_filename(paper: Paper) -> str:
    return f"{paper.source}_{paper.source_id}.yaml"


def render_annotation_template(paper: Paper, domain: str) -> str:
    year = paper.publication_date.year if paper.publication_date else ""
    title = _yaml_str(paper.title)
    url = _yaml_str(paper.url or "")

    return f"""\
# Benchmark annotation - fill in every field below by hand.
# Source of truth: ResearchBridge.md Sec 25 (Annotation Schema).
# Leave a field blank only if the paper genuinely doesn't state it.

paper_id: "{paper.id}"
source: {paper.source}
source_id: "{paper.source_id}"
title: {title}
domain: "{domain}"
year: {year}
url: {url}

# --- Annotation ---

problem: ""                # What problem does the paper address?
research_question: ""      # What research question is being investigated?
method: ""                 # What approach/method is proposed?
dataset: ""                # What data is used?
main_contribution: ""      # What is actually new?
results: ""                # What are the major findings?
limitations: ""            # What limitations are explicitly stated?

research_gap:
  addressed: ""            # What gap does the paper address?
  remaining: ""            # What gap remains?

applications: ""           # What applications are explicitly supported?

key_evidence: []           # Passages supporting the above, e.g.:
                            #   - text: "quoted passage from the paper"
                            #     section: "Introduction"
"""


def _yaml_str(value: str) -> str:
    """Double-quoted YAML scalar via JSON escaping (YAML's double-quoted form is JSON-compatible)."""
    return json.dumps(value)
