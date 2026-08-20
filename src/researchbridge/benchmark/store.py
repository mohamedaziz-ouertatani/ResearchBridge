"""Reads and writes the benchmark annotation YAML files.

The files stay the source of truth (not a database table): they are
hand-written research artifacts, they belong in git next to the code they
evaluate, and a half-finished annotation should survive anything the
tooling does.

Writes preserve the identity block verbatim and only replace the annotation
fields, so a file never loses which paper it describes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ANNOTATION_FIELDS = (
    "problem",
    "research_question",
    "method",
    "dataset",
    "main_contribution",
    "results",
    "limitations",
    "applications",
)

IDENTITY_FIELDS = ("source", "source_id", "title", "domain", "year", "url")


@dataclass
class Annotation:
    source_id: str
    path: Path
    identity: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, str] = field(default_factory=dict)
    research_gap: dict[str, str] = field(default_factory=dict)
    key_evidence: list[dict[str, str]] = field(default_factory=list)

    @property
    def filled_count(self) -> int:
        filled = sum(1 for name in ANNOTATION_FIELDS if (self.fields.get(name) or "").strip())
        filled += sum(1 for key in ("addressed", "remaining") if (self.research_gap.get(key) or "").strip())
        return filled

    @property
    def total_count(self) -> int:
        return len(ANNOTATION_FIELDS) + 2  # + research_gap.addressed / .remaining

    @property
    def is_complete(self) -> bool:
        return self.filled_count == self.total_count


def annotation_dir(base: Path) -> Path:
    return base / "annotations"


def load(path: Path) -> Annotation:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    gap = raw.get("research_gap") or {}

    return Annotation(
        source_id=str(raw.get("source_id", path.stem)),
        path=path,
        identity={name: raw.get(name) for name in IDENTITY_FIELDS},
        fields={name: raw.get(name) or "" for name in ANNOTATION_FIELDS},
        research_gap={"addressed": gap.get("addressed") or "", "remaining": gap.get("remaining") or ""},
        key_evidence=list(raw.get("key_evidence") or []),
    )


def load_all(base: Path) -> list[Annotation]:
    directory = annotation_dir(base)
    if not directory.exists():
        return []
    return [load(path) for path in sorted(directory.glob("*.yaml"))]


def save(annotation: Annotation) -> None:
    """Rewrite the file with updated annotation content.

    Written via yaml.safe_dump rather than the comment-carrying template:
    once a human is editing through the workbench, the inline field prompts
    have done their job, and round-tripping them by hand risks corrupting
    real annotation text.
    """
    document: dict[str, Any] = {name: annotation.identity.get(name) for name in IDENTITY_FIELDS}
    document.update({name: annotation.fields.get(name, "") for name in ANNOTATION_FIELDS})
    document["research_gap"] = {
        "addressed": annotation.research_gap.get("addressed", ""),
        "remaining": annotation.research_gap.get("remaining", ""),
    }
    document["key_evidence"] = annotation.key_evidence

    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100)
    header = (
        "# Benchmark annotation - ResearchBridge.md Sec 25.\n"
        "# Written by hand through the annotation workbench.\n\n"
    )
    annotation.path.write_text(header + body, encoding="utf-8")


def apply_updates(annotation: Annotation, payload: dict[str, Any]) -> Annotation:
    """Merge a partial update, ignoring keys that aren't annotation content."""
    for name in ANNOTATION_FIELDS:
        if name in payload:
            annotation.fields[name] = payload[name] or ""

    gap = payload.get("research_gap")
    if isinstance(gap, dict):
        for key in ("addressed", "remaining"):
            if key in gap:
                annotation.research_gap[key] = gap[key] or ""

    evidence = payload.get("key_evidence")
    if isinstance(evidence, list):
        annotation.key_evidence = [
            {"text": str(item.get("text", "")), "section": str(item.get("section", ""))}
            for item in evidence
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        ]

    return annotation
