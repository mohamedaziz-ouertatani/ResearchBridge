from __future__ import annotations

import csv
import uuid
from datetime import date

import yaml

from researchbridge.benchmark.cli_sample import _write_annotation_files, _write_manifest
from researchbridge.db.models import Paper


def _paper(source_id: str) -> Paper:
    return Paper(
        id=uuid.uuid4(),
        source="arxiv",
        source_id=source_id,
        title=f"Paper {source_id}",
        publication_date=date(2024, 1, 1),
        url=f"https://arxiv.org/abs/{source_id}",
        raw_metadata={},
        ingestion_metadata={},
    )


def test_write_annotation_files_creates_one_file_per_paper(tmp_path) -> None:
    sample = {"NLP": [_paper("a"), _paper("b")], "Systems": [_paper("c")]}

    written, skipped = _write_annotation_files(sample, tmp_path)

    assert written == 3
    assert skipped == 0
    files = sorted(p.name for p in (tmp_path / "annotations").glob("*.yaml"))
    assert files == ["arxiv_a.yaml", "arxiv_b.yaml", "arxiv_c.yaml"]


def test_write_annotation_files_never_overwrites_existing(tmp_path) -> None:
    paper = _paper("a")
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir(parents=True)
    existing = annotations_dir / "arxiv_a.yaml"
    existing.write_text("problem: \"already filled in by hand\"\n", encoding="utf-8")

    written, skipped = _write_annotation_files({"NLP": [paper]}, tmp_path)

    assert written == 0
    assert skipped == 1
    assert "already filled in by hand" in existing.read_text(encoding="utf-8")


def test_annotation_file_content_matches_template(tmp_path) -> None:
    paper = _paper("a")
    _write_annotation_files({"NLP": [paper]}, tmp_path)

    content = (tmp_path / "annotations" / "arxiv_a.yaml").read_text(encoding="utf-8")
    parsed = yaml.safe_load(content)
    assert parsed["domain"] == "NLP"
    assert parsed["source_id"] == "a"


def test_write_manifest_lists_every_paper_with_domain(tmp_path) -> None:
    sample = {"NLP": [_paper("a")], "Systems": [_paper("b"), _paper("c")]}

    _write_manifest(sample, tmp_path)

    with (tmp_path / "sample_manifest.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    by_source_id = {row["source_id"]: row for row in rows}
    assert by_source_id["a"]["domain"] == "NLP"
    assert by_source_id["b"]["domain"] == "Systems"
    assert by_source_id["a"]["annotation_file"] == "arxiv_a.yaml"
