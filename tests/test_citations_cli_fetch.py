from __future__ import annotations

import json

from researchbridge.citations.batch import BatchSummary
from researchbridge.citations.cli_fetch import SUMMARY_PATH_BY_SOURCE, _build_summary_json, summary_path_for, write_summary_json


def test_build_summary_json_shapes_batch_summary() -> None:
    summary = BatchSummary(papers_seen=10, papers_failed=1, edges_created=5, edges_already_existed=2)

    result = _build_summary_json(summary)

    assert "generated_at" in result
    assert result["papers_seen"] == 10
    assert result["papers_failed"] == 1
    assert result["edges_created"] == 5
    assert result["edges_already_existed"] == 2


def test_write_summary_json_creates_parent_dir_and_writes_valid_json(tmp_path) -> None:
    output_path = tmp_path / "nested" / "citations_fetch_summary.json"
    summary = BatchSummary(papers_seen=3, papers_failed=0, edges_created=1, edges_already_existed=0)

    write_summary_json(output_path, summary)

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written["papers_seen"] == 3
    assert written["edges_created"] == 1


def test_summary_path_for_gives_each_source_its_own_file() -> None:
    """Semantic Scholar and CrossRef must not clobber each other's last-run
    summary - they're independent sources with independent coverage."""
    assert summary_path_for("semantic_scholar") != summary_path_for("crossref")
    assert summary_path_for("semantic_scholar") == SUMMARY_PATH_BY_SOURCE["semantic_scholar"]
    assert summary_path_for("crossref") == SUMMARY_PATH_BY_SOURCE["crossref"]
