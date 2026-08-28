from __future__ import annotations

import json

from researchbridge.citations.batch import BatchSummary
from researchbridge.citations.cli_fetch import _build_summary_json, write_summary_json


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
