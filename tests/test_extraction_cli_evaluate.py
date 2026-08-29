from __future__ import annotations

import json

from researchbridge.extraction.cli_evaluate import _build_results_json, write_results_json
from researchbridge.extraction.evaluation import FieldScore


def test_build_results_json_shapes_scores_by_extractor() -> None:
    scores_by_extractor = {
        "heuristic": {
            "problem": FieldScore("problem", true_positives=3, false_positives=1, false_negatives=1),
            "method": FieldScore("method", true_positives=2, false_positives=0, false_negatives=0),
        },
        "hybrid": {
            "problem": FieldScore("problem", true_positives=4, false_positives=0, false_negatives=0),
        },
    }

    result = _build_results_json(scores_by_extractor, threshold=0.5, paper_count=5)

    assert "generated_at" in result
    assert result["threshold"] == 0.5
    assert result["paper_count"] == 5
    assert result["extractors"]["heuristic"]["problem"] == {
        "precision": 0.75, "recall": 0.75, "f1": 0.75,
    }
    assert result["extractors"]["heuristic"]["method"] == {
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
    }
    assert result["extractors"]["hybrid"]["problem"] == {
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
    }


def test_build_results_json_handles_zero_denominator_scores() -> None:
    scores_by_extractor = {"heuristic": {"problem": FieldScore("problem")}}

    result = _build_results_json(scores_by_extractor, threshold=0.5, paper_count=0)

    assert result["extractors"]["heuristic"]["problem"] == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
    }


def test_write_results_json_creates_parent_dir_and_writes_valid_json(tmp_path) -> None:
    output_path = tmp_path / "nested" / "extraction_eval_results.json"
    scores_by_extractor = {"heuristic": {"problem": FieldScore("problem", true_positives=1)}}

    write_results_json(output_path, scores_by_extractor, threshold=0.5, paper_count=1)

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written["threshold"] == 0.5
    assert written["extractors"]["heuristic"]["problem"]["precision"] == 1.0
