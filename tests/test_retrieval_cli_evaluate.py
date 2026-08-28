from __future__ import annotations

import json

from researchbridge.retrieval.cli_evaluate import _build_results_json, write_results_json


def test_build_results_json_shapes_rows_by_query_set() -> None:
    query_set_data = {
        "self": {
            "queries": 5,
            "skipped": 1,
            "rows": [
                ("tfidf", {"precision": 1.0, "recall": 1.0, "ndcg": 1.0, "mrr": 1.0}),
                ("bm25", {"precision": 0.9, "recall": 0.8, "ndcg": 0.85, "mrr": 0.7}),
            ],
        },
        "topical": {
            "queries": 12,
            "skipped": 0,
            "rows": [
                ("tfidf", {"precision": 0.3, "recall": 0.4, "ndcg": 0.35, "mrr": 0.5}),
            ],
        },
    }

    result = _build_results_json(query_set_data, k=10)

    assert result["k"] == 10
    assert "generated_at" in result
    assert result["query_sets"]["self"]["queries"] == 5
    assert result["query_sets"]["self"]["skipped"] == 1
    assert result["query_sets"]["self"]["results"] == [
        {"method": "tfidf", "precision": 1.0, "recall": 1.0, "ndcg": 1.0, "mrr": 1.0},
        {"method": "bm25", "precision": 0.9, "recall": 0.8, "ndcg": 0.85, "mrr": 0.7},
    ]
    assert result["query_sets"]["topical"]["queries"] == 12
    assert result["query_sets"]["topical"]["results"] == [
        {"method": "tfidf", "precision": 0.3, "recall": 0.4, "ndcg": 0.35, "mrr": 0.5},
    ]


def test_build_results_json_handles_a_query_set_with_no_usable_queries() -> None:
    query_set_data = {"self": {"queries": 0, "skipped": 3, "rows": []}}

    result = _build_results_json(query_set_data, k=10)

    assert result["query_sets"]["self"]["results"] == []
    assert result["query_sets"]["self"]["skipped"] == 3


def test_write_results_json_creates_parent_dir_and_writes_valid_json(tmp_path) -> None:
    output_path = tmp_path / "nested" / "retrieval_eval_results.json"
    query_set_data = {"self": {"queries": 1, "skipped": 0, "rows": [("tfidf", {"precision": 1.0, "recall": 1.0, "ndcg": 1.0, "mrr": 1.0})]}}

    write_results_json(output_path, query_set_data, k=5)

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written["k"] == 5
    assert written["query_sets"]["self"]["results"][0]["method"] == "tfidf"
