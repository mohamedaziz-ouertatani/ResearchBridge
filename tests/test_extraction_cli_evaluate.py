from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import Mock

from researchbridge.benchmark.store import Annotation
from researchbridge.db.models import Paper, PaperFullText
from researchbridge.extraction.cli_evaluate import _build_results_json, _evaluate_one, write_results_json
from researchbridge.extraction.evaluation import FieldScore


def _paper(source_id: str, abstract: str = "An abstract.") -> Paper:
    return Paper(
        id=uuid.uuid4(), source="arxiv", source_id=source_id, title="t", abstract=abstract,
        raw_metadata={}, ingestion_metadata={},
    )


def _annotation(source_id: str) -> Annotation:
    return Annotation(
        source_id=source_id,
        path=Path(f"{source_id}.json"),
        fields={
            "problem": "", "research_question": "", "method": "", "dataset": "",
            "main_contribution": "", "results": "", "limitations": "", "applications": "",
        },
        research_gap={"addressed": "", "remaining": ""},
        key_evidence=[],
    )


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


def test_evaluate_one_passes_empty_sections_when_paper_has_no_fulltext_row(session_factory) -> None:
    session = session_factory()
    paper = _paper("P1")
    session.add(paper)
    session.commit()

    extractor = Mock()
    extractor.extract.return_value = []
    annotation = _annotation("P1")

    _evaluate_one(session, extractor, [annotation], {"P1": paper}, embedder=Mock(), threshold=0.5)

    extractor.extract.assert_called_once_with(paper, {})
    session.close()


def test_evaluate_one_passes_a_papers_own_sections_when_a_fulltext_row_exists(session_factory) -> None:
    session = session_factory()
    paper = _paper("P1")
    session.add(paper)
    session.flush()
    session.add(
        PaperFullText(
            paper_id=paper.id, sections={"methods": "We propose a method."},
            source_url="https://example.com/p1.pdf",
        )
    )
    session.commit()

    extractor = Mock()
    extractor.extract.return_value = []
    annotation = _annotation("P1")

    _evaluate_one(session, extractor, [annotation], {"P1": paper}, embedder=Mock(), threshold=0.5)

    extractor.extract.assert_called_once_with(paper, {"methods": "We propose a method."})
    session.close()
