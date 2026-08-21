from __future__ import annotations

import uuid

import yaml

from researchbridge.db.models import Paper
from researchbridge.retrieval.topical_queries import build_topical_query_set


def _write_yaml(path, queries: list[dict]) -> None:
    path.write_text(yaml.safe_dump({"queries": queries}), encoding="utf-8")


def _corpus_paper(session, source_id: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=source_id, title=f"Title {source_id}",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    return paper


def test_missing_file_returns_empty(tmp_path) -> None:
    judgments, skipped = build_topical_query_set(None, tmp_path / "nope.yaml")
    assert judgments == []
    assert skipped == []


def test_loads_query_with_multiple_relevant_docs(tmp_path, session_factory) -> None:
    session = session_factory()
    a = _corpus_paper(session, "1111.11111")
    b = _corpus_paper(session, "2222.22222")
    session.commit()

    path = tmp_path / "queries.yaml"
    _write_yaml(path, [{"query": "graph neural networks", "relevant_source_ids": ["1111.11111", "2222.22222"]}])

    judgments, skipped = build_topical_query_set(session, path)

    session.close()
    assert skipped == []
    assert len(judgments) == 1
    assert judgments[0].query == "graph neural networks"
    assert judgments[0].relevant_ids == {a.id, b.id}


def test_multiple_queries_each_become_a_judgment(tmp_path, session_factory) -> None:
    session = session_factory()
    _corpus_paper(session, "1111.11111")
    _corpus_paper(session, "2222.22222")
    session.commit()

    path = tmp_path / "queries.yaml"
    _write_yaml(
        path,
        [
            {"query": "topic A", "relevant_source_ids": ["1111.11111"]},
            {"query": "topic B", "relevant_source_ids": ["2222.22222"]},
        ],
    )

    judgments, _ = build_topical_query_set(session, path)

    session.close()
    assert {j.query for j in judgments} == {"topic A", "topic B"}


def test_unresolved_source_id_is_skipped_not_silently_dropped(tmp_path, session_factory) -> None:
    session = session_factory()
    _corpus_paper(session, "1111.11111")
    session.commit()

    path = tmp_path / "queries.yaml"
    _write_yaml(path, [{"query": "topic A", "relevant_source_ids": ["1111.11111", "9999.99999"]}])

    judgments, skipped = build_topical_query_set(session, path)

    session.close()
    assert skipped == ["9999.99999"]
    assert len(judgments[0].relevant_ids) == 1  # the one resolvable paper still counts


def test_query_with_zero_resolvable_relevant_docs_is_dropped_entirely(tmp_path, session_factory) -> None:
    session = session_factory()
    session.commit()  # empty corpus

    path = tmp_path / "queries.yaml"
    _write_yaml(path, [{"query": "topic A", "relevant_source_ids": ["9999.99999"]}])

    judgments, skipped = build_topical_query_set(session, path)

    session.close()
    assert judgments == []
    assert skipped == ["9999.99999"]


def test_empty_queries_list_returns_empty(tmp_path, session_factory) -> None:
    session = session_factory()
    path = tmp_path / "queries.yaml"
    _write_yaml(path, [])

    judgments, skipped = build_topical_query_set(session, path)

    session.close()
    assert judgments == []
    assert skipped == []
