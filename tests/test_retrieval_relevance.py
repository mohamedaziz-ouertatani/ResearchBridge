from __future__ import annotations

import uuid
from datetime import date

from researchbridge.benchmark.annotation_template import render_annotation_template
from researchbridge.benchmark.store import apply_updates, load, save
from researchbridge.db.models import Paper
from researchbridge.retrieval.relevance import build_query_set


def _write_template(benchmark_dir, source_id: str, domain: str = "NLP") -> None:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=source_id, title=f"Title {source_id}",
        publication_date=date(2024, 1, 1), url=f"https://arxiv.org/abs/{source_id}",
        raw_metadata={}, ingestion_metadata={},
    )
    directory = benchmark_dir / "annotations"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"arxiv_{source_id}.yaml").write_text(
        render_annotation_template(paper, domain=domain), encoding="utf-8"
    )


def _fill(benchmark_dir, source_id: str, **fields) -> None:
    path = benchmark_dir / "annotations" / f"arxiv_{source_id}.yaml"
    annotation = load(path)
    apply_updates(annotation, fields)
    save(annotation)


def _corpus_paper(session, source_id: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="arxiv", source_id=source_id, title=f"Title {source_id}",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    return paper


def test_uses_research_question_as_query(tmp_path, session_factory) -> None:
    session = session_factory()
    _corpus_paper(session, "1111.11111")
    session.commit()

    _write_template(tmp_path, "1111.11111")
    _fill(tmp_path, "1111.11111", research_question="How does X affect Y?", problem="A different problem statement.")

    judgments, skipped = build_query_set(session, tmp_path)

    session.close()
    assert skipped == []
    assert judgments[0].query == "How does X affect Y?"


def test_falls_back_to_problem_when_research_question_is_blank(tmp_path, session_factory) -> None:
    session = session_factory()
    _corpus_paper(session, "1111.11111")
    session.commit()

    _write_template(tmp_path, "1111.11111")
    _fill(tmp_path, "1111.11111", problem="The paper addresses Z.")

    judgments, skipped = build_query_set(session, tmp_path)

    session.close()
    assert judgments[0].query == "The paper addresses Z."


def test_relevant_ids_is_the_papers_own_id(tmp_path, session_factory) -> None:
    session = session_factory()
    paper = _corpus_paper(session, "1111.11111")
    session.commit()
    paper_id = paper.id

    _write_template(tmp_path, "1111.11111")
    _fill(tmp_path, "1111.11111", research_question="Some question.")

    judgments, _ = build_query_set(session, tmp_path)

    session.close()
    assert judgments[0].relevant_ids == {paper_id}


def test_paper_with_no_query_text_is_skipped(tmp_path, session_factory) -> None:
    session = session_factory()
    _corpus_paper(session, "1111.11111")
    session.commit()

    _write_template(tmp_path, "1111.11111")  # never filled in - both fields blank

    judgments, skipped = build_query_set(session, tmp_path)

    session.close()
    assert judgments == []
    assert skipped == ["1111.11111"]


def test_paper_missing_from_corpus_is_skipped(tmp_path, session_factory) -> None:
    session = session_factory()
    # note: no matching Paper row is created in the corpus at all
    session.commit()

    _write_template(tmp_path, "9999.99999")
    _fill(tmp_path, "9999.99999", research_question="Some question.")

    judgments, skipped = build_query_set(session, tmp_path)

    session.close()
    assert judgments == []
    assert skipped == ["9999.99999"]


def test_multiple_papers_each_become_a_query(tmp_path, session_factory) -> None:
    session = session_factory()
    _corpus_paper(session, "1111.11111")
    _corpus_paper(session, "2222.22222")
    session.commit()

    for sid in ("1111.11111", "2222.22222"):
        _write_template(tmp_path, sid)
        _fill(tmp_path, sid, research_question=f"Question for {sid}")

    judgments, skipped = build_query_set(session, tmp_path)

    session.close()
    assert skipped == []
    assert {j.label for j in judgments} == {"1111.11111", "2222.22222"}
