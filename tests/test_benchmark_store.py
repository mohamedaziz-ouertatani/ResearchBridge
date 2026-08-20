from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import yaml

from researchbridge.benchmark.annotation_template import render_annotation_template
from researchbridge.benchmark.store import apply_updates, load, load_all, save
from researchbridge.db.models import Paper


def _write_template(base: Path, source_id: str = "2401.01234") -> Path:
    paper = Paper(
        id=uuid.uuid4(),
        source="arxiv",
        source_id=source_id,
        title='A "Tricky" Title: With Colons & Quotes',
        publication_date=date(2024, 3, 1),
        url=f"https://arxiv.org/abs/{source_id}",
        raw_metadata={},
        ingestion_metadata={},
    )
    directory = base / "annotations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"arxiv_{source_id}.yaml"
    path.write_text(render_annotation_template(paper, domain="NLP"), encoding="utf-8")
    return path


def test_load_reads_identity_and_empty_fields(tmp_path) -> None:
    annotation = load(_write_template(tmp_path))

    assert annotation.source_id == "2401.01234"
    assert annotation.identity["domain"] == "NLP"
    assert annotation.fields["problem"] == ""
    assert annotation.filled_count == 0
    assert annotation.total_count == 10  # 8 flat fields + 2 research_gap keys
    assert annotation.is_complete is False


def test_save_round_trips_annotation_content(tmp_path) -> None:
    path = _write_template(tmp_path)
    annotation = load(path)

    apply_updates(
        annotation,
        {
            "problem": "Retrieval degrades on long documents.",
            "research_gap": {"addressed": "No long-context benchmark existed."},
            "key_evidence": [{"text": "We observe a 12% drop.", "section": "Results"}],
        },
    )
    save(annotation)

    reloaded = load(path)
    assert reloaded.fields["problem"] == "Retrieval degrades on long documents."
    assert reloaded.research_gap["addressed"] == "No long-context benchmark existed."
    assert reloaded.research_gap["remaining"] == ""
    assert reloaded.key_evidence == [{"text": "We observe a 12% drop.", "section": "Results"}]


def test_save_preserves_identity_block(tmp_path) -> None:
    path = _write_template(tmp_path)
    annotation = load(path)

    apply_updates(annotation, {"method": "A two-stage reranker."})
    save(annotation)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["source"] == "arxiv"
    assert raw["source_id"] == "2401.01234"
    assert raw["domain"] == "NLP"
    assert raw["title"] == 'A "Tricky" Title: With Colons & Quotes'


def test_filled_count_tracks_progress(tmp_path) -> None:
    annotation = load(_write_template(tmp_path))

    apply_updates(annotation, {"problem": "x", "method": "y"})
    assert annotation.filled_count == 2

    apply_updates(annotation, {"research_gap": {"addressed": "z", "remaining": "w"}})
    assert annotation.filled_count == 4


def test_whitespace_only_does_not_count_as_filled(tmp_path) -> None:
    annotation = load(_write_template(tmp_path))

    apply_updates(annotation, {"problem": "   \n  "})

    assert annotation.filled_count == 0


def test_apply_updates_ignores_unknown_keys(tmp_path) -> None:
    annotation = load(_write_template(tmp_path))

    apply_updates(annotation, {"problem": "kept", "source_id": "hacked", "nonsense": "ignored"})

    assert annotation.fields["problem"] == "kept"
    assert annotation.identity["source_id"] == "2401.01234"  # identity is not writable through updates


def test_apply_updates_drops_empty_evidence_entries(tmp_path) -> None:
    annotation = load(_write_template(tmp_path))

    apply_updates(
        annotation,
        {"key_evidence": [{"text": "real", "section": "Intro"}, {"text": "   ", "section": "Discussion"}]},
    )

    assert annotation.key_evidence == [{"text": "real", "section": "Intro"}]


def test_load_all_returns_every_annotation(tmp_path) -> None:
    _write_template(tmp_path, "1111.11111")
    _write_template(tmp_path, "2222.22222")

    assert {a.source_id for a in load_all(tmp_path)} == {"1111.11111", "2222.22222"}


def test_load_all_on_missing_directory_is_empty(tmp_path) -> None:
    assert load_all(tmp_path / "nope") == []
