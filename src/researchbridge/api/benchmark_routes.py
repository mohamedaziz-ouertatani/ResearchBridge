"""Annotation workbench endpoints.

Unlike the corpus routes these mutate: they write the researcher's own
annotation text back to the YAML files, which are the benchmark's source
of truth. Nothing here generates annotation content - that has to stay
human-written, or the benchmark can't serve as ground truth for evaluating
extraction later (Sec 24).
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from researchbridge.benchmark.fulltext import fulltext_path
from researchbridge.benchmark.store import (
    ANNOTATION_FIELDS,
    Annotation,
    annotation_dir,
    apply_updates,
    load,
    load_all,
    save,
)

router = APIRouter(prefix="/api/benchmark")


def get_benchmark_dir() -> Path:
    return Path(os.environ.get("BENCHMARK_DIR", "benchmark"))


class EvidenceItem(BaseModel):
    text: str = ""
    section: str = ""


class AnnotationPayload(BaseModel):
    problem: str | None = None
    research_question: str | None = None
    method: str | None = None
    dataset: str | None = None
    main_contribution: str | None = None
    results: str | None = None
    limitations: str | None = None
    applications: str | None = None
    research_gap: dict[str, str] | None = None
    key_evidence: list[EvidenceItem] | None = None


class AnnotationSummary(BaseModel):
    source_id: str
    title: str | None
    domain: str | None
    year: int | None
    filled: int
    total: int
    is_complete: bool
    has_fulltext: bool


class AnnotationDetail(AnnotationSummary):
    url: str | None
    fields: dict[str, str]
    research_gap: dict[str, str]
    key_evidence: list[EvidenceItem]
    fulltext: str | None


class BenchmarkProgress(BaseModel):
    papers: int
    complete: int
    fields_filled: int
    fields_total: int


def _summary(annotation: Annotation, benchmark_dir: Path) -> AnnotationSummary:
    identity = annotation.identity
    year = identity.get("year")
    return AnnotationSummary(
        source_id=annotation.source_id,
        title=identity.get("title"),
        domain=identity.get("domain"),
        year=int(year) if year else None,
        filled=annotation.filled_count,
        total=annotation.total_count,
        is_complete=annotation.is_complete,
        has_fulltext=fulltext_path(benchmark_dir / "fulltext", annotation.source_id).exists(),
    )


def _find(source_id: str, benchmark_dir: Path) -> Annotation:
    path = annotation_dir(benchmark_dir) / f"arxiv_{source_id}.yaml"
    if not path.exists():
        # fall back to a scan: the filename encodes source, which may differ later
        for candidate in load_all(benchmark_dir):
            if candidate.source_id == source_id:
                return candidate
        raise HTTPException(status_code=404, detail=f"No benchmark annotation for {source_id}")
    return load(path)


@router.get("/papers", response_model=list[AnnotationSummary])
def list_annotations(benchmark_dir: Path = Depends(get_benchmark_dir)) -> list[AnnotationSummary]:
    return [_summary(a, benchmark_dir) for a in load_all(benchmark_dir)]


@router.get("/progress", response_model=BenchmarkProgress)
def progress(benchmark_dir: Path = Depends(get_benchmark_dir)) -> BenchmarkProgress:
    annotations = load_all(benchmark_dir)
    return BenchmarkProgress(
        papers=len(annotations),
        complete=sum(1 for a in annotations if a.is_complete),
        fields_filled=sum(a.filled_count for a in annotations),
        fields_total=sum(a.total_count for a in annotations),
    )


@router.get("/papers/{source_id}", response_model=AnnotationDetail)
def get_annotation(
    source_id: str, benchmark_dir: Path = Depends(get_benchmark_dir)
) -> AnnotationDetail:
    annotation = _find(source_id, benchmark_dir)
    path = fulltext_path(benchmark_dir / "fulltext", source_id)

    return AnnotationDetail(
        **_summary(annotation, benchmark_dir).model_dump(),
        url=annotation.identity.get("url"),
        fields={name: annotation.fields.get(name, "") for name in ANNOTATION_FIELDS},
        research_gap=annotation.research_gap,
        key_evidence=[EvidenceItem(**item) for item in annotation.key_evidence],
        fulltext=path.read_text(encoding="utf-8") if path.exists() else None,
    )


@router.put("/papers/{source_id}", response_model=AnnotationSummary)
def update_annotation(
    source_id: str,
    payload: AnnotationPayload,
    benchmark_dir: Path = Depends(get_benchmark_dir),
) -> AnnotationSummary:
    annotation = _find(source_id, benchmark_dir)
    apply_updates(annotation, payload.model_dump(exclude_unset=True))
    save(annotation)
    return _summary(annotation, benchmark_dir)
