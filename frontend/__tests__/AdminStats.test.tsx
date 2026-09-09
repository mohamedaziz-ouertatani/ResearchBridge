import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { AdminStats } from "@/components/AdminStats";
import { api, type CorpusStats } from "@/lib/api";
import { assessmentApi } from "@/lib/assessmentApi";
import type { PipelineStatus } from "@/lib/adminApi";

const status = {
  total_papers: 4,
  papers_with_claims: 3,
  papers_with_embeddings: 2,
  papers_with_fulltext: 2,
  papers_by_source: { arxiv: 4 },
  corpus_health: {
    missing_doi: 0,
    excluded: 0,
    claims_without_embeddings: 1,
    no_citation_coverage: 0,
    not_gap_processed: 0,
  },
  assessment_stats: { total: 0, needs_review: 0 },
  gap_stats: {
    pending: 0,
    approved: 0,
    rejected: 0,
    mean_correctness: null,
    mean_relevance: null,
    mean_novelty: null,
    mean_evidence_support: null,
    mean_usefulness: null,
  },
  ingestion_errors_by_type: {},
  extraction_errors_by_type: { extractor_error: 2 },
  fulltext_errors_by_type: { pdf_parse: 1 },
  analysis_claims_by_type: {},
  ingestion_runs: [],
  extraction_runs: [],
  embedding_runs: [],
  citation_fetch_runs: [],
  gap_detection_runs: [],
  fulltext_fetch_runs: [],
  running: {
    ingestion_arxiv: false,
    ingestion_springer: false,
    ingestion_semantic_scholar: false,
    ingestion_core: false,
    extraction: false,
    embedding: false,
    retrieval_eval: false,
    extraction_eval: false,
    citations_fetch: false,
    gaps: false,
    fulltext: false,
  },
} as PipelineStatus;

const corpusStats: CorpusStats = {
  total_papers: 4,
  total_authors: 0,
  embedded_papers: 2,
  papers_with_claims: 3,
  papers_with_fulltext: 2,
  papers_by_year: { "2026": 4 },
  papers_by_category: { "cs.LG": 4 },
  papers_by_source: { arxiv: 4 },
  papers_by_language: { unknown: 3, en: 1 },
};

describe("AdminStats", () => {
  afterEach(() => vi.restoreAllMocks());

  it("renders language coverage and recent extraction/full-text errors", async () => {
    vi.spyOn(api, "stats").mockResolvedValue(corpusStats);
    vi.spyOn(assessmentApi, "list").mockResolvedValue({
      items: [],
      total: 0,
      limit: 50,
      offset: 0,
    });

    render(<AdminStats status={status} />);

    await waitFor(() =>
      expect(screen.getByText("papers by language")).toBeInTheDocument(),
    );
    expect(screen.getByText("unknown")).toBeInTheDocument();
    expect(
      screen.getByText("recent extraction errors by type"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("recent full-text errors by type"),
    ).toBeInTheDocument();
  });
});
