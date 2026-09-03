import { API_BASE, type PaperSummary } from "./api";

export type PipelineRun = {
  id: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  error_summary: string | null;
  source: string | null;
  counts: Record<string, number>;
};

export type PipelineKey =
  | "ingestion_arxiv"
  | "ingestion_springer"
  | "ingestion_semantic_scholar"
  | "ingestion_core"
  | "extraction"
  | "embedding"
  | "retrieval_eval"
  | "extraction_eval"
  | "citations_fetch"
  | "gaps";

export type AssessmentStats = {
  total: number;
  needs_review: number;
};

export type CorpusHealth = {
  missing_doi: number;
  excluded: number;
  claims_without_embeddings: number;
  no_citation_coverage: number;
  not_gap_processed: number;
};

export type GapReviewStats = {
  pending: number;
  approved: number;
  rejected: number;
  mean_correctness: number | null;
  mean_relevance: number | null;
  mean_novelty: number | null;
  mean_evidence_support: number | null;
  mean_usefulness: number | null;
};

export type PipelineStatus = {
  total_papers: number;
  papers_with_claims: number;
  papers_with_embeddings: number;
  papers_by_source: Record<string, number>;
  corpus_health: CorpusHealth;
  assessment_stats: AssessmentStats;
  gap_stats: GapReviewStats;
  ingestion_errors_by_type: Record<string, number>;
  ingestion_runs: PipelineRun[];
  extraction_runs: PipelineRun[];
  embedding_runs: PipelineRun[];
  citation_fetch_runs: PipelineRun[];
  gap_detection_runs: PipelineRun[];
  running: Record<PipelineKey, boolean>;
};

export type PipelineTriggerResult = {
  started: boolean;
  pipeline: string;
  log_file: string;
};

export type Notification = {
  id: string;
  type: string;
  severity: "info" | "error";
  message: string;
  created_at: string;
};

export type RetrievalEvalMethodResult = {
  method: string;
  precision: number;
  recall: number;
  ndcg: number;
  mrr: number;
};

export type RetrievalEvalQuerySet = {
  queries: number;
  skipped: number;
  results: RetrievalEvalMethodResult[];
};

export type RetrievalEvalResult = {
  available: boolean;
  generated_at: string | null;
  k: number | null;
  query_sets: Record<string, RetrievalEvalQuerySet> | null;
};

export type ExtractionEvalFieldScore = {
  precision: number;
  recall: number;
  f1: number;
};

export type ExtractionEvalResult = {
  available: boolean;
  generated_at: string | null;
  threshold: number | null;
  paper_count: number | null;
  extractors: Record<string, Record<string, ExtractionEvalFieldScore>> | null;
};

async function post<T>(path: string, body: Record<string, unknown>): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!response.ok) {
    if (response.status === 409) throw new Error("Already running");
    throw new Error(`Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const adminApi = {
  pipelineStatus: () =>
    fetch(`${API_BASE}/api/admin/pipeline`, { cache: "no-store" }).then((response) => {
      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      return response.json() as Promise<PipelineStatus>;
    }),

  excludePaper: (id: string, excluded: boolean) =>
    fetch(`${API_BASE}/api/admin/papers/${id}/exclude`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ excluded }),
      cache: "no-store",
    }).then((response) => {
      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      return response.json() as Promise<PaperSummary>;
    }),

  triggerArxivIngestion: (params: { search_query?: string; page_size?: number; max_pages?: number }) =>
    post<PipelineTriggerResult>("/api/admin/ingestion/arxiv/run", params),

  triggerSpringerIngestion: (params: { query?: string; page_size?: number; max_pages?: number }) =>
    post<PipelineTriggerResult>("/api/admin/ingestion/springer/run", params),

  triggerSemanticScholarIngestion: (params: { query?: string; max_pages?: number }) =>
    post<PipelineTriggerResult>("/api/admin/ingestion/semantic-scholar/run", params),

  triggerCoreIngestion: (params: { query?: string; page_size?: number; max_pages?: number }) =>
    post<PipelineTriggerResult>("/api/admin/ingestion/core/run", params),

  triggerExtraction: (params: { limit?: number; extractor?: string; force?: boolean }) =>
    post<PipelineTriggerResult>("/api/admin/extraction/run", params),

  triggerEmbedding: (params: { limit?: number; force?: boolean }) =>
    post<PipelineTriggerResult>("/api/admin/embedding/run", params),

  triggerRetrievalEval: (params: { k?: number }) =>
    post<PipelineTriggerResult>("/api/admin/retrieval-eval/run", params),

  retrievalEval: () =>
    fetch(`${API_BASE}/api/admin/retrieval-eval`, { cache: "no-store" }).then((response) => {
      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      return response.json() as Promise<RetrievalEvalResult>;
    }),

  triggerExtractionEval: (params: { threshold?: number; extractor?: string }) =>
    post<PipelineTriggerResult>("/api/admin/extraction-eval/run", params),

  extractionEval: () =>
    fetch(`${API_BASE}/api/admin/extraction-eval`, { cache: "no-store" }).then((response) => {
      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      return response.json() as Promise<ExtractionEvalResult>;
    }),

  triggerCitationsFetch: (params: { source?: string; force?: boolean }) =>
    post<PipelineTriggerResult>("/api/admin/citations-fetch/run", params),

  log: (key: PipelineKey, lines = 200) =>
    fetch(`${API_BASE}/api/admin/${key}/log?lines=${lines}`, { cache: "no-store" }).then((response) => {
      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      return (response.json() as Promise<{ log: string }>).then((body) => body.log);
    }),

  notifications: () =>
    fetch(`${API_BASE}/api/admin/notifications`, { cache: "no-store" }).then((response) => {
      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      return response.json() as Promise<Notification[]>;
    }),

  stopPipeline: (key: PipelineKey) =>
    fetch(`${API_BASE}/api/admin/${key}/stop`, { method: "POST", cache: "no-store" }).then((response) => {
      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      return response.json() as Promise<{ stopped: boolean; pipeline: string }>;
    }),
};
