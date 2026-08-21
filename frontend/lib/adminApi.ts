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

export type PipelineKey = "ingestion_arxiv" | "ingestion_springer" | "extraction" | "embedding";

export type PipelineStatus = {
  total_papers: number;
  papers_with_claims: number;
  papers_with_embeddings: number;
  ingestion_runs: PipelineRun[];
  extraction_runs: PipelineRun[];
  embedding_runs: PipelineRun[];
  running: Record<PipelineKey, boolean>;
};

export type PipelineTriggerResult = {
  started: boolean;
  pipeline: string;
  log_file: string;
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

  triggerExtraction: (params: { limit?: number; extractor?: string }) =>
    post<PipelineTriggerResult>("/api/admin/extraction/run", params),

  triggerEmbedding: (params: { limit?: number }) =>
    post<PipelineTriggerResult>("/api/admin/embedding/run", params),
};
