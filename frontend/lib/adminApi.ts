import { API_BASE, type PaperSummary } from "./api";

export type PipelineRun = {
  id: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  error_summary: string | null;
  counts: Record<string, number>;
};

export type PipelineStatus = {
  total_papers: number;
  papers_with_claims: number;
  papers_with_embeddings: number;
  ingestion_runs: PipelineRun[];
  extraction_runs: PipelineRun[];
  embedding_runs: PipelineRun[];
};

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
};
