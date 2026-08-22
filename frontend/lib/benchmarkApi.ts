import { API_BASE } from "./api";

export type EvidenceItem = { text: string; section: string };

export type AnnotationSummary = {
  source_id: string;
  title: string | null;
  domain: string | null;
  year: number | null;
  filled: number;
  total: number;
  is_complete: boolean;
  has_fulltext: boolean;
  has_fulltext_nougat: boolean;
};

export type AnnotationDetail = AnnotationSummary & {
  url: string | null;
  fields: Record<string, string>;
  research_gap: { addressed: string; remaining: string };
  key_evidence: EvidenceItem[];
  fulltext: string | null;
  fulltext_nougat: string | null;
};

export type BenchmarkProgress = {
  papers: number;
  complete: number;
  fields_filled: number;
  fields_total: number;
};

/** Field order and prompts come straight from ResearchBridge.md Sec 25. */
export const ANNOTATION_FIELDS: { name: string; label: string; prompt: string }[] = [
  { name: "problem", label: "Problem", prompt: "What problem does the paper address?" },
  { name: "research_question", label: "Research question", prompt: "What is being investigated?" },
  { name: "method", label: "Method", prompt: "What approach is proposed?" },
  { name: "dataset", label: "Dataset", prompt: "What data is used?" },
  { name: "main_contribution", label: "Main contribution", prompt: "What is actually new?" },
  { name: "results", label: "Results", prompt: "What are the major findings?" },
  { name: "limitations", label: "Limitations", prompt: "What limitations are explicitly stated?" },
  { name: "applications", label: "Applications", prompt: "What applications are explicitly supported?" },
];

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}

export const benchmarkApi = {
  list: () => request<AnnotationSummary[]>("/api/benchmark/papers"),
  progress: () => request<BenchmarkProgress>("/api/benchmark/progress"),
  detail: (sourceId: string) => request<AnnotationDetail>(`/api/benchmark/papers/${sourceId}`),
  save: (sourceId: string, payload: Record<string, unknown>) =>
    request<AnnotationSummary>(`/api/benchmark/papers/${sourceId}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
};
