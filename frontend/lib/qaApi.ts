import { API_BASE } from "./api";

export type QuoteHit = {
  evidence_id?: string | null;
  paper_id: string;
  paper_title: string;
  paper_source: string;
  claim_type: string;
  text: string;
  section: string | null;
  confidence: string;
  score: number;
};

export type AskResponse = {
  hits: QuoteHit[];
  summarization_available: boolean;
};

export type SummarizeResponse = {
  summary: string;
  citations: number[];
};

export type QaQuestion = {
  id: string;
  collection_id: string;
  question: string;
  hits: QuoteHit[];
  summary: string | null;
  summary_citations: number[] | null;
  created_at: string;
};

export type QaCollectionSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  question_count: number;
};

export type QaCollection = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  questions: QaQuestion[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    cache: "no-store",
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Request failed (${response.status})`);
  }
  return response.json();
}

async function requestNoContent(
  path: string,
  init?: RequestInit,
): Promise<void> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    cache: "no-store",
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Request failed (${response.status})`);
  }
}

export const qaApi = {
  listCollections: () => request<QaCollectionSummary[]>("/api/qa/collections"),
  createCollection: (title: string) =>
    request<QaCollectionSummary>("/api/qa/collections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),
  getCollection: (id: string) =>
    request<QaCollection>(`/api/qa/collections/${id}`),
  renameCollection: (id: string, title: string) =>
    request<QaCollectionSummary>(`/api/qa/collections/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),
  deleteCollection: (id: string) =>
    requestNoContent(`/api/qa/collections/${id}`, { method: "DELETE" }),
  askInCollection: (collectionId: string, question: string) =>
    request<QaQuestion>(`/api/qa/collections/${collectionId}/questions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),
  summarizeQuestion: (questionId: string) =>
    request<QaQuestion>(`/api/qa/questions/${questionId}/summarize`, {
      method: "POST",
    }),

  ask: (question: string) =>
    request<AskResponse>("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),
  summarize: (question: string, hits: QuoteHit[]) =>
    request<SummarizeResponse>("/api/ask/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, hits }),
    }),
};
