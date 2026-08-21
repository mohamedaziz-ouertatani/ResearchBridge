import { API_BASE } from "./api";

export type GapEvidence = {
  paper_id: string;
  paper_title: string;
  text: string;
  section: string | null;
};

export type CandidateGap = {
  id: string;
  seed_paper_id: string;
  seed_paper_title: string;
  observation: string;
  gap_type: string;
  status: "pending" | "approved" | "rejected";
  contributing_paper_count: number;
  similarity_threshold: number;
  detection_method: string;
  review_note: string | null;
  evidence: GapEvidence[];
};

export type CandidateGapPage = {
  items: CandidateGap[];
  total: number;
  limit: number;
  offset: number;
};

export type GapStatusFilter = "pending" | "approved" | "rejected" | "all";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}

export const gapsApi = {
  list: (status: GapStatusFilter = "pending") =>
    request<CandidateGapPage>(`/api/gaps?status=${status}&limit=50`),

  review: (id: string, status: "approved" | "rejected", reviewNote?: string) =>
    request<CandidateGap>(`/api/gaps/${id}`, {
      method: "PUT",
      body: JSON.stringify({ status, review_note: reviewNote || null }),
    }),
};
