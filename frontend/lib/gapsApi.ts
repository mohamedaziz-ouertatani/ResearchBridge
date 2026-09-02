import { API_BASE } from "./api";

export type GapEvidence = {
  paper_id: string;
  paper_title: string;
  text: string;
  section: string | null;
  claim_type: string;
  validation_tier: "strong" | "weak" | null;
  claim_role: "anchor" | "supporting" | null;
  self_resolution_signal: boolean;
  field_scope_signal: boolean;
  own_contribution_overlap: number;
};

export type GapRatings = {
  correctness_rating: number | null;
  relevance_rating: number | null;
  novelty_rating: number | null;
  evidence_support_rating: number | null;
  usefulness_rating: number | null;
};

export const RATING_DIMENSIONS: { key: keyof GapRatings; label: string }[] = [
  { key: "correctness_rating", label: "correctness" },
  { key: "relevance_rating", label: "relevance" },
  { key: "novelty_rating", label: "novelty" },
  { key: "evidence_support_rating", label: "evidence support" },
  { key: "usefulness_rating", label: "usefulness" },
];

export type CandidateGap = GapRatings & {
  id: string;
  seed_paper_id: string;
  seed_paper_title: string;
  observation: string;
  gap_type: string;
  status: "pending" | "approved" | "rejected";
  gap_status: "strong_gap" | "potential_gap" | "known_limitation" | null;
  resolution_note: string | null;
  contributing_paper_count: number;
  similarity_threshold: number;
  detection_method: string;
  review_note: string | null;
  evidence: GapEvidence[];
};

export const GAP_STATUS_LABELS: Record<NonNullable<CandidateGap["gap_status"]>, string> = {
  strong_gap: "strong gap",
  potential_gap: "potential gap",
  known_limitation: "known limitation",
};

export type CandidateGapPage = {
  items: CandidateGap[];
  total: number;
  limit: number;
  offset: number;
};

export type GapStatusFilter = "pending" | "approved" | "rejected" | "all";

export type GapsDetectStatus = {
  running: boolean;
  log: string;
};

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

  review: (id: string, status: "approved" | "rejected", reviewNote?: string, ratings?: Partial<GapRatings>) =>
    request<CandidateGap>(`/api/gaps/${id}`, {
      method: "PUT",
      body: JSON.stringify({ status, review_note: reviewNote || null, ...ratings }),
    }),

  detect: () => request<{ started: boolean; pipeline: string; log_file: string }>(`/api/gaps/detect`, { method: "POST" }),

  detectStatus: () => request<GapsDetectStatus>(`/api/gaps/detect/status`),
};
