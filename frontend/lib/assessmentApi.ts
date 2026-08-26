import { API_BASE } from "./api";

/** Which report field a quoted passage backs. */
export type EvidenceRole =
  | "comparison"
  | "novelty"
  | "research_gap"
  | "application"
  | "feasibility"
  | "risk"
  | "opportunity";

export type AssessmentEvidence = {
  role: EvidenceRole;
  paper_id: string;
  paper_title: string;
  text: string;
  section: string | null;
};

export type ResearchInput = {
  id: string;
  input_type: "idea" | "document";
  raw_text: string;
  title: string | null;
  matched_paper_id: string | null;
};

export type PotentialApplication = {
  application: string;
  source_paper: string;
  paper_id: string;
};

export type ResearchAssessment = {
  id: string;
  research_input: ResearchInput;
  status: string;
  retrieved_paper_ids: string[];
  comparison_summary: string | null;
  novelty_level: string;
  novelty_reasoning: string | null;
  research_gap_text: string | null;
  research_gap_source: string | null;
  candidate_gap_id: string | null;
  potential_applications: PotentialApplication[] | null;
  technical_feasibility_level: string;
  technical_feasibility_reasoning: string | null;
  potential_opportunities: unknown[] | null;
  risks_and_limitations: string | null;
  external_validation_needed: string;
  recommendation: string | null;
  confidence: string | null;
  human_reviewed: boolean;
  /** Every populated field above traces back to real quoted passages here. */
  evidence: AssessmentEvidence[];
};

export type AssessmentHistoryItem = {
  id: string;
  status: string;
  novelty_level: string;
  human_reviewed: boolean;
  created_at: string;
};

export type AssessmentSummary = {
  id: string;
  created_at: string;
  status: string;
  novelty_level: string;
  recommendation: string | null;
  confidence: string | null;
  human_reviewed: boolean;
  research_input_id: string;
  input_type: "idea" | "document";
  input_preview: string;
};

export type AssessmentSummaryPage = {
  items: AssessmentSummary[];
  total: number;
  limit: number;
  offset: number;
};

export type ReviewFilter = "all" | "reviewed" | "needs_review";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { ...init, cache: "no-store" });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Request failed (${response.status})`);
  }
  return response.json();
}

async function requestNoContent(path: string, init?: RequestInit): Promise<void> {
  const response = await fetch(`${API_BASE}${path}`, { ...init, cache: "no-store" });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Request failed (${response.status})`);
  }
}

export const assessmentApi = {
  create: (rawText: string) =>
    request<ResearchAssessment>("/api/assessments", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ raw_text: rawText }),
    }),

  // no Content-Type header: the browser sets the multipart boundary itself
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<ResearchAssessment>("/api/assessments/upload", { method: "POST", body });
  },

  get: (id: string) => request<ResearchAssessment>(`/api/assessments/${id}`),

  review: (id: string, humanReviewed: boolean) =>
    request<ResearchAssessment>(`/api/assessments/${id}/review`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ human_reviewed: humanReviewed }),
    }),

  rerun: (id: string) => request<ResearchAssessment>(`/api/assessments/${id}/rerun`, { method: "POST" }),

  /** Deletes the whole assessment thread (every rerun for the same input,
   * not just this one id) - see the backend route's docstring. */
  remove: (id: string) => requestNoContent(`/api/assessments/${id}`, { method: "DELETE" }),

  history: (id: string) => request<AssessmentHistoryItem[]>(`/api/assessments/${id}/history`),

  list: (review: ReviewFilter = "all") =>
    request<AssessmentSummaryPage>(`/api/assessments?review=${review}&limit=50`),
};
