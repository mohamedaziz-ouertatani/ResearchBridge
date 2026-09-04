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

/** The Sec 16 structured-reasoning mirror of a plain-text report field -
 * see assessment/claims.py. "fact" for comparison_summary, "inference" for
 * everything else this backs. Status stays "pending" indefinitely for an
 * assessment-derived claim (no review state to sync against). */
export type AnalysisClaim = {
  id: string;
  claim_type: "fact" | "inference" | "hypothesis" | "opportunity" | "speculation";
  claim_text: string;
  confidence: string;
  status: "pending" | "approved" | "rejected";
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

/** One application already shown above, cited as grounding for a
 * synthesized opportunity - see potential_opportunities below. */
export type OpportunitySourceApplication = {
  application: string;
  paper_id: string;
  paper_title: string;
};

export type PotentialOpportunity = {
  tier: "direct" | "adjacent" | "speculative";
  opportunity: string;
  source_applications: OpportunitySourceApplication[];
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
  potential_opportunities: PotentialOpportunity[] | null;
  risks_and_limitations: string | null;
  external_validation_needed: string;
  recommendation: string | null;
  confidence: string | null;
  human_reviewed: boolean;
  /** Every populated field above traces back to real quoted passages here. */
  evidence: AssessmentEvidence[];
  /** Structured Evidence -> Inference mirror of comparison_summary/
   * novelty_reasoning/research_gap_text/technical_feasibility_reasoning/
   * risks_and_limitations. Empty for an assessment predating this layer. */
  claims: AnalysisClaim[];
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
  technical_feasibility_level: string;
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

export type GraphNode = {
  id: string;
  type: "input" | "paper";
  title: string;
  distance_to_input: number | null;
  claim_counts: Record<string, number>;
};

export type GraphEdge = {
  source: string;
  target: string;
  distance: number;
};

export type GraphData = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type ReviewFilter = "all" | "reviewed" | "needs_review";
export type AssessmentSort = "newest" | "priority";
export type CategoricalLevel = "high" | "medium" | "low" | "not_assessed";

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

  /** Synthesizes Direct/Adjacent/Speculative opportunities via a local LLM
   * (off unless the backend has OLLAMA_ENABLED=true) and persists them -
   * 422 if the assessment has no potential_applications to ground this in,
   * 503 if the local model is unavailable or unable to produce a validly
   * cited result after one retry. See docs/superpowers/specs/
   * 2026-09-03-opportunities-synthesis-design.md. */
  synthesizeOpportunities: (id: string) =>
    request<ResearchAssessment>(`/api/assessments/${id}/opportunities`, { method: "POST" }),

  /** Deletes the whole assessment thread (every rerun for the same input,
   * not just this one id) - see the backend route's docstring. */
  remove: (id: string) => requestNoContent(`/api/assessments/${id}`, { method: "DELETE" }),

  history: (id: string) => request<AssessmentHistoryItem[]>(`/api/assessments/${id}/history`),

  graph: (id: string) => request<GraphData>(`/api/assessments/${id}/graph`),

  list: (
    review: ReviewFilter = "all",
    options?: { sort?: AssessmentSort; novelty?: CategoricalLevel; feasibility?: CategoricalLevel },
  ) => {
    const params = new URLSearchParams({ review, limit: "50" });
    if (options?.sort) params.set("sort", options.sort);
    if (options?.novelty) params.set("novelty", options.novelty);
    if (options?.feasibility) params.set("feasibility", options.feasibility);
    return request<AssessmentSummaryPage>(`/api/assessments?${params.toString()}`);
  },
};
