import { API_BASE } from "./api";

export type ClaimEvidence = {
  paper_id: string;
  paper_title: string;
  text: string;
  section: string | null;
  relationship: "supports" | "contradicts" | "contextualizes";
};

export type ClaimType = "fact" | "inference" | "hypothesis" | "opportunity" | "speculation";
export type SourceTable = "candidate_gaps" | "research_assessments";

export type AnalysisClaim = {
  id: string;
  claim_type: ClaimType;
  claim_text: string;
  confidence: string;
  status: "pending" | "approved" | "rejected";
  source_table: SourceTable;
  source_id: string;
  created_at: string;
  evidence: ClaimEvidence[];
};

export type AnalysisClaimPage = {
  items: AnalysisClaim[];
  total: number;
  limit: number;
  offset: number;
};

export type ClaimStatusFilter = AnalysisClaim["status"] | "all";
export type ClaimTypeFilter = ClaimType | "all";
export type SourceTableFilter = SourceTable | "all";

export const CLAIM_TYPE_LABELS: Record<ClaimType, string> = {
  fact: "fact",
  inference: "inference",
  hypothesis: "hypothesis",
  opportunity: "opportunity",
  speculation: "speculation",
};

export const SOURCE_TABLE_LABELS: Record<SourceTable, string> = {
  candidate_gaps: "candidate gap",
  research_assessments: "assessment",
};

// Sec 28's raw per-paper extraction fields - a different vocabulary and a
// different table (extracted_claims) from ClaimType/AnalysisClaim above
// (the Sec 16 analysis-claims layer). No status or source_table: a
// paper's extraction has neither concept.
export type ExtractedClaimType =
  | "problem"
  | "method"
  | "research_question"
  | "main_contribution"
  | "limitations"
  | "results"
  | "dataset"
  | "research_gap"
  | "applications";

export type ExtractedClaim = {
  id: string;
  claim_type: ExtractedClaimType;
  text: string;
  confidence: string;
  section: string | null;
  extraction_method: string;
  paper_id: string;
  paper_title: string;
  created_at: string;
};

export type ExtractedClaimPage = {
  items: ExtractedClaim[];
  total: number;
  limit: number;
  offset: number;
};

export type ExtractedClaimTypeFilter = ExtractedClaimType | "all";

// Ordered by how common each field is corpus-wide (see the admin panel's
// claim-type coverage) rather than alphabetically, matching
// AdminStats.tsx's EXTRACTED_CLAIM_TYPE_ORDER.
export const EXTRACTED_CLAIM_TYPE_LABELS: Record<ExtractedClaimType, string> = {
  problem: "problem",
  method: "method",
  research_question: "research question",
  main_contribution: "main contribution",
  limitations: "limitations",
  results: "results",
  dataset: "dataset",
  research_gap: "research gap",
  applications: "applications",
};

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}

export const claimsApi = {
  list: (
    filters: { status?: ClaimStatusFilter; claim_type?: ClaimTypeFilter; source_table?: SourceTableFilter } = {},
    limit = 20,
    offset = 0,
  ) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (filters.status && filters.status !== "all") params.set("status", filters.status);
    if (filters.claim_type && filters.claim_type !== "all") params.set("claim_type", filters.claim_type);
    if (filters.source_table && filters.source_table !== "all") params.set("source_table", filters.source_table);
    return request<AnalysisClaimPage>(`/api/claims?${params.toString()}`);
  },
};

export const extractedClaimsApi = {
  list: (filters: { claim_type?: ExtractedClaimTypeFilter } = {}, limit = 20, offset = 0) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (filters.claim_type && filters.claim_type !== "all") params.set("claim_type", filters.claim_type);
    return request<ExtractedClaimPage>(`/api/extracted-claims?${params.toString()}`);
  },
};
