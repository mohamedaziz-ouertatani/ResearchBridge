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
