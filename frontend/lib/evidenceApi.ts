import { API_BASE } from "./api";
import type { EvidenceReview } from "./assessmentApi";

export type EvidenceReviewPayload = {
  evidence_id: string;
  surface_type: "assessment" | "qa_question";
  surface_id: string;
  role?: string;
  verdict: "supported" | "unclear" | "not_supported";
  note?: string | null;
};

export const evidenceApi = {
  review: async (payload: EvidenceReviewPayload): Promise<EvidenceReview> => {
    const response = await fetch(`${API_BASE}/api/evidence/review`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      throw new Error(detail?.detail ?? `Request failed (${response.status})`);
    }
    return response.json();
  },
};
