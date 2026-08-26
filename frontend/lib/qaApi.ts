import { API_BASE } from "./api";

export type QuoteHit = {
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
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { ...init, cache: "no-store" });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Request failed (${response.status})`);
  }
  return response.json();
}

export const qaApi = {
  ask: (question: string) =>
    request<AskResponse>("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),
};
