export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export type PaperSummary = {
  id: string;
  source: string;
  source_id: string;
  title: string;
  abstract: string | null;
  publication_date: string | null;
  url: string | null;
  primary_category: string | null;
  categories: string[];
  authors: string[];
};

export type ExtractedClaim = {
  claim_type: string;
  text: string;
  /** The extractor's own self-reported confidence ("medium" | "low") - not a validated accuracy score. */
  confidence: string;
  section: string | null;
  extraction_method: string;
};

export type SearchHit = {
  paper: PaperSummary;
  /** pgvector cosine distance: 0 is identical, larger is further apart in meaning. */
  distance: number;
};

export type PaperPage = {
  items: PaperSummary[];
  total: number;
  limit: number;
  offset: number;
};

export type CorpusStats = {
  total_papers: number;
  total_authors: number;
  embedded_papers: number;
  papers_by_year: Record<string, number>;
  papers_by_category: Record<string, number>;
};

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function get<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(`${API_BASE}${path}`);
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== "") url.searchParams.set(key, String(value));
  }

  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new ApiError(detail?.detail ?? `Request failed (${response.status})`, response.status);
  }
  return response.json();
}

export const api = {
  stats: () => get<CorpusStats>("/api/stats"),

  papers: (params: { limit?: number; offset?: number; year?: number; category?: string; q?: string }) =>
    get<PaperPage>("/api/papers", params),

  paper: (id: string) => get<PaperSummary>(`/api/papers/${id}`),

  similar: (id: string, topK = 8) => get<SearchHit[]>(`/api/papers/${id}/similar`, { top_k: topK }),

  claims: (id: string) => get<ExtractedClaim[]>(`/api/papers/${id}/claims`),

  search: (q: string, topK = 12) => get<SearchHit[]>("/api/search", { q, top_k: topK }),
};

export { ApiError };
