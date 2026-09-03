import { afterEach, describe, expect, it, vi } from "vitest";
import { qaApi } from "@/lib/qaApi";

function mockFetchOnce(body: unknown, init: { ok?: boolean; status?: number } = {}) {
  const { ok = true, status = 200 } = init;
  const response = {
    ok,
    status,
    json: () => Promise.resolve(body),
  } as Response;
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));
  return response;
}

// API_BASE (lib/api.ts) falls back to this when NEXT_PUBLIC_API_BASE is
// unset, which is always true under Vitest - no env stubbing needed since
// the constant is read once at module load, before any test runs.
const BASE = "http://localhost:8000";

describe("qaApi", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("ask() POSTs the question as JSON", async () => {
    mockFetchOnce({ hits: [], summarization_available: false });

    await qaApi.ask("what is federated learning?");

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/ask`,
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: "what is federated learning?" }),
      }),
    );
  });

  it("summarize() POSTs the question and hits as JSON", async () => {
    mockFetchOnce({ summary: "a summary", citations: [1, 2] });
    const hits = [
      {
        paper_id: "p1",
        paper_title: "Paper",
        paper_source: "arxiv",
        claim_type: "results",
        text: "we achieve 90% accuracy",
        section: null,
        confidence: "high",
        score: 0.8,
      },
    ];

    await qaApi.summarize("how well does it perform?", hits);

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/ask/summarize`,
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: "how well does it perform?", hits }),
      }),
    );
  });

  it("throws the backend's detail message on a non-ok response", async () => {
    mockFetchOnce({ detail: "empty question" }, { ok: false, status: 422 });

    await expect(qaApi.ask("")).rejects.toThrow("empty question");
  });

  it("falls back to a generic message when the error body isn't JSON", async () => {
    const response = {
      ok: false,
      status: 500,
      json: () => Promise.reject(new Error("not json")),
    } as unknown as Response;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));

    await expect(qaApi.ask("x")).rejects.toThrow("Request failed (500)");
  });
});
