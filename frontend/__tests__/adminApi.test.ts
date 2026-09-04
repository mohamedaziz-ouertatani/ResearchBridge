import { afterEach, describe, expect, it, vi } from "vitest";
import { adminApi } from "@/lib/adminApi";

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

describe("adminApi", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("pipelineStatus() requests the pipeline snapshot with no-store caching", async () => {
    mockFetchOnce({ total_papers: 1 });

    await adminApi.pipelineStatus();

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/admin/pipeline`,
      expect.objectContaining({ cache: "no-store" }),
    );
  });

  it("excludePaper() PUTs the excluded flag", async () => {
    mockFetchOnce({ id: "p1", excluded_at: "2026-01-01" });

    await adminApi.excludePaper("p1", true);

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/admin/papers/p1/exclude`,
      expect.objectContaining({
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ excluded: true }),
      }),
    );
  });

  it("triggerArxivIngestion() posts the ingestion params as JSON", async () => {
    mockFetchOnce({ started: true, pipeline: "ingestion_arxiv", log_file: "x.log" });

    await adminApi.triggerArxivIngestion({ search_query: "fraud", page_size: 50 });

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/admin/ingestion/arxiv/run`,
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ search_query: "fraud", page_size: 50 }),
      }),
    );
  });

  it("triggerExtraction() posts to the extraction run endpoint", async () => {
    mockFetchOnce({ started: true, pipeline: "extraction", log_file: "x.log" });

    await adminApi.triggerExtraction({ limit: 10, force: true });

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/admin/extraction/run`,
      expect.objectContaining({ method: "POST", body: JSON.stringify({ limit: 10, force: true }) }),
    );
  });

  it("triggerFulltext() posts to the fulltext run endpoint", async () => {
    mockFetchOnce({ started: true, pipeline: "fulltext", log_file: "x.log" });

    await adminApi.triggerFulltext({ limit: 20 });

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/admin/fulltext/run`,
      expect.objectContaining({ method: "POST", body: JSON.stringify({ limit: 20 }) }),
    );
  });

  it("log() requests the pipeline log with a lines param and unwraps the log field", async () => {
    mockFetchOnce({ log: "line1\nline2" });

    const log = await adminApi.log("extraction", 50);

    expect(fetch).toHaveBeenCalledWith(`${BASE}/api/admin/extraction/log?lines=50`, expect.objectContaining({ cache: "no-store" }));
    expect(log).toBe("line1\nline2");
  });

  it("log() defaults lines to 200", async () => {
    mockFetchOnce({ log: "" });

    await adminApi.log("embedding");

    expect(fetch).toHaveBeenCalledWith(`${BASE}/api/admin/embedding/log?lines=200`, expect.anything());
  });

  it("notifications() requests the notifications feed", async () => {
    mockFetchOnce([]);

    await adminApi.notifications();

    expect(fetch).toHaveBeenCalledWith(`${BASE}/api/admin/notifications`, expect.objectContaining({ cache: "no-store" }));
  });

  it("stopPipeline() POSTs to the pipeline's stop endpoint", async () => {
    mockFetchOnce({ stopped: true, pipeline: "extraction" });

    await adminApi.stopPipeline("extraction");

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/admin/extraction/stop`,
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("retrievalEval() requests the retrieval eval snapshot", async () => {
    mockFetchOnce({ available: false, generated_at: null, k: null, query_sets: null });

    await adminApi.retrievalEval();

    expect(fetch).toHaveBeenCalledWith(`${BASE}/api/admin/retrieval-eval`, expect.objectContaining({ cache: "no-store" }));
  });

  it("a post() helper call surfaces a distinct 'Already running' message on 409", async () => {
    mockFetchOnce(null, { ok: false, status: 409 });

    await expect(adminApi.triggerExtraction({})).rejects.toThrow("Already running");
  });

  it("a post() helper call falls back to a generic message on other non-ok statuses", async () => {
    mockFetchOnce(null, { ok: false, status: 500 });

    await expect(adminApi.triggerExtraction({})).rejects.toThrow("Request failed (500)");
  });

  it("a plain fetch-chain call (pipelineStatus) throws a generic message on a non-ok response", async () => {
    mockFetchOnce(null, { ok: false, status: 503 });

    await expect(adminApi.pipelineStatus()).rejects.toThrow("Request failed (503)");
  });
});
