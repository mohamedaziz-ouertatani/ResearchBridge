import { afterEach, describe, expect, it, vi } from "vitest";
import { benchmarkApi } from "@/lib/benchmarkApi";

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

describe("benchmarkApi", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("list() requests the annotation papers list", async () => {
    mockFetchOnce([]);

    await benchmarkApi.list();

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/benchmark/papers`,
      expect.objectContaining({ cache: "no-store" }),
    );
  });

  it("progress() requests the benchmark progress summary", async () => {
    mockFetchOnce({ papers: 10, complete: 3, fields_filled: 24, fields_total: 80 });

    await benchmarkApi.progress();

    expect(fetch).toHaveBeenCalledWith(`${BASE}/api/benchmark/progress`, expect.objectContaining({ cache: "no-store" }));
  });

  it("detail() requests a single paper's annotation detail by source id", async () => {
    mockFetchOnce({ source_id: "s1" });

    await benchmarkApi.detail("s1");

    expect(fetch).toHaveBeenCalledWith(`${BASE}/api/benchmark/papers/s1`, expect.objectContaining({ cache: "no-store" }));
  });

  it("save() PUTs the field payload as JSON", async () => {
    mockFetchOnce({ source_id: "s1", filled: 1, total: 8, is_complete: false });
    const payload = { problem: "the stated problem" };

    await benchmarkApi.save("s1", payload);

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/benchmark/papers/s1`,
      expect.objectContaining({
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    );
  });

  it("throws a generic message on a non-ok response", async () => {
    mockFetchOnce(null, { ok: false, status: 404 });

    await expect(benchmarkApi.detail("missing")).rejects.toThrow("Request failed (404)");
  });
});
