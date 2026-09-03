import { afterEach, describe, expect, it, vi } from "vitest";
import { gapsApi } from "@/lib/gapsApi";

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

describe("gapsApi", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("list() defaults to status=pending with a limit of 50", async () => {
    mockFetchOnce({ items: [], total: 0, limit: 50, offset: 0 });

    await gapsApi.list();

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/gaps?status=pending&limit=50`,
      expect.objectContaining({ cache: "no-store" }),
    );
  });

  it("list() passes a custom status filter through", async () => {
    mockFetchOnce({ items: [], total: 0, limit: 50, offset: 0 });

    await gapsApi.list("approved");

    expect(fetch).toHaveBeenCalledWith(`${BASE}/api/gaps?status=approved&limit=50`, expect.anything());
  });

  it("review() PUTs status, review_note, and any ratings", async () => {
    mockFetchOnce({ id: "g1", status: "approved" });

    await gapsApi.review("g1", "approved", "looks solid", { correctness_rating: 4 });

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/gaps/g1`,
      expect.objectContaining({
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: "approved", review_note: "looks solid", correctness_rating: 4 }),
      }),
    );
  });

  it("review() sends null review_note when omitted", async () => {
    mockFetchOnce({ id: "g1", status: "rejected" });

    await gapsApi.review("g1", "rejected");

    const [, init] = vi.mocked(fetch).mock.calls[0];
    expect(JSON.parse(init?.body as string)).toEqual({ status: "rejected", review_note: null });
  });

  it("detect() POSTs to the detect endpoint", async () => {
    mockFetchOnce({ started: true, pipeline: "gaps", log_file: "x.log" });

    await gapsApi.detect();

    expect(fetch).toHaveBeenCalledWith(`${BASE}/api/gaps/detect`, expect.objectContaining({ method: "POST" }));
  });

  it("detectStatus() requests the detect run's status", async () => {
    mockFetchOnce({ running: false, log: "" });

    await gapsApi.detectStatus();

    expect(fetch).toHaveBeenCalledWith(`${BASE}/api/gaps/detect/status`, expect.objectContaining({ cache: "no-store" }));
  });

  it("throws a generic message on a non-ok response", async () => {
    mockFetchOnce(null, { ok: false, status: 404 });

    await expect(gapsApi.list()).rejects.toThrow("Request failed (404)");
  });
});
