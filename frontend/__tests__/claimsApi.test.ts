import { afterEach, describe, expect, it, vi } from "vitest";
import { claimsApi } from "@/lib/claimsApi";

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

const BASE = "http://localhost:8000";

describe("claimsApi", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("list() defaults to limit=20 offset=0 with no filters", async () => {
    mockFetchOnce({ items: [], total: 0, limit: 20, offset: 0 });

    await claimsApi.list();

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/claims?limit=20&offset=0`,
      expect.objectContaining({ cache: "no-store" }),
    );
  });

  it("list() includes status/claim_type/source_table filters when not 'all'", async () => {
    mockFetchOnce({ items: [], total: 0, limit: 20, offset: 0 });

    await claimsApi.list({ status: "approved", claim_type: "fact", source_table: "candidate_gaps" });

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/claims?limit=20&offset=0&status=approved&claim_type=fact&source_table=candidate_gaps`,
      expect.anything(),
    );
  });

  it("list() omits filters set to 'all'", async () => {
    mockFetchOnce({ items: [], total: 0, limit: 20, offset: 0 });

    await claimsApi.list({ status: "all", claim_type: "all", source_table: "all" });

    expect(fetch).toHaveBeenCalledWith(`${BASE}/api/claims?limit=20&offset=0`, expect.anything());
  });

  it("list() passes a custom limit and offset through", async () => {
    mockFetchOnce({ items: [], total: 0, limit: 50, offset: 40 });

    await claimsApi.list({}, 50, 40);

    expect(fetch).toHaveBeenCalledWith(`${BASE}/api/claims?limit=50&offset=40`, expect.anything());
  });

  it("throws a generic message on a non-ok response", async () => {
    mockFetchOnce(null, { ok: false, status: 500 });

    await expect(claimsApi.list()).rejects.toThrow("Request failed (500)");
  });
});
