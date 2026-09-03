import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";

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

describe("api", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("stats() requests corpus stats with no-store caching and no params by default", async () => {
    mockFetchOnce({ total_papers: 1 });

    await api.stats();

    const [url, init] = vi.mocked(fetch).mock.calls[0];
    expect(String(url)).toBe(`${BASE}/api/stats`);
    expect(init).toMatchObject({ cache: "no-store" });
  });

  it("stats() includes the year param when given", async () => {
    mockFetchOnce({ total_papers: 1 });

    await api.stats({ year: 2024 });

    const [url] = vi.mocked(fetch).mock.calls[0];
    expect(String(url)).toBe(`${BASE}/api/stats?year=2024`);
  });

  it("trends() requests the given category", async () => {
    mockFetchOnce({ category: "cs.AI", years: [], series: {} });

    await api.trends("cs.AI");

    const [url] = vi.mocked(fetch).mock.calls[0];
    expect(String(url)).toBe(`${BASE}/api/trends?category=cs.AI`);
  });

  it("papers() omits undefined and empty-string params", async () => {
    mockFetchOnce({ items: [], total: 0, limit: 20, offset: 0 });

    await api.papers({ limit: 20, offset: 0, q: "", category: undefined, source: "arxiv" });

    const [url] = vi.mocked(fetch).mock.calls[0];
    const params = new URL(String(url)).searchParams;
    expect(params.get("limit")).toBe("20");
    expect(params.get("offset")).toBe("0");
    expect(params.has("q")).toBe(false);
    expect(params.has("category")).toBe(false);
    expect(params.get("source")).toBe("arxiv");
  });

  it("paper() requests a single paper by id", async () => {
    mockFetchOnce({ id: "p1" });

    await api.paper("p1");

    const [url] = vi.mocked(fetch).mock.calls[0];
    expect(String(url)).toBe(`${BASE}/api/papers/p1`);
  });

  it("similar() defaults top_k to 8", async () => {
    mockFetchOnce([]);

    await api.similar("p1");

    const [url] = vi.mocked(fetch).mock.calls[0];
    expect(String(url)).toBe(`${BASE}/api/papers/p1/similar?top_k=8`);
  });

  it("similar() passes a custom top_k through", async () => {
    mockFetchOnce([]);

    await api.similar("p1", 3);

    const [url] = vi.mocked(fetch).mock.calls[0];
    expect(String(url)).toBe(`${BASE}/api/papers/p1/similar?top_k=3`);
  });

  it("citations() requests the citation graph for a paper", async () => {
    mockFetchOnce({ nodes: [], edges: [] });

    await api.citations("p1");

    const [url] = vi.mocked(fetch).mock.calls[0];
    expect(String(url)).toBe(`${BASE}/api/papers/p1/citations`);
  });

  it("claims() requests extracted claims for a paper", async () => {
    mockFetchOnce([]);

    await api.claims("p1");

    const [url] = vi.mocked(fetch).mock.calls[0];
    expect(String(url)).toBe(`${BASE}/api/papers/p1/claims`);
  });

  it("search() defaults top_k to 12", async () => {
    mockFetchOnce([]);

    await api.search("federated learning");

    const [url] = vi.mocked(fetch).mock.calls[0];
    expect(String(url)).toBe(`${BASE}/api/search?q=federated+learning&top_k=12`);
  });

  it("throws an ApiError carrying the backend's detail message and status on a non-ok response", async () => {
    mockFetchOnce({ detail: "paper not found" }, { ok: false, status: 404 });

    await expect(api.paper("missing")).rejects.toThrow("paper not found");
    try {
      await api.paper("missing");
      expect.unreachable();
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).status).toBe(404);
    }
  });

  it("falls back to a generic message when the error body isn't JSON", async () => {
    const response = {
      ok: false,
      status: 500,
      json: () => Promise.reject(new Error("not json")),
    } as unknown as Response;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));

    await expect(api.paper("p1")).rejects.toThrow("Request failed (500)");
  });
});
