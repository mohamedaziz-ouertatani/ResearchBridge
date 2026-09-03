import { afterEach, describe, expect, it, vi } from "vitest";
import { assessmentApi } from "@/lib/assessmentApi";

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

describe("assessmentApi", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("get() requests the assessment by id with no-store caching", async () => {
    mockFetchOnce({ id: "a1" });

    await assessmentApi.get("a1");

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/assessments/a1`,
      expect.objectContaining({ cache: "no-store" }),
    );
  });

  it("create() posts raw_text as JSON", async () => {
    mockFetchOnce({ id: "a1" });

    await assessmentApi.create("a new research idea");

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/assessments`,
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ raw_text: "a new research idea" }),
      }),
    );
  });

  it("upload() sends a multipart FormData body without a manual Content-Type header", async () => {
    mockFetchOnce({ id: "a1" });
    const file = new File(["contents"], "paper.pdf", { type: "application/pdf" });

    await assessmentApi.upload(file);

    const [url, init] = vi.mocked(fetch).mock.calls[0];
    expect(url).toBe(`${BASE}/api/assessments/upload`);
    expect(init?.method).toBe("POST");
    expect(init?.headers).toBeUndefined();
    expect(init?.body).toBeInstanceOf(FormData);
    expect((init?.body as FormData).get("file")).toBe(file);
  });

  it("review() PUTs the human_reviewed flag", async () => {
    mockFetchOnce({ id: "a1", human_reviewed: true });

    await assessmentApi.review("a1", true);

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/assessments/a1/review`,
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({ human_reviewed: true }),
      }),
    );
  });

  it("rerun() POSTs with no body", async () => {
    mockFetchOnce({ id: "a2" });

    await assessmentApi.rerun("a1");

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/assessments/a1/rerun`,
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("remove() DELETEs and resolves without parsing a body", async () => {
    mockFetchOnce(null);

    await expect(assessmentApi.remove("a1")).resolves.toBeUndefined();

    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/assessments/a1`,
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("list() defaults to review=all and limit=50 with no extra params", async () => {
    mockFetchOnce({ items: [], total: 0, limit: 50, offset: 0 });

    await assessmentApi.list();

    const [url] = vi.mocked(fetch).mock.calls[0];
    expect(url).toBe(`${BASE}/api/assessments?review=all&limit=50`);
  });

  it("list() includes sort/novelty/feasibility only when provided", async () => {
    mockFetchOnce({ items: [], total: 0, limit: 50, offset: 0 });

    await assessmentApi.list("needs_review", { sort: "priority", novelty: "high" });

    const [url] = vi.mocked(fetch).mock.calls[0];
    expect(url).toBe(
      `${BASE}/api/assessments?review=needs_review&limit=50&sort=priority&novelty=high`,
    );
  });

  it("throws the backend's detail message on a non-ok response", async () => {
    mockFetchOnce({ detail: "assessment not found" }, { ok: false, status: 404 });

    await expect(assessmentApi.get("missing")).rejects.toThrow("assessment not found");
  });

  it("falls back to a generic message when the error body isn't JSON", async () => {
    const response = {
      ok: false,
      status: 500,
      json: () => Promise.reject(new Error("not json")),
    } as unknown as Response;
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));

    await expect(assessmentApi.get("a1")).rejects.toThrow("Request failed (500)");
  });
});
