import { afterEach, describe, expect, it, vi } from "vitest";
import { projectApi } from "@/lib/projectApi";

function mockFetchOnce(
  body: unknown,
  init: { ok?: boolean; status?: number } = {},
) {
  const { ok = true, status = 200 } = init;
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValue({
        ok,
        status,
        json: () => Promise.resolve(body),
      } as Response),
  );
}

const BASE = "http://localhost:8000";

describe("projectApi", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("creates and updates a project with notes and tags", async () => {
    mockFetchOnce({ id: "p1", name: "Graphs", tags: ["graphs"] });
    await projectApi.create("Graphs", "Working notes", ["graphs"]);
    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/projects`,
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          name: "Graphs",
          notes: "Working notes",
          tags: ["graphs"],
        }),
      }),
    );

    mockFetchOnce({ id: "p1", name: "Updated" });
    await projectApi.update("p1", { name: "Updated" });
    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/projects/p1`,
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ name: "Updated" }),
      }),
    );
  });

  it("attaches and removes a stable research input", async () => {
    mockFetchOnce(null, { status: 204 });
    await projectApi.addAssessment("p1", "input-1");
    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/projects/p1/assessments/input-1`,
      expect.objectContaining({ method: "PUT" }),
    );

    mockFetchOnce(null, { status: 204 });
    await projectApi.removeAssessment("p1", "input-1");
    expect(fetch).toHaveBeenCalledWith(
      `${BASE}/api/projects/p1/assessments/input-1`,
      expect.objectContaining({ method: "DELETE" }),
    );
  });
});
