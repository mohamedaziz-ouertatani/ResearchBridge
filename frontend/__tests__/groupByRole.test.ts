import { describe, expect, it } from "vitest";
import { groupByRole } from "@/components/AssessmentReport";
import type { AssessmentEvidence } from "@/lib/assessmentApi";

function evidence(role: AssessmentEvidence["role"], text: string): AssessmentEvidence {
  return { role, paper_id: "p1", paper_title: "Paper", text, section: null };
}

describe("groupByRole", () => {
  it("returns an empty map for no evidence", () => {
    expect(groupByRole([]).size).toBe(0);
  });

  it("groups evidence items under their own role", () => {
    const items = [evidence("novelty", "a"), evidence("risk", "b")];

    const grouped = groupByRole(items);

    expect(grouped.get("novelty")).toEqual([items[0]]);
    expect(grouped.get("risk")).toEqual([items[1]]);
  });

  it("preserves insertion order for multiple items in the same role", () => {
    const items = [evidence("application", "first"), evidence("application", "second")];

    const grouped = groupByRole(items);

    expect(grouped.get("application")?.map((e) => e.text)).toEqual(["first", "second"]);
  });

  it("does not create an entry for a role with no evidence", () => {
    const grouped = groupByRole([evidence("novelty", "a")]);

    expect(grouped.has("feasibility")).toBe(false);
    expect(grouped.get("feasibility")).toBeUndefined();
  });
});
