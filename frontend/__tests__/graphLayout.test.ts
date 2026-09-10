import { describe, expect, it } from "vitest";
import { layoutClaimGraph } from "@/lib/graphLayout";
import type { ClaimGraphEdge, ClaimGraphNode } from "@/lib/assessmentApi";

function node(id: string, kind: ClaimGraphNode["kind"]): ClaimGraphNode {
  return {
    id,
    kind,
    label: id,
    claim_type: null,
    confidence: null,
    status: null,
    paper_id: null,
    paper_title: null,
    section: null,
    gap_status: null,
    categories: [],
  };
}

describe("layoutClaimGraph", () => {
  it("positions evidence above the claim it supports, and the claim above the gap it addresses", () => {
    const nodes = [node("claim-1", "claim"), node("evidence-1", "evidence"), node("gap-1", "gap")];
    const edges: ClaimGraphEdge[] = [
      { source: "claim-1", target: "evidence-1", relationship: "supports" },
      { source: "claim-1", target: "gap-1", relationship: "addresses_gap" },
    ];

    const positions = layoutClaimGraph(nodes, edges);
    const byId = Object.fromEntries(positions.map((p) => [p.id, p]));

    expect(byId["evidence-1"].y).toBeLessThan(byId["claim-1"].y);
    expect(byId["claim-1"].y).toBeLessThan(byId["gap-1"].y);
  });

  it("gives every node a finite position even with no edges", () => {
    const nodes = [node("claim-1", "claim")];

    const positions = layoutClaimGraph(nodes, []);

    expect(positions).toHaveLength(1);
    expect(Number.isFinite(positions[0].x)).toBe(true);
    expect(Number.isFinite(positions[0].y)).toBe(true);
  });

  it("returns one position per input node", () => {
    const nodes = [node("a", "claim"), node("b", "evidence"), node("c", "gap")];

    const positions = layoutClaimGraph(nodes, []);

    expect(positions.map((p) => p.id).sort()).toEqual(["a", "b", "c"]);
  });
});
