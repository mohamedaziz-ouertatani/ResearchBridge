import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import type { EdgeProps } from "@xyflow/react";
import { RelationshipEdge } from "@/components/graph/RelationshipEdge";

function props(relationship: string): EdgeProps {
  return {
    id: "e1", source: "a", target: "b", sourceX: 0, sourceY: 0, targetX: 100, targetY: 100,
    sourcePosition: "bottom", targetPosition: "top", data: { relationship },
  } as unknown as EdgeProps;
}

function renderEdge(relationship: string) {
  return render(
    <ReactFlowProvider>
      <svg>
        <RelationshipEdge {...props(relationship)} />
      </svg>
    </ReactFlowProvider>,
  );
}

describe("RelationshipEdge", () => {
  it("renders a solid line for a supports relationship", () => {
    const { container } = renderEdge("supports");

    const path = container.querySelector("path") as SVGPathElement;
    expect(path.style.strokeDasharray).toBe("");
  });

  it("renders a dashed line for a contradicts relationship", () => {
    const { container } = renderEdge("contradicts");

    const path = container.querySelector("path") as SVGPathElement;
    expect(path.style.strokeDasharray).toBe("6 4");
  });

  it("renders a dotted line for a contextualizes relationship", () => {
    const { container } = renderEdge("contextualizes");

    const path = container.querySelector("path") as SVGPathElement;
    expect(path.style.strokeDasharray).toBe("2 4");
  });

  it("renders a distinct dash pattern for an addresses_gap relationship", () => {
    const { container } = renderEdge("addresses_gap");

    const path = container.querySelector("path") as SVGPathElement;
    expect(path.style.strokeDasharray).toBe("10 4");
  });
});
