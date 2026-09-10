import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { EvidenceNode, type EvidenceFlowNode } from "@/components/graph/EvidenceNode";

function props(data: EvidenceFlowNode["data"]): NodeProps<EvidenceFlowNode> {
  return { id: "evidence-1", type: "evidence", data, selected: false, dragging: false, isConnectable: true, zIndex: 0 } as unknown as NodeProps<EvidenceFlowNode>;
}

describe("EvidenceNode", () => {
  it("renders the quoted passage, paper title, and section", () => {
    render(
      <ReactFlowProvider>
        <EvidenceNode {...props({ label: "a quoted passage", paper_title: "Some Paper", section: "results" })} />
      </ReactFlowProvider>,
    );

    expect(screen.getByText('"a quoted passage"')).toBeInTheDocument();
    expect(screen.getByText("Some Paper")).toBeInTheDocument();
    expect(screen.getByText("evidence · results")).toBeInTheDocument();
  });

  it("omits the paper title line when paper_title is null", () => {
    render(
      <ReactFlowProvider>
        <EvidenceNode {...props({ label: "a quoted passage", paper_title: null, section: null })} />
      </ReactFlowProvider>,
    );

    expect(screen.getByText("evidence")).toBeInTheDocument();
  });
});
