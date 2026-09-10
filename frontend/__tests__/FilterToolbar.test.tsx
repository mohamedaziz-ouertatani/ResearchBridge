import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { FilterToolbar } from "@/components/graph/FilterToolbar";

const allKinds = { claim: true, evidence: true, gap: true };
const allConfidence = { high: true, medium: true, low: true };

describe("FilterToolbar", () => {
  it("renders a checked checkbox for each node kind and confidence level", () => {
    render(
      <FilterToolbar kinds={allKinds} onKindsChange={() => {}} confidence={allConfidence} onConfidenceChange={() => {}} />,
    );

    expect(screen.getByRole("checkbox", { name: "claims" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "evidence" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "gaps" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "high" })).toBeChecked();
  });

  it("calls onKindsChange with only the toggled kind flipped", () => {
    const onKindsChange = vi.fn();
    render(
      <FilterToolbar kinds={allKinds} onKindsChange={onKindsChange} confidence={allConfidence} onConfidenceChange={() => {}} />,
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "gaps" }));

    expect(onKindsChange).toHaveBeenCalledWith({ ...allKinds, gap: false });
  });

  it("calls onConfidenceChange with only the toggled level flipped", () => {
    const onConfidenceChange = vi.fn();
    render(
      <FilterToolbar kinds={allKinds} onKindsChange={() => {}} confidence={allConfidence} onConfidenceChange={onConfidenceChange} />,
    );

    fireEvent.click(screen.getByRole("checkbox", { name: "low" }));

    expect(onConfidenceChange).toHaveBeenCalledWith({ ...allConfidence, low: false });
  });
});
