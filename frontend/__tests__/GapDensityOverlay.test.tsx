import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { GapDensityOverlay } from "@/components/graph/GapDensityOverlay";
import { assessmentApi } from "@/lib/assessmentApi";

describe("GapDensityOverlay", () => {
  afterEach(() => vi.restoreAllMocks());

  it("does not report a highlight before the toggle is enabled", async () => {
    vi.spyOn(assessmentApi, "gapDensity").mockResolvedValue({
      buckets: [{ category: "Machine Learning", gap_count: 10 }],
    });
    const onHighlightChange = vi.fn();

    render(<GapDensityOverlay assessmentId="a1" gapCategories={["Machine Learning"]} onHighlightChange={onHighlightChange} />);

    await waitFor(() => expect(screen.getByRole("button")).not.toBeDisabled());
    expect(onHighlightChange).toHaveBeenLastCalledWith(false);
  });

  it("reports a highlight once enabled, when the gap's category is at or above the average bucket count", async () => {
    vi.spyOn(assessmentApi, "gapDensity").mockResolvedValue({
      buckets: [
        { category: "Machine Learning", gap_count: 10 },
        { category: "Robotics", gap_count: 2 },
      ],
    });
    const onHighlightChange = vi.fn();

    render(<GapDensityOverlay assessmentId="a1" gapCategories={["Machine Learning"]} onHighlightChange={onHighlightChange} />);

    await waitFor(() => expect(screen.getByRole("button")).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button"));

    await waitFor(() => expect(onHighlightChange).toHaveBeenLastCalledWith(true));
  });

  it("does not report a highlight for a below-average category", async () => {
    vi.spyOn(assessmentApi, "gapDensity").mockResolvedValue({
      buckets: [
        { category: "Machine Learning", gap_count: 10 },
        { category: "Robotics", gap_count: 2 },
      ],
    });
    const onHighlightChange = vi.fn();

    render(<GapDensityOverlay assessmentId="a1" gapCategories={["Robotics"]} onHighlightChange={onHighlightChange} />);

    await waitFor(() => expect(screen.getByRole("button")).not.toBeDisabled());
    fireEvent.click(screen.getByRole("button"));

    await waitFor(() => expect(onHighlightChange).toHaveBeenLastCalledWith(false));
  });
});
