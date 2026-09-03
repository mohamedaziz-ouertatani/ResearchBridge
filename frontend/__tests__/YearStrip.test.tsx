import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { YearStrip } from "@/components/YearStrip";

describe("YearStrip", () => {
  it("renders nothing when there are no years", () => {
    const { container } = render(<YearStrip byYear={{}} activeYear={null} onSelectYear={vi.fn()} />);

    expect(container).toBeEmptyDOMElement();
  });

  it("renders one button per year, sorted ascending, with the count in its accessible label", () => {
    render(
      <YearStrip byYear={{ "2023": 5, "2021": 2, "2022": 8 }} activeYear={null} onSelectYear={vi.fn()} />,
    );

    const buttons = screen.getAllByRole("button");
    expect(buttons.map((b) => b.getAttribute("aria-label"))).toEqual([
      "2 papers from 2021",
      "8 papers from 2022",
      "5 papers from 2023",
    ]);
  });

  it("marks the active year as pressed and notes it in the label", () => {
    render(<YearStrip byYear={{ "2023": 5 }} activeYear={2023} onSelectYear={vi.fn()} />);

    const button = screen.getByRole("button");
    expect(button).toHaveAttribute("aria-pressed", "true");
    expect(button).toHaveAttribute("aria-label", "5 papers from 2023 (filter active)");
  });

  it("selecting an inactive year calls onSelectYear with that year", () => {
    const onSelectYear = vi.fn();
    render(<YearStrip byYear={{ "2023": 5, "2024": 3 }} activeYear={null} onSelectYear={onSelectYear} />);

    fireEvent.click(screen.getByRole("button", { name: "3 papers from 2024" }));

    expect(onSelectYear).toHaveBeenCalledWith(2024);
  });

  it("clicking the already-active year clears the filter (calls onSelectYear with null)", () => {
    const onSelectYear = vi.fn();
    render(<YearStrip byYear={{ "2023": 5 }} activeYear={2023} onSelectYear={onSelectYear} />);

    fireEvent.click(screen.getByRole("button"));

    expect(onSelectYear).toHaveBeenCalledWith(null);
  });
});
