import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { AnnotationQueue } from "@/components/AnnotationQueue";
import type { AnnotationSummary } from "@/lib/benchmarkApi";

function summary(overrides: Partial<AnnotationSummary> = {}): AnnotationSummary {
  return {
    source_id: "s1",
    title: "A Paper Title",
    domain: "cs.LG",
    year: 2024,
    filled: 3,
    total: 8,
    is_complete: false,
    has_fulltext: true,
    has_fulltext_nougat: false,
    ...overrides,
  };
}

describe("AnnotationQueue", () => {
  it("links each paper to its annotation page", () => {
    render(<AnnotationQueue papers={[summary({ source_id: "s1" }), summary({ source_id: "s2" })]} activeId="s1" />);

    const links = screen.getAllByRole("link");
    expect(links.map((l) => l.getAttribute("href"))).toEqual(["/annotate/s1", "/annotate/s2"]);
  });

  it("marks the active paper's link with aria-current", () => {
    render(<AnnotationQueue papers={[summary({ source_id: "s1" }), summary({ source_id: "s2" })]} activeId="s2" />);

    expect(screen.getByRole("link", { current: "page" })).toHaveAttribute("href", "/annotate/s2");
  });

  it("shows the fill count for an incomplete paper and 'done' for a complete one", () => {
    render(
      <AnnotationQueue
        papers={[
          summary({ source_id: "s1", filled: 3, total: 8, is_complete: false }),
          summary({ source_id: "s2", is_complete: true }),
        ]}
        activeId="s1"
      />,
    );

    expect(screen.getByText("3/8")).toBeInTheDocument();
    expect(screen.getByText("done")).toBeInTheDocument();
  });

  it("falls back to the source id when the paper has no title", () => {
    render(<AnnotationQueue papers={[summary({ source_id: "s1", title: null })]} activeId="s1" />);

    expect(screen.getAllByText("s1")).toHaveLength(2); // eyebrow id + title fallback
  });

  it("flags a paper with no full text", () => {
    render(<AnnotationQueue papers={[summary({ has_fulltext: false })]} activeId="s1" />);

    expect(screen.getByText("no full text")).toBeInTheDocument();
  });

  it("does not flag a paper that has full text", () => {
    render(<AnnotationQueue papers={[summary({ has_fulltext: true })]} activeId="s1" />);

    expect(screen.queryByText("no full text")).not.toBeInTheDocument();
  });
});
