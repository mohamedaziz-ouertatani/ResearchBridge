import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { PaperRow } from "@/components/PaperRow";
import type { PaperSummary } from "@/lib/api";

function paper(overrides: Partial<PaperSummary> = {}): PaperSummary {
  return {
    id: "p1",
    source: "arxiv",
    source_id: "2401.00001",
    title: "A Study of Federated Fraud Detection",
    abstract: "An abstract.",
    publication_date: "2024-03-15",
    url: "https://arxiv.org/abs/2401.00001",
    primary_category: "cs.LG",
    categories: ["cs.LG"],
    authors: ["Alice Author", "Bob Builder"],
    excluded_at: null,
    ...overrides,
  };
}

describe("PaperRow", () => {
  it("links to the paper's detail page", () => {
    render(
      <ul>
        <PaperRow paper={paper()} />
      </ul>,
    );

    expect(screen.getByRole("link")).toHaveAttribute("href", "/papers/p1");
  });

  it("shows the title, source id, category, and publication year", () => {
    render(
      <ul>
        <PaperRow paper={paper()} />
      </ul>,
    );

    expect(screen.getByText("A Study of Federated Fraud Detection")).toBeInTheDocument();
    expect(screen.getByText("2401.00001")).toBeInTheDocument();
    expect(screen.getByText("cs.LG")).toBeInTheDocument();
    expect(screen.getByText("2024")).toBeInTheDocument();
  });

  it("shows an em dash for a missing publication date", () => {
    render(
      <ul>
        <PaperRow paper={paper({ publication_date: null })} />
      </ul>,
    );

    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("lists up to three authors as-is", () => {
    render(
      <ul>
        <PaperRow paper={paper({ authors: ["Alice", "Bob", "Carol"] })} />
      </ul>,
    );

    expect(screen.getByText("Alice, Bob, Carol")).toBeInTheDocument();
  });

  it("truncates more than three authors with a +N suffix", () => {
    render(
      <ul>
        <PaperRow paper={paper({ authors: ["Alice", "Bob", "Carol", "Dave", "Eve"] })} />
      </ul>,
    );

    expect(screen.getByText("Alice, Bob, Carol +2")).toBeInTheDocument();
  });

  it("shows 'Unattributed' when there are no authors", () => {
    render(
      <ul>
        <PaperRow paper={paper({ authors: [] })} />
      </ul>,
    );

    expect(screen.getByText("Unattributed")).toBeInTheDocument();
  });

  it("shows an 'excluded' badge only when the paper is excluded", () => {
    const { rerender } = render(
      <ul>
        <PaperRow paper={paper()} />
      </ul>,
    );
    expect(screen.queryByText("excluded")).not.toBeInTheDocument();

    rerender(
      <ul>
        <PaperRow paper={paper({ excluded_at: "2026-01-01T00:00:00Z" })} />
      </ul>,
    );
    expect(screen.getByText("excluded")).toBeInTheDocument();
  });

  it("renders the proximity gauge only when a distance is passed", () => {
    const { rerender } = render(
      <ul>
        <PaperRow paper={paper()} />
      </ul>,
    );
    expect(screen.queryByRole("img")).not.toBeInTheDocument();

    rerender(
      <ul>
        <PaperRow paper={paper()} distance={0.42} />
      </ul>,
    );
    expect(screen.getByText("0.420")).toBeInTheDocument();
  });
});
