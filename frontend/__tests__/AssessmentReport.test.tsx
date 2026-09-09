import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { AssessmentReport } from "@/components/AssessmentReport";
import type { ResearchAssessment } from "@/lib/assessmentApi";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

function baseAssessment(overrides: Partial<ResearchAssessment> = {}): ResearchAssessment {
  return {
    id: "a1",
    research_input: {
      id: "ri1",
      input_type: "idea",
      raw_text: "A research idea about federated fraud detection.",
      title: null,
      matched_paper_id: null,
    },
    status: "completed",
    retrieved_paper_ids: [],
    comparison_summary: null,
    novelty_level: "high",
    novelty_reasoning: null,
    research_gap_text: null,
    research_gap_source: null,
    candidate_gap_id: null,
    potential_applications: null,
    potential_applications_status: "not_assessed",
    potential_opportunities_status: "not_assessed",
    corpus_coverage_status: "in_corpus",
    nearest_distance: 0.21,
    mean_distance: 0.38,
    input_language_caveat: false,
    technical_feasibility_level: "not_assessed",
    technical_feasibility_reasoning: null,
    potential_opportunities: null,
    risks_and_limitations: null,
    recommendation: "worth pursuing",
    confidence: "medium",
    human_reviewed: false,
    evidence: [],
    claims: [],
    ...overrides,
  };
}

// AssessmentHistory (a child of AssessmentReport) always fetches on mount -
// stub it out globally so every test resolves to "no sibling re-runs" and
// renders nothing from that section, which is what most cases below expect.
function stubHistoryFetch() {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve([]),
    } as Response),
  );
}

/** Routes by URL substring: the /history call (see stubHistoryFetch above)
 * and the /opportunities call the Opportunities section triggers need
 * different responses within the same test. */
function stubFetchByUrl(handlers: Record<string, { ok?: boolean; status?: number; body: unknown }>) {
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => {
      const match = Object.entries(handlers).find(([substring]) => url.includes(substring));
      const { ok = true, status = 200, body } = match?.[1] ?? { body: [] };
      return Promise.resolve({ ok, status, json: () => Promise.resolve(body) } as Response);
    }),
  );
}

describe("AssessmentReport", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders the recommendation and confidence from the assessment", async () => {
    stubHistoryFetch();
    render(<AssessmentReport assessment={baseAssessment()} />);

    expect(screen.getByText("worth pursuing")).toBeInTheDocument();
    expect(screen.getByText("medium")).toBeInTheDocument();
    await waitFor(() => expect(fetch).toHaveBeenCalled());
  });

  it("shows an unassessed placeholder for every ungrounded field rather than leaving it blank", async () => {
    stubHistoryFetch();
    render(<AssessmentReport assessment={baseAssessment()} />);

    // comparison_summary, research_gap_text, potential_applications and
    // risks_and_limitations are all null in the fixture above - the report
    // must never render an empty section for a null field
    expect(screen.getByText(/No retrieved paper had extracted claims to compare against/)).toBeInTheDocument();
    expect(screen.getByText(/No gap was found in the retrieved literature/)).toBeInTheDocument();
    expect(screen.getByText(/No retrieved paper stated a limitation/)).toBeInTheDocument();
  });

  it("renders a matched-paper link only when the input has a matched_paper_id", async () => {
    stubHistoryFetch();
    const { rerender } = render(<AssessmentReport assessment={baseAssessment()} />);
    expect(screen.queryByText("view it")).not.toBeInTheDocument();

    rerender(
      <AssessmentReport
        assessment={baseAssessment({
          research_input: {
            id: "ri1",
            input_type: "document",
            raw_text: "text",
            title: null,
            matched_paper_id: "paper-123",
          },
        })}
      />,
    );
    expect(screen.getByText("view it")).toBeInTheDocument();
  });

  it("lists potential applications with a link back to their source paper", async () => {
    stubHistoryFetch();
    render(
      <AssessmentReport
        assessment={baseAssessment({
          potential_applications: [
            { application: "real-time fraud screening", source_paper: "Fraud Paper", paper_id: "p1" },
          ],
        })}
      />,
    );

    expect(screen.getByText("real-time fraud screening")).toBeInTheDocument();
    expect(screen.getByText("Fraud Paper")).toBeInTheDocument();
  });

  it("shows no synthesize button when there are no potential applications to ground it in", async () => {
    stubHistoryFetch();
    render(<AssessmentReport assessment={baseAssessment({ potential_applications: null })} />);

    expect(screen.queryByRole("button", { name: /synthesize opportunities/ })).not.toBeInTheDocument();
    // wording changed 2026-09-09: the message now names extraction coverage
    // as the limiting factor rather than implying the idea was at fault
    expect(
      screen.getByText(/this stage runs only when the potential-applications field/),
    ).toBeInTheDocument();
  });

  it("synthesizes opportunities on click and renders all three tiers with source links", async () => {
    const applications = [{ application: "fraud screening", source_paper: "Fraud Paper", paper_id: "p1" }];
    stubFetchByUrl({
      "/history": { body: [] },
      "/opportunities": {
        body: {
          ...baseAssessment({ potential_applications: applications }),
          potential_opportunities: [
            {
              tier: "direct",
              opportunity: "fraud-scoring API",
              source_applications: [{ application: "fraud screening", paper_id: "p1", paper_title: "Fraud Paper" }],
            },
            {
              tier: "adjacent",
              opportunity: "risk platform",
              source_applications: [{ application: "fraud screening", paper_id: "p1", paper_title: "Fraud Paper" }],
            },
            {
              tier: "speculative",
              opportunity: "fraud network",
              source_applications: [{ application: "fraud screening", paper_id: "p1", paper_title: "Fraud Paper" }],
            },
          ],
        },
      },
    });

    render(<AssessmentReport assessment={baseAssessment({ potential_applications: applications })} />);

    fireEvent.click(screen.getByText(/synthesize opportunities/));

    await waitFor(() => expect(screen.getByText("fraud-scoring API")).toBeInTheDocument());
    expect(screen.getByText("risk platform")).toBeInTheDocument();
    expect(screen.getByText("fraud network")).toBeInTheDocument();
    expect(screen.getAllByText("Fraud Paper").length).toBeGreaterThan(0);
  });

  it("shows an inline error when synthesis fails, without touching the rest of the report", async () => {
    const applications = [{ application: "fraud screening", source_paper: "Fraud Paper", paper_id: "p1" }];
    stubFetchByUrl({
      "/history": { body: [] },
      "/opportunities": { ok: false, status: 503, body: { detail: "local LLM opportunity synthesis is not enabled" } },
    });

    render(<AssessmentReport assessment={baseAssessment({ potential_applications: applications })} />);

    fireEvent.click(screen.getByText(/synthesize opportunities/));

    await waitFor(() => expect(screen.getByText(/local LLM unavailable/)).toBeInTheDocument());
    expect(screen.getByText("worth pursuing")).toBeInTheDocument();
  });

  it("renders already-synthesized opportunities without showing a button", async () => {
    stubHistoryFetch();
    render(
      <AssessmentReport
        assessment={baseAssessment({
          potential_applications: [{ application: "fraud screening", source_paper: "Fraud Paper", paper_id: "p1" }],
          potential_opportunities: [
            {
              tier: "direct",
              opportunity: "fraud-scoring API",
              source_applications: [{ application: "fraud screening", paper_id: "p1", paper_title: "Fraud Paper" }],
            },
            {
              tier: "adjacent",
              opportunity: "risk platform",
              source_applications: [{ application: "fraud screening", paper_id: "p1", paper_title: "Fraud Paper" }],
            },
            {
              tier: "speculative",
              opportunity: "fraud network",
              source_applications: [{ application: "fraud screening", paper_id: "p1", paper_title: "Fraud Paper" }],
            },
          ],
        })}
      />,
    );

    expect(screen.getByText("fraud-scoring API")).toBeInTheDocument();
    expect(screen.queryByText(/synthesize opportunities/)).not.toBeInTheDocument();
    expect(screen.getByText(/AI-synthesized from the applications above/)).toBeInTheDocument();
  });
});


describe("AssessmentReport reliability banners", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows the out-of-corpus banner when the corpus has nothing relevant to say", () => {
    // this banner reached the exported file but never the web report, so the
    // same assessment told two different stories depending on where it was read
    stubHistoryFetch();
    render(
      <AssessmentReport
        assessment={baseAssessment({ corpus_coverage_status: "out_of_corpus" })}
      />,
    );

    expect(
      screen.getByText(/falls outside the corpus this system searches/),
    ).toBeInTheDocument();
  });

  it("shows no out-of-corpus banner for an in-corpus idea", () => {
    stubHistoryFetch();
    render(
      <AssessmentReport
        assessment={baseAssessment({ corpus_coverage_status: "in_corpus" })}
      />,
    );

    expect(
      screen.queryByText(/falls outside the corpus this system searches/),
    ).not.toBeInTheDocument();
  });

  it("shows the non-English banner above the report rather than buried in novelty prose", () => {
    stubHistoryFetch();
    render(
      <AssessmentReport assessment={baseAssessment({ input_language_caveat: true })} />,
    );

    expect(
      screen.getByText(/does not appear to be written in English/),
    ).toBeInTheDocument();
  });

  it("shows both banners together when both apply", () => {
    stubHistoryFetch();
    render(
      <AssessmentReport
        assessment={baseAssessment({
          corpus_coverage_status: "out_of_corpus",
          input_language_caveat: true,
        })}
      />,
    );

    expect(screen.getByText(/falls outside the corpus/)).toBeInTheDocument();
    expect(
      screen.getByText(/does not appear to be written in English/),
    ).toBeInTheDocument();
  });

  it("shows the retrieval distances the corpus-coverage verdict was decided from", () => {
    stubHistoryFetch();
    render(
      <AssessmentReport
        assessment={baseAssessment({ nearest_distance: 0.0068, mean_distance: 0.4123 })}
      />,
    );

    expect(screen.getByText("0.007 / 0.412")).toBeInTheDocument();
  });

  it("renders a dash for an assessment predating persisted distances", () => {
    stubHistoryFetch();
    render(
      <AssessmentReport
        assessment={baseAssessment({ nearest_distance: null, mean_distance: null })}
      />,
    );

    expect(screen.getByText("nearest / mean distance")).toBeInTheDocument();
  });

  it("labels the limitations field as related work, not as risks of the idea", () => {
    // every line is a limitation a RETRIEVED PAPER stated about its own work
    stubHistoryFetch();
    render(<AssessmentReport assessment={baseAssessment()} />);

    expect(screen.getAllByText("limitations in related work").length).toBeGreaterThan(0);
    expect(screen.queryByText("risks / limitations")).not.toBeInTheDocument();
  });

  it("marks a generic future-work gap as inferred rather than explicitly stated", () => {
    stubHistoryFetch();
    render(
      <AssessmentReport
        assessment={baseAssessment({
          research_gap_text:
            "No explicit gap stated; nearest related paper mentions future work generically: \"more work is needed\" (from \"A Paper\")",
          research_gap_source: "input_specific",
        })}
      />,
    );

    expect(
      screen.getByText("inferred from generic future-work wording"),
    ).toBeInTheDocument();
  });

  it("marks an author-stated gap as explicitly stated", () => {
    stubHistoryFetch();
    render(
      <AssessmentReport
        assessment={baseAssessment({
          research_gap_text:
            'Explicitly stated in "A Paper": "no prior work evaluates this under distribution shift"',
          research_gap_source: "input_specific",
        })}
      />,
    );

    expect(screen.getByText("explicitly stated")).toBeInTheDocument();
  });

  it("explains the evidence gutter so the count is readable without narration", () => {
    stubHistoryFetch();
    render(<AssessmentReport assessment={baseAssessment()} />);

    expect(
      screen.getByText(/count of the real quoted passages backing it/),
    ).toBeInTheDocument();
  });
});
