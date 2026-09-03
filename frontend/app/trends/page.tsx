"use client";

import { useEffect, useState } from "react";
import { api, type CorpusStats, type TrendsData } from "@/lib/api";
import { Nav } from "@/components/Nav";

// Same SVG line-chart shape as AssessmentReport.tsx's IdeaYearLineChart
// (per-assessment "supporting passages by year" chart) - one series per
// claim_type here instead of one series of passage counts, same
// zero-filled-year-axis and hover-tooltip conventions. Unlike that fixed-
// width chart, this one's pixel width scales with year count * zoom and
// scrolls horizontally once it exceeds the container - a category can span
// many more years than one assessment's retrieved papers ever would.
const CHART_HEIGHT = 260;
const CHART_PAD_X = 16;
const CHART_PAD_TOP = 16;
const CHART_PAD_BOTTOM = 28;
const CHART_MIN_WIDTH = 640;
const BASE_PX_PER_YEAR = 48;
const ZOOM_MIN = 0.5;
const ZOOM_MAX = 4;
const ZOOM_STEP = 1.25;

// Fixed order matching the blueprint's documented claim_type list, not
// whatever order the API returns keys in - keeps the page from reflowing
// as you switch categories.
const CLAIM_TYPE_ORDER = [
  "problem",
  "method",
  "dataset",
  "metric",
  "result",
  "limitation",
  "research_gap",
  "application",
  "contribution",
];

const SELECT_CLASS =
  "eyebrow rounded-[2px] border border-[var(--rule)] bg-transparent px-2 py-1 text-[0.6875rem] text-[var(--ink-soft)] hover:border-[var(--ink)] hover:text-[var(--ink)] focus:border-[var(--ink)] focus:outline-none";

export default function TrendsPage() {
  const [categories, setCategories] = useState<string[]>([]);
  const [category, setCategory] = useState<string>("");
  const [data, setData] = useState<TrendsData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    api
      .stats()
      .then((stats: CorpusStats) => {
        const sorted = Object.keys(stats.papers_by_category).sort(
          (a, b) => stats.papers_by_category[b] - stats.papers_by_category[a],
        );
        setCategories(sorted);
        if (sorted.length > 0) setCategory(sorted[0]);
      })
      .catch(() => setError("Couldn't load categories."));
  }, []);

  useEffect(() => {
    if (!category) return;
    // Resetting data/error before a fetch keyed on `category` is intentional -
    // not the accidental-derived-state case this rule otherwise targets.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setData(null);
    setError(null);
    api
      .trends(category)
      .then(setData)
      .catch(() => setError("Couldn't load trends."));
  }, [category]);

  return (
    <main className="mx-auto max-w-[80rem] px-6 pb-24 sm:px-8">
      <Nav />

      <div className="pt-12">
        <span className="eyebrow">trends</span>
        <p className="mt-3 max-w-[60ch] text-[0.9375rem] leading-relaxed text-[var(--ink-soft)]">
          How many extracted claims of each type appear per year, within one
          category - a real count of what&apos;s already in the corpus, not an
          inferred pattern.
        </p>

        <div className="mt-6 flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2">
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
              category
            </span>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className={SELECT_CLASS}
            >
              {categories.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </label>

          <div className="flex items-center gap-1">
            <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
              zoom
            </span>
            <button
              type="button"
              onClick={() => setZoom((z) => Math.max(ZOOM_MIN, z / ZOOM_STEP))}
              disabled={zoom <= ZOOM_MIN}
              aria-label="Zoom out"
              className="eyebrow rounded-[2px] border border-[var(--rule)] px-2 py-1 text-[0.6875rem] hover:border-[var(--ink)] hover:text-[var(--ink)] disabled:opacity-40"
            >
              −
            </button>
            <span className="readout w-10 text-center text-[0.6875rem] tabular-nums text-[var(--ink-faint)]">
              {Math.round(zoom * 100)}%
            </span>
            <button
              type="button"
              onClick={() => setZoom((z) => Math.min(ZOOM_MAX, z * ZOOM_STEP))}
              disabled={zoom >= ZOOM_MAX}
              aria-label="Zoom in"
              className="eyebrow rounded-[2px] border border-[var(--rule)] px-2 py-1 text-[0.6875rem] hover:border-[var(--ink)] hover:text-[var(--ink)] disabled:opacity-40"
            >
              +
            </button>
          </div>
        </div>

        {error && (
          <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">
            {error}
          </p>
        )}

        {!error && !data && <p className="eyebrow py-16">loading…</p>}

        {!error && data && data.years.length === 0 && (
          <p className="py-16 text-[0.9375rem] text-[var(--ink-soft)]">
            No extracted claims yet for this category.
          </p>
        )}

        {!error && data && data.years.length > 0 && (
          <div className="mt-8 space-y-10">
            {CLAIM_TYPE_ORDER.filter((claimType) => data.series[claimType]).map(
              (claimType) => (
                <TrendLineChart
                  key={claimType}
                  label={claimType.replace("_", " ")}
                  years={data.years}
                  counts={data.series[claimType]}
                  zoom={zoom}
                />
              ),
            )}
          </div>
        )}
      </div>
    </main>
  );
}

function TrendLineChart({
  label,
  years,
  counts,
  zoom,
}: {
  label: string;
  years: number[];
  counts: number[];
  zoom: number;
}) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const peak = Math.max(...counts, 1);

  // Wider than the container once there are enough years * zoom - the
  // wrapping div below scrolls horizontally rather than squeezing points
  // together, so a category spanning many years stays readable at any zoom.
  const chartWidth = Math.max(
    CHART_MIN_WIDTH,
    Math.round(years.length * BASE_PX_PER_YEAR * zoom),
  );

  const plotWidth = chartWidth - CHART_PAD_X * 2;
  const plotHeight = CHART_HEIGHT - CHART_PAD_TOP - CHART_PAD_BOTTOM;

  const xFor = (i: number) =>
    CHART_PAD_X +
    (years.length === 1 ? plotWidth / 2 : (i / (years.length - 1)) * plotWidth);
  const yFor = (count: number) =>
    CHART_PAD_TOP + plotHeight - (count / peak) * plotHeight;

  const linePath = counts
    .map((count, i) => `${i === 0 ? "M" : "L"} ${xFor(i)} ${yFor(count)}`)
    .join(" ");
  const baselineY = CHART_PAD_TOP + plotHeight;

  return (
    <div>
      <span className="eyebrow text-[0.625rem] text-[var(--ink-faint)]">
        {label}
      </span>
      <div className="no-scrollbar mt-2 overflow-x-auto pt-14 px-14">
        <div className="relative" style={{ width: chartWidth }}>
          <svg
            width={chartWidth}
            height={CHART_HEIGHT}
            viewBox={`0 0 ${chartWidth} ${CHART_HEIGHT}`}
            role="img"
            aria-label={`${label} claim count by year, ${years[0]} to ${years[years.length - 1]}`}
            onMouseLeave={() => setHoverIndex(null)}
          >
            <line
              x1={CHART_PAD_X}
              y1={baselineY}
              x2={chartWidth - CHART_PAD_X}
              y2={baselineY}
              stroke="var(--rule-soft)"
              strokeWidth={1}
            />

            <path
              d={linePath}
              fill="none"
              stroke="var(--ink)"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {years.map((year, i) => (
              <g key={year}>
                <rect
                  x={xFor(i) - plotWidth / years.length / 2}
                  y={0}
                  width={plotWidth / years.length}
                  height={CHART_HEIGHT}
                  fill="transparent"
                  onMouseEnter={() => setHoverIndex(i)}
                />
                <circle
                  cx={xFor(i)}
                  cy={yFor(counts[i])}
                  r={hoverIndex === i ? 5 : 3}
                  fill={counts[i] > 0 ? "var(--ink)" : "var(--rule)"}
                />
                <text
                  x={xFor(i)}
                  y={CHART_HEIGHT - 6}
                  textAnchor="middle"
                  className="readout"
                  style={{
                    fontSize: "9px",
                    fill: hoverIndex === i ? "var(--near)" : "var(--ink-faint)",
                  }}
                >
                  {year}
                </text>
              </g>
            ))}

            {hoverIndex !== null && (
              <line
                x1={xFor(hoverIndex)}
                y1={CHART_PAD_TOP}
                x2={xFor(hoverIndex)}
                y2={baselineY}
                stroke="var(--rule)"
                strokeWidth={1}
                strokeDasharray="2 2"
              />
            )}
          </svg>

          {hoverIndex !== null && (
            <div
              className="pointer-events-none absolute -translate-x-1/2 -translate-y-full rounded-[2px] border border-[var(--rule)] bg-[var(--panel)] px-2 py-1 shadow-sm"
              style={{ left: xFor(hoverIndex), top: yFor(counts[hoverIndex]) }}
            >
              <p className="readout text-[0.6875rem] text-[var(--ink)] tabular-nums">
                {years[hoverIndex]} · {counts[hoverIndex].toLocaleString()}{" "}
                claim{counts[hoverIndex] === 1 ? "" : "s"}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
