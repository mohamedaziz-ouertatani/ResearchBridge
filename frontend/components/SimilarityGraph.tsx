"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import type { ComponentType } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { assessmentApi, type GraphData, type GraphNode } from "@/lib/assessmentApi";
import { loadForceGraph2D, type ForceGraphInstance } from "@/lib/forceGraphLoader";

type GraphLink = { source: string; target: string; distance: number };

const ForceGraph2D = dynamic(loadForceGraph2D, {
  ssr: false,
}) as unknown as ComponentType<{
  graphData: { nodes: GraphNode[]; links: GraphLink[] };
  nodeId: string;
  nodeLabel: (node: GraphNode) => string | HTMLElement;
  nodeColor: (node: GraphNode) => string;
  nodeVal: (node: GraphNode) => number;
  linkColor: (link: GraphLink) => string;
  linkWidth: (link: GraphLink) => number;
  onNodeClick: (node: GraphNode) => void;
  onBackgroundClick: () => void;
  width: number;
  height: number;
  ref?: (instance: ForceGraphInstance | null) => void;
}>;

// Wider spacing than the library's d3-force defaults (charge strength -30,
// link distance ~30) so nodes/edges don't render as an unreadably tight
// cluster once there are more than a couple of retrieved papers. A callback
// ref (not a plain ref + effect) because it needs to fire exactly once,
// right when the underlying force-graph instance mounts - same reasoning as
// measuredContainerRef below.
const CHARGE_STRENGTH = -200;

// Link length encodes the edge's real cosine distance, not a uniform
// constant - a paper barely related to the input should visibly sit farther
// out than a near-duplicate. MIN_LINK_DISTANCE keeps distance-0 edges from
// collapsing nodes on top of each other; the scale spreads the practical
// 0-1 distance range out to a readable radius.
const MIN_LINK_DISTANCE = 40;
const LINK_DISTANCE_SCALE = 300;

function configureForces(instance: ForceGraphInstance | null) {
  if (!instance) return;
  instance.d3Force("charge")?.strength?.(CHARGE_STRENGTH);
  instance.d3Force("link")?.distance?.((link) => MIN_LINK_DISTANCE + link.distance * LINK_DISTANCE_SCALE);
}

/** Same "teal near, washed-out far" convention as ProximityGauge - this ramp
 * is reserved for measured cosine distance and nothing else (see globals.css).
 *
 * react-force-graph hands this string straight to a <canvas> 2D context's
 * fillStyle/strokeStyle. CanvasRenderingContext2D parses CSS <color> values
 * with no element and no cascade, so `var(--near)`/`var(--far)` can never
 * resolve there - the raw color-mix(...) string would be silently dropped,
 * leaving the canvas default (black). Resolve the custom properties to a
 * concrete color via a real DOM element (which does have a cascade) before
 * handing anything to the canvas API, and cache by rounded percentage so a
 * force-simulation tick isn't creating/removing DOM nodes per node/link.
 */
let colorResolutionEl: HTMLDivElement | null = null;
const resolvedColorCache = new Map<number, string>();

function resolveColorMix(roundedPosition: number): string {
  if (typeof document === "undefined") {
    // SSR guard: this component is already dynamic()-imported with
    // ssr:false, so this path shouldn't run, but cost nothing to guard.
    return "#000000";
  }
  const cached = resolvedColorCache.get(roundedPosition);
  if (cached) return cached;

  if (!colorResolutionEl) {
    colorResolutionEl = document.createElement("div");
    colorResolutionEl.style.display = "none";
    document.body.appendChild(colorResolutionEl);
  }
  colorResolutionEl.style.color = `color-mix(in oklab, var(--near), var(--far) ${roundedPosition}%)`;
  const resolved = getComputedStyle(colorResolutionEl).color;
  resolvedColorCache.set(roundedPosition, resolved);
  return resolved;
}

function distanceColor(distance: number): string {
  const position = Math.round(Math.min(Math.max(distance, 0), 1) * 100);
  return resolveColorMix(position);
}

// react-force-graph forwards nodeLabel's return value to float-tooltip,
// which calls d3 selection.html(text) on a string - a real innerHTML sink.
// Paper.title is untrusted (arXiv/Springer/user uploads), so returning the
// raw string here would be an XSS hole. float-tooltip only calls .html()
// for strings; an HTMLElement is used via its DOM/textContent as-is, so
// building the tooltip content as an element with textContent sidesteps
// the sink entirely instead of trying to escape the string ourselves.
function nodeTooltipLabel(node: GraphNode): string | HTMLElement {
  if (typeof document === "undefined") return node.title;
  const el = document.createElement("span");
  el.textContent = node.title;
  return el;
}

// Matches the h-[420px] wrapper below. react-force-graph falls back to
// window.innerWidth/innerHeight when width/height aren't passed explicitly,
// so without this the canvas overflows its bordered container instead of
// being clipped to it.
const GRAPH_HEIGHT = 420;

export function SimilarityGraph({ assessmentId }: { assessmentId: string }) {
  const [data, setData] = useState<GraphData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [graphWidth, setGraphWidth] = useState(0);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);

  useEffect(() => {
    // Clearing the stale graph/error/selection before fetching by
    // `assessmentId` is intentional - not the accidental-derived-state case
    // this rule otherwise targets. Matches app/assessments/[id]/page.tsx.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setData(null);
    setError(null);
    setSelected(null);
    assessmentApi
      .graph(assessmentId)
      .then(setData)
      .catch(() => setError("The similarity graph isn't available."));
  }, [assessmentId]);

  // A callback ref (not a plain ref + effect with `[]` deps) because the
  // measured div only enters the tree once `data` resolves and the
  // "no papers" branch is ruled out - an effect gated on mount would run
  // before that div exists and never re-fire once it does. This fires
  // exactly when the node attaches (or detaches), so it measures reliably
  // however many times the "no papers" branch toggles.
  const measuredContainerRef = useCallback((el: HTMLDivElement | null) => {
    resizeObserverRef.current?.disconnect();
    resizeObserverRef.current = null;
    if (!el) return;
    setGraphWidth(el.getBoundingClientRect().width);
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;
      if (width) setGraphWidth(width);
    });
    observer.observe(el);
    resizeObserverRef.current = observer;
  }, []);

  const graphData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };
    return {
      nodes: data.nodes.map((n) => ({ ...n })),
      links: data.edges.map((e) => ({ source: e.source, target: e.target, distance: e.distance })),
    };
  }, [data]);

  if (error) {
    return (
      <section className="border-t border-[var(--rule)] py-8">
        <span className="eyebrow">similarity graph</span>
        <p className="mt-3 text-[0.8125rem] text-[var(--ink-soft)]">{error}</p>
      </section>
    );
  }

  if (!data) {
    return <p className="eyebrow py-6">loading graph…</p>;
  }

  const hasPapers = data.nodes.some((n) => n.type === "paper");

  return (
    <section className="border-t border-[var(--rule)] py-8">
      <span className="eyebrow">similarity graph</span>

      {!hasPapers ? (
        <p className="mt-3 text-[0.8125rem] text-[var(--ink-soft)]">
          No related papers found in the corpus.
        </p>
      ) : (
        <div className="mt-4 flex flex-col gap-4 sm:flex-row">
          <div ref={measuredContainerRef} className="h-[420px] flex-1 overflow-hidden border border-[var(--rule)]">
            {graphWidth > 0 && (
              <ForceGraph2D
                ref={configureForces}
                graphData={graphData}
                nodeId="id"
                nodeLabel={nodeTooltipLabel}
                nodeColor={(node: GraphNode) =>
                  node.type === "input" ? "var(--ink)" : distanceColor(node.distance_to_input ?? 1)
                }
                nodeVal={(node: GraphNode) => (node.type === "input" ? 8 : 4)}
                linkColor={(link: { distance: number }) => distanceColor(link.distance)}
                linkWidth={(link: { distance: number }) => Math.max(0.5, 3 * (1 - link.distance))}
                onNodeClick={(node: GraphNode) => setSelected(node.type === "paper" ? node : null)}
                onBackgroundClick={() => setSelected(null)}
                width={graphWidth}
                height={GRAPH_HEIGHT}
              />
            )}
          </div>

          {selected && (
            <aside className="w-full shrink-0 border border-[var(--rule)] bg-[var(--panel)] p-4 sm:w-64">
              <Link
                href={`/papers/${selected.id}`}
                className="text-[0.9375rem] leading-[1.5] underline decoration-[var(--rule)] underline-offset-4 hover:decoration-[var(--ink)]"
              >
                {selected.title}
              </Link>
              <p className="eyebrow mt-3">
                distance to input: {selected.distance_to_input?.toFixed(3) ?? "—"}
              </p>
              {Object.keys(selected.claim_counts).length === 0 ? (
                <p className="mt-2 text-[0.8125rem] text-[var(--ink-soft)]">No extracted claims.</p>
              ) : (
                <ul className="mt-2 space-y-1">
                  {Object.entries(selected.claim_counts).map(([claimType, count]) => (
                    <li key={claimType} className="text-[0.8125rem] text-[var(--ink-soft)]">
                      {count} × {claimType.replace(/_/g, " ")}
                    </li>
                  ))}
                </ul>
              )}
            </aside>
          )}
        </div>
      )}
    </section>
  );
}
