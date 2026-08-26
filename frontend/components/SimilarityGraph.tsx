"use client";

import dynamic from "next/dynamic";
import type { ComponentType } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { assessmentApi, type GraphData, type GraphNode } from "@/lib/assessmentApi";

type GraphLink = { source: string; target: string; distance: number };

// react-force-graph draws to a <canvas> and reads `window` at import time -
// it cannot run during server-side rendering, same reason ProximityGauge's
// sibling gauge components stay client components.
//
// The "react-force-graph" barrel unconditionally imports its VR/AR bundles
// alongside ForceGraph2D, and those assume the A-Frame ecosystem's usual
// script-tag setup: window.AFRAME and window.THREE already populated before
// they load. Without that they throw ReferenceErrors at module-evaluation
// time and crash the whole page, even though this app only ever renders the
// 2D graph and never touches the VR/AR code paths. Populate window.THREE
// with the real "three" module (a direct dependency in package.json, pinned
// to the version react-force-graph itself resolves, so an upgrade of either
// package can't silently break this import) so any eager
// `class X extends THREE.Y` evaluates correctly, and stub window.AFRAME
// with a no-op proxy since nothing here ever registers a real A-Frame scene.
function loadForceGraph2D() {
  const globalWindow = typeof window !== "undefined" ? (window as unknown as { AFRAME?: unknown; THREE?: unknown }) : undefined;
  if (globalWindow && !globalWindow.AFRAME) {
    globalWindow.AFRAME = new Proxy({}, { get: () => () => undefined });
  }
  const threeReady =
    globalWindow && !globalWindow.THREE
      ? // "three" ships no type declarations here and this import exists purely to
        // populate a runtime global, not to use typed exports - see comment above.
        // @ts-expect-error -- untyped module, see comment above
        import("three").then((THREE) => {
          // ES module namespace objects are frozen; several three.js loader
          // add-ons patch themselves onto window.THREE directly
          // (`THREE.ColladaLoader = ...`), so it must be a plain, extensible object.
          globalWindow.THREE = { ...THREE };
        })
      : Promise.resolve();
  return threeReady.then(() => import("react-force-graph")).then((mod) => mod.ForceGraph2D);
}

// next/dynamic() strips the generic type parameters off ForceGraph2D, so its
// accessor props would otherwise widen to `{}`. Recast to the concrete
// node/link shape this component actually passes in - no behavior change.
const ForceGraph2D = dynamic(loadForceGraph2D, {
  ssr: false,
}) as unknown as ComponentType<{
  graphData: { nodes: GraphNode[]; links: GraphLink[] };
  nodeId: string;
  nodeLabel: (node: GraphNode) => string;
  nodeColor: (node: GraphNode) => string;
  nodeVal: (node: GraphNode) => number;
  linkColor: (link: GraphLink) => string;
  linkWidth: (link: GraphLink) => number;
  onNodeClick: (node: GraphNode) => void;
  width: number;
  height: number;
}>;

/** Same "teal near, washed-out far" convention as ProximityGauge - this ramp
 * is reserved for measured cosine distance and nothing else (see globals.css). */
function distanceColor(distance: number): string {
  const position = Math.min(Math.max(distance, 0), 1) * 100;
  return `color-mix(in oklab, var(--near), var(--far) ${position}%)`;
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
    return <p className="py-6 text-[0.8125rem] text-[var(--ink-soft)]">{error}</p>;
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
                graphData={graphData}
                nodeId="id"
                nodeLabel={(node: GraphNode) => node.title}
                nodeColor={(node: GraphNode) =>
                  node.type === "input" ? "var(--ink)" : distanceColor(node.distance_to_input ?? 1)
                }
                nodeVal={(node: GraphNode) => (node.type === "input" ? 8 : 4)}
                linkColor={(link: { distance: number }) => distanceColor(link.distance)}
                linkWidth={(link: { distance: number }) => Math.max(0.5, 3 * (1 - link.distance))}
                onNodeClick={(node: GraphNode) => setSelected(node.type === "paper" ? node : null)}
                width={graphWidth}
                height={GRAPH_HEIGHT}
              />
            )}
          </div>

          {selected && (
            <aside className="w-full shrink-0 border border-[var(--rule)] bg-[var(--panel)] p-4 sm:w-64">
              <p className="text-[0.9375rem] leading-[1.5]">{selected.title}</p>
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
