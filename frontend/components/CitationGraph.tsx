"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import type { ComponentType } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type CitationEdge, type CitationGraphData, type CitationNode } from "@/lib/api";
import { loadForceGraph2D, type ForceGraphInstance } from "@/lib/forceGraphLoader";

type GraphLink = { source: string; target: string; direction: CitationEdge["direction"] };

const ForceGraph2D = dynamic(loadForceGraph2D, {
  ssr: false,
}) as unknown as ComponentType<{
  graphData: { nodes: CitationNode[]; links: GraphLink[] };
  nodeId: string;
  nodeLabel: (node: CitationNode) => string | HTMLElement;
  nodeColor: (node: CitationNode) => string;
  nodeVal: (node: CitationNode) => number;
  linkColor: (link: GraphLink) => string;
  linkDirectionalArrowLength: number;
  linkDirectionalArrowRelPos: number;
  onNodeClick: (node: CitationNode) => void;
  onBackgroundClick: () => void;
  width: number;
  height: number;
  ref?: (instance: ForceGraphInstance | null) => void;
}>;

// Same spacing convention as SimilarityGraph - see that component's comment
// for why these values (not the library defaults) were chosen.
const CHARGE_STRENGTH = -200;
const LINK_DISTANCE = 120;

function configureForces(instance: ForceGraphInstance | null) {
  if (!instance) return;
  instance.d3Force("charge")?.strength?.(CHARGE_STRENGTH);
  instance.d3Force("link")?.distance?.(LINK_DISTANCE);
}

// Same XSS-avoidance reasoning as SimilarityGraph's nodeTooltipLabel: paper
// titles are untrusted (arXiv/Springer/Semantic Scholar/CORE/user uploads),
// and float-tooltip calls .html() on a plain string, so build the tooltip
// as a real element with textContent instead of returning raw text.
function nodeTooltipLabel(node: CitationNode): string | HTMLElement {
  if (typeof document === "undefined") return node.title;
  const el = document.createElement("span");
  el.textContent = node.title;
  return el;
}

const GRAPH_HEIGHT = 420;

export function CitationGraph({ paperId }: { paperId: string }) {
  const [data, setData] = useState<CitationGraphData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<CitationNode | null>(null);
  const [graphWidth, setGraphWidth] = useState(0);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setData(null);
    setError(null);
    setSelected(null);
    api
      .citations(paperId)
      .then(setData)
      .catch(() => setError("The citation graph isn't available."));
  }, [paperId]);

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
      links: data.edges.map((e) => ({ source: e.source, target: e.target, direction: e.direction })),
    };
  }, [data]);

  if (error) {
    return (
      <section className="border-t border-[var(--rule)] py-8">
        <span className="eyebrow">citation graph</span>
        <p className="mt-3 text-[0.8125rem] text-[var(--ink-soft)]">{error}</p>
      </section>
    );
  }

  if (!data) {
    return <p className="eyebrow py-6">loading citations…</p>;
  }

  const hasEdges = data.edges.length > 0;

  return (
    <section className="border-t border-[var(--rule)] py-8">
      <span className="eyebrow">citation graph</span>

      {!hasEdges ? (
        <p className="mt-3 text-[0.8125rem] text-[var(--ink-soft)]">
          No citation data yet for this paper.
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
                nodeColor={(node: CitationNode) => (node.type === "center" ? "var(--ink)" : "var(--ink-soft)")}
                nodeVal={(node: CitationNode) => (node.type === "center" ? 8 : 4)}
                linkColor={(link: GraphLink) => (link.direction === "cites" ? "var(--near)" : "var(--far)")}
                linkDirectionalArrowLength={4}
                linkDirectionalArrowRelPos={1}
                onNodeClick={(node: CitationNode) => setSelected(node.type === "paper" ? node : null)}
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
            </aside>
          )}
        </div>
      )}
    </section>
  );
}
