"use client";

import { useEffect, useMemo, useState } from "react";
import { ReactFlow, ReactFlowProvider, Background, Controls } from "@xyflow/react";
import type { Edge, Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { assessmentApi, type ClaimGraphEdge, type ClaimGraphNode } from "@/lib/assessmentApi";
import { layoutClaimGraph } from "@/lib/graphLayout";
import { ClaimNode, type ClaimFlowNode } from "@/components/graph/ClaimNode";
import { EvidenceNode, type EvidenceFlowNode } from "@/components/graph/EvidenceNode";
import { GapNode, type GapFlowNode } from "@/components/graph/GapNode";
import { RelationshipEdge } from "@/components/graph/RelationshipEdge";
import { InspectorDrawer } from "@/components/graph/InspectorDrawer";
import { FilterToolbar, type ConfidenceFilter, type KindFilter } from "@/components/graph/FilterToolbar";
import { GapDensityOverlay } from "@/components/graph/GapDensityOverlay";

const nodeTypes = { claim: ClaimNode, evidence: EvidenceNode, gap: GapNode };
const edgeTypes = { relationship: RelationshipEdge };

export type ClaimEvidenceGraphProps = { assessmentId: string };

function toFlowNode(node: ClaimGraphNode, position: { x: number; y: number }, gapHighlighted: boolean): Node {
  if (node.kind === "claim") {
    const flowNode: ClaimFlowNode = {
      id: node.id,
      type: "claim",
      position,
      data: { label: node.label, claim_type: node.claim_type, confidence: node.confidence, status: node.status },
    };
    return flowNode;
  }
  if (node.kind === "evidence") {
    const flowNode: EvidenceFlowNode = {
      id: node.id,
      type: "evidence",
      position,
      data: { label: node.label, paper_title: node.paper_title, section: node.section },
    };
    return flowNode;
  }
  const flowNode: GapFlowNode = {
    id: node.id,
    type: "gap",
    position,
    data: { label: node.label, gap_status: node.gap_status, highlighted: gapHighlighted },
  };
  return flowNode;
}

function toFlowEdge(edge: ClaimGraphEdge, index: number): Edge {
  return {
    id: `${edge.source}-${edge.target}-${index}`,
    source: edge.source,
    target: edge.target,
    type: "relationship",
    data: { relationship: edge.relationship },
  };
}

export function ClaimEvidenceGraph({ assessmentId }: ClaimEvidenceGraphProps) {
  const [nodes, setNodes] = useState<ClaimGraphNode[]>([]);
  const [edges, setEdges] = useState<ClaimGraphEdge[]>([]);
  const [selected, setSelected] = useState<ClaimGraphNode | null>(null);
  const [kindFilter, setKindFilter] = useState<KindFilter>({ claim: true, evidence: true, gap: true });
  const [confidenceFilter, setConfidenceFilter] = useState<ConfidenceFilter>({ high: true, medium: true, low: true });
  const [gapHighlighted, setGapHighlighted] = useState(false);

  useEffect(() => {
    assessmentApi.claimGraph(assessmentId).then((data) => {
      setNodes(data.nodes);
      setEdges(data.edges);
    });
  }, [assessmentId]);

  const visibleNodes = useMemo(
    () =>
      nodes.filter((node) => {
        if (!kindFilter[node.kind]) return false;
        if (node.kind === "claim" && node.confidence && node.confidence in confidenceFilter) {
          return confidenceFilter[node.confidence as keyof ConfidenceFilter];
        }
        return true;
      }),
    [nodes, kindFilter, confidenceFilter],
  );

  const visibleIds = useMemo(() => new Set(visibleNodes.map((n) => n.id)), [visibleNodes]);
  const visibleEdges = useMemo(
    () => edges.filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target)),
    [edges, visibleIds],
  );

  const positions = useMemo(() => layoutClaimGraph(visibleNodes, visibleEdges), [visibleNodes, visibleEdges]);
  const positionById = useMemo(() => Object.fromEntries(positions.map((p) => [p.id, p])), [positions]);

  const gapCategories = useMemo(() => nodes.find((n) => n.kind === "gap")?.categories ?? [], [nodes]);

  const flowNodes = visibleNodes.map((node) =>
    toFlowNode(node, positionById[node.id] ?? { x: 0, y: 0 }, node.kind === "gap" && gapHighlighted),
  );
  const flowEdges = visibleEdges.map(toFlowEdge);

  return (
    <div className="mt-8">
      <FilterToolbar
        kinds={kindFilter}
        onKindsChange={setKindFilter}
        confidence={confidenceFilter}
        onConfidenceChange={setConfidenceFilter}
      />
      <div className="mt-3">
        <GapDensityOverlay assessmentId={assessmentId} gapCategories={gapCategories} onHighlightChange={setGapHighlighted} />
      </div>
      <div style={{ height: 480 }} className="relative mt-4 border border-[var(--rule-soft)]">
        <ReactFlowProvider>
          <ReactFlow
            nodes={flowNodes}
            edges={flowEdges}
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            onNodeClick={(_, node) => setSelected(nodes.find((n) => n.id === node.id) ?? null)}
            fitView
          >
            <Background />
            <Controls />
          </ReactFlow>
        </ReactFlowProvider>
        <InspectorDrawer node={selected} onClose={() => setSelected(null)} />
      </div>
    </div>
  );
}
