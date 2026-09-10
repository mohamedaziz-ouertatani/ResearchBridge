import dagre from "@dagrejs/dagre";
import type { ClaimGraphEdge, ClaimGraphNode } from "./assessmentApi";

const NODE_WIDTH = 220;
const NODE_HEIGHT = 90;

export type LayoutedPosition = { id: string; x: number; y: number };

/**
 * Layered top-to-bottom DAG: evidence ranks above the claims it backs,
 * claims rank above the gap they address. Edges are stored claim -> evidence
 * and claim -> gap (see ClaimGraphEdge), so the "supports"/"contradicts"/
 * "contextualizes" edges are fed to dagre reversed (evidence -> claim) to
 * get the desired rank order; "addresses_gap" edges are fed as-is
 * (claim -> gap already ranks claims above the gap).
 */
export function layoutClaimGraph(nodes: ClaimGraphNode[], edges: ClaimGraphEdge[]): LayoutedPosition[] {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "TB", nodesep: 40, ranksep: 90 });

  for (const node of nodes) {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const edge of edges) {
    if (edge.relationship === "addresses_gap") {
      g.setEdge(edge.source, edge.target);
    } else {
      g.setEdge(edge.target, edge.source);
    }
  }

  dagre.layout(g);

  return nodes.map((node) => {
    const position = g.node(node.id);
    return { id: node.id, x: position.x, y: position.y };
  });
}
