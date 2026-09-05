// react-force-graph draws to a <canvas> and reads `window` at import time -
// it cannot run during server-side rendering.
//
// This app only ever renders the 2D graph and never touches VR/AR. The
// umbrella "react-force-graph" package unconditionally bundles ForceGraph2D
// together with its 3D/AR/VR siblings (3d-force-graph, -ar, -vr), which
// drag in the A-Frame ecosystem and their own separately-versioned copies
// of three.js - triggering three.js's "Multiple instances of Three.js
// being imported" console warning and shipping a lot of unused code to the
// client. "react-force-graph-2d" is the same underlying ForceGraph engine
// (force-graph, plain <canvas> - no three.js at all) published standalone,
// so importing it directly avoids all of that rather than working around it.
export function loadForceGraph2D() {
  return import("react-force-graph-2d").then((mod) => mod.default);
}

// next/dynamic() strips the generic type parameters off ForceGraph2D, so its
// accessor props would otherwise widen to `{}`. Each consumer recasts to its
// own concrete node/link shape - see SimilarityGraph.tsx/CitationGraph.tsx.
// `distance`'s callback form is intentionally untyped (`any`): the two
// consumers pass link shapes with different fields (distance vs. direction),
// and this interface only describes the shared d3-force plumbing underneath
// react-force-graph, not either consumer's own link type.
export type ForceGraphInstance = {
  // The getter form reads back a force force-graph already knows by name
  // (e.g. "link"/"charge"). The two-argument form instead REGISTERS a new
  // named force (e.g. "collide") on the underlying d3 simulation - method
  // shorthand for both signatures so TS treats them as real overloads
  // rather than two incompatible property types.
  d3Force(
    name: "link" | "charge" | "center" | string,
  ): { strength?: (v: number) => unknown; distance?: (v: number | ((link: any) => number)) => unknown } | undefined; // eslint-disable-line @typescript-eslint/no-explicit-any
  d3Force(name: string, forceFn: any): void; // eslint-disable-line @typescript-eslint/no-explicit-any
};
