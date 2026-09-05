// No published types for this package. Minimal ambient declaration covering
// only the export this app actually uses (forceCollide, for SimilarityGraph's
// node-overlap-avoidance force) - not a full d3-force-3d type surface.
declare module "d3-force-3d" {
  export function forceCollide<T = unknown>(radius?: number | ((node: T) => number)): unknown;
}
