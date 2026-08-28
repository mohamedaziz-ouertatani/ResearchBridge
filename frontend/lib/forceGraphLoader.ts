import type { ComponentType } from "react";

// react-force-graph draws to a <canvas> and reads `window` at import time -
// it cannot run during server-side rendering.
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
export function loadForceGraph2D() {
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
// accessor props would otherwise widen to `{}`. Each consumer recasts to its
// own concrete node/link shape - see SimilarityGraph.tsx/CitationGraph.tsx.
// `distance`'s callback form is intentionally untyped (`any`): the two
// consumers pass link shapes with different fields (distance vs. direction),
// and this interface only describes the shared d3-force plumbing underneath
// react-force-graph, not either consumer's own link type.
export type ForceGraphInstance = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  d3Force: (name: "link" | "charge" | "center" | string) =>
    | { strength?: (v: number) => unknown; distance?: (v: number | ((link: any) => number)) => unknown }
    | undefined;
};
