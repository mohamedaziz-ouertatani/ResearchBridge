import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";
import "@testing-library/jest-dom/vitest";

// Without Vitest's `test.globals: true` (deliberately left off so tests
// must import their own `describe`/`it`/`expect`), RTL's auto-cleanup
// can't detect a global `afterEach` to hook into, so unmounted trees from
// one test stay in the jsdom document and leak into the next test's
// queries within the same file - do it explicitly instead.
afterEach(() => {
  cleanup();
});
