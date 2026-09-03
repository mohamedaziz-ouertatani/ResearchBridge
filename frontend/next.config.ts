import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    // Next's root-detection walks up from this directory looking for a
    // lockfile and finds one at C:\Users\user\package-lock.json (outside
    // this git repo, unrelated to this project) before it reaches this
    // project's own frontend/package-lock.json - pin it explicitly so
    // Turbopack always resolves from here, not that ancestor.
    root: __dirname,
  },
};

export default nextConfig;
