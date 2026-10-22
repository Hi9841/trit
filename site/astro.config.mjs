// @ts-check
import { defineConfig } from "astro/config";

// GitHub Pages serves the site at https://hi9841.github.io/tdt; the deploy
// workflow sets SITE_URL and SITE_BASE. A local build serves from the root.
export default defineConfig({
  site: process.env.SITE_URL,
  base: process.env.SITE_BASE || "/",
});
