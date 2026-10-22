// The site reads the repository at build time, so the pages and the
// repository never disagree: the model catalog, the README, the benchmark
// results and the benchmark report.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

// scripts/astro.mjs starts Astro in trit/site, so the repository is two up.
const root = resolve(process.cwd(), "..", "..");

export const readRepo = (path: string) => readFileSync(resolve(root, path), "utf8");

export const repoUrl = "https://github.com/Hi9841/tdt";
export const blobUrl = (path: string) => `${repoUrl}/blob/main/${path}`;
