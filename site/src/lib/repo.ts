// The site reads the repository at build time, so the pages and the
// repository never disagree: the model catalog, the README, the benchmark
// results and the benchmark report.
import { execFileSync } from "node:child_process";
import { readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";

// scripts/astro.mjs starts Astro in trit/site, so the repository is two up.
const root = resolve(process.cwd(), "..", "..");

export const readRepo = (path: string) => readFileSync(resolve(root, path), "utf8");

export const repoUrl = "https://github.com/Hi9841/tdt";
export const blobUrl = (path: string) => `${repoUrl}/blob/main/${path}`;

const git = (...args: string[]) => {
  try {
    return execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
  } catch {
    return "";
  }
};

// The day a file last changed: its last commit, or the file's own time when
// it has changes not committed yet (or git is not there). A fresh checkout
// gives every file the time of the checkout, so the commit is the truth.
function changed(path: string) {
  const committed = git("log", "-1", "--format=%cs", "--", path);
  const edited = git("status", "--porcelain", "--", path) !== "";

  return committed && !edited ? committed : statSync(resolve(root, path)).mtime.toISOString().slice(0, 10);
}

/** The day the newest of `paths` (files of the repository) last changed, as YYYY-MM-DD. */
export const updatedOf = (paths: string[]) => paths.map(changed).sort().at(-1) ?? "";
