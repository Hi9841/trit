// Runs the Astro CLI from the on-disk spelling of this folder.
//
// On Windows a folder can be entered as `...\desktop\work\...` while it is
// `...\Desktop\Work\...` on disk. Vite then sees each module under two
// spellings, and the build drops the page's CSS at random. Starting Astro
// from the real path gives every module one spelling.
import { spawnSync } from "node:child_process";
import { realpathSync } from "node:fs";
import { join } from "node:path";

const root = realpathSync.native(join(import.meta.dirname, ".."));
const cli = join(root, "node_modules", "astro", "bin", "astro.mjs");

const run = spawnSync(process.execPath, [cli, ...process.argv.slice(2)], {
  cwd: root,
  stdio: "inherit",
});

process.exit(run.status ?? 1);
