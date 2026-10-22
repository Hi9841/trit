# Trit site

The one-page site of Trit, built with Astro. Run the commands in `trit/site/`.

| Task | Command |
|---|---|
| Install | `bun install` |
| Develop | `bun run dev` |
| Build to `dist/` | `bun run build` |
| Lint (Oxlint and anti-slop) | `bun run lint` |
| Typecheck | `bun run check` |

## Rules the code does not show

- Start Astro only through `scripts/astro.mjs`; the package scripts do this.
  On Windows, a folder entered in a different letter case than on disk (for
  example `desktop\work` for `Desktop\Work`) makes the build drop the page's
  CSS at random. The launcher runs Astro from the on-disk path.
- The scripts run Astro on Node, not on Bun. Bun is the package manager only.
- `astro check` needs TypeScript 6. Do not move `typescript` to 7 until
  `@astrojs/check` supports it.
- The drawings are SVG strings built in `src/lib/sketch.ts` and styled in
  `src/styles/global.css`, because `set:html` content gets no scoped styles.
- Each drawing acts once it is in view: `data-act` on its figure picks the
  action, `src/lib/motion.ts` sets the classes, and the motion is CSS in
  `global.css` ("Drawing actions"). Start states sit behind `.js`, so without
  JavaScript, and with reduced motion, every drawing shows its last frame.
- No line of text is longer than 80 characters: text blocks use
  `max-width: var(--measure)` (52ch: at most about 78 Geist characters).
- The pages read the repository at build time (`src/lib/data.ts`,
  `src/lib/repo.ts`), so a rebuild picks up every change there:
  - `/models`: `trit/src/catalog.rs`, and the search order from the README.
  - `/benchmarks`: `desktop/bench/runtimes-results.jsonl` for the tables, and
    the summary table, cleanup, method and reproduce sections of
    `docs/engine-comparison.md`. It leaves out the report's sections 4 and 5
    and the summary's numbered notes: they describe an earlier Trit.
  - `/docs`: `trit/README.md`, all sections except Contents and Performance.
- The home page's numbers come from the same files; update them by hand.
- Colours are tokens in `src/styles/global.css`: a light and a dark palette.
  The header button stores the choice in `localStorage`; without a choice the
  page follows the system.
