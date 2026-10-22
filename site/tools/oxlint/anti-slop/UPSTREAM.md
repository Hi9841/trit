# Upstream

- Source: https://github.com/dmmulroy/anti-slop
- Commit: 13909c9e07590f5a85d93c62712259fe14a883f7 (the snapshot bundled in the
  install-anti-slop skill, `UPSTREAM.txt`)
- Installed: 2026-10-22, into `trit/site/tools/oxlint/anti-slop/`
- Plugins: `index.ts` (generic rules, enabled in `trit/site/.oxlintrc.json`),
  `effect/index.ts` (not enabled: the site does not depend on Effect)
- Deviations: none. The vendored Stylistic code keeps its own `LICENSE` and
  `UPSTREAM.md` in `vendor/eslint-stylistic/`.
