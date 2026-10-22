// llms-full.txt: everything an agent needs in one file: what Trit is, the
// FAQ, the comparison, the docs (trit/README.md) and the benchmark report.
import { findings, rivals, sharedOf } from "../lib/compare";
import { faq, stat, tagline, version, whatIs } from "../lib/facts";
import { readRepo } from "../lib/repo";

export function GET() {
  const compare = rivals.flatMap((rival) => [
    `### Trit vs ${rival.name}`,
    "",
    `${rival.name} (${rival.url}): ${rival.license}; models: ${rival.models}; GPU: ${rival.gpu}; streaming: ${rival.streaming}.`,
    "",
    ...sharedOf(rival).flatMap((s) => {
      const found = findings(s, rival);

      return [`${found.model}:`, ...found.lines.map((line) => `- ${line}`), ""];
    }),
    `Pick ${rival.name} when:`,
    ...rival.pick.map((p) => `- ${p}`),
    "",
  ]);

  const text = [
    "# Trit",
    "",
    `> ${tagline} ${whatIs}`,
    "",
    `Version ${version}. Measured fact (${stat.date}): ${stat.text}`,
    "",
    "## FAQ",
    "",
    ...faq.flatMap(({ q, a }) => [`### ${q}`, "", a, ""]),
    "## Compare",
    "",
    ...compare,
    "## Docs (trit/README.md)",
    "",
    readRepo("trit/README.md").replace(/^# Trit\s*/, ""),
    "",
    "## Benchmark report (docs/engine-comparison.md)",
    "",
    readRepo("docs/engine-comparison.md").replace(/^# .*\n/, ""),
  ];

  return new Response(text.join("\n"), { headers: { "Content-Type": "text/plain; charset=utf-8" } });
}
