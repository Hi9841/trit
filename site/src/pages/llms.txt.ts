// llms.txt (https://llmstxt.org): what Trit is and where each fact is, for
// language models and agents. Built from the same facts as the pages.
import { faq, stat, tagline, version, whatIs } from "../lib/facts";
import { rivals } from "../lib/compare";
import { links } from "../lib/links";

export function GET({ site }: { site?: URL }) {
  // Absolute addresses when the build knows the site's address.
  const url = (path: string) => (site ? new URL(path, site).href : path);

  const text = [
    "# Trit",
    "",
    `> ${tagline} ${whatIs}`,
    "",
    `Version ${version}. License: PolyForm Noncommercial 1.0.0 (free for noncommercial use; commercial licenses on ` +
      `request on X ${links.x} or in a GitHub issue ${links.issues}).`,
    "",
    `Measured fact (${stat.date}): ${stat.text}`,
    "",
    "Quick start:",
    "",
    "```",
    "cd trit && cargo build --release",
    "./target/release/trit models",
    "./target/release/trit transcribe take.wav",
    "```",
    "",
    "## Pages",
    "",
    `- [Home](${url(links.home)}): what Trit is, a time race of three runtimes on one model, and the headline numbers`,
    `- [Models](${url(links.models)}): the Hugging Face checkpoints Trit runs, with size, languages, time and the download command`,
    `- [Benchmarks](${url(links.benchmarks)}): the same models on every CPU runtime: time per clip, word error rate, memory`,
    `- [Compare](${url(links.compare)}): Trit against ${rivals.map((r) => r.name).join(", ")}, with when to pick each`,
    `- [Docs](${url(links.docs)}): the command line, the Rust crate, the C ABI, and the Python, C#, Node and Bun bindings`,
    `- [Pricing](${url(links.pricing)}): free for noncommercial use; how to get a commercial license`,
    `- [FAQ](${url(links.faq)}): ten questions with short answers`,
    "",
    "## Data",
    "",
    `- [llms-full.txt](${url(links.llmsFull)}): this file, the FAQ, the full docs and the full benchmark report in one file`,
    `- [benchmarks.json](${url(links.data)}): every benchmark run as JSON (model, runtime, precision, ms, WER, errors, memory)`,
    `- [Raw results](${links.rawResults}): one JSON line per run, per clip set`,
    `- [Source](${links.repo}): the repository; Trit is in trit/`,
    "",
    "## FAQ",
    "",
    ...faq.map(({ q, a }) => `- ${q} ${a}`),
    "",
  ];

  return new Response(text.join("\n"), { headers: { "Content-Type": "text/plain; charset=utf-8" } });
}
