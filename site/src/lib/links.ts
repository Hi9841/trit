const repo = "https://github.com/Hi9841/trit";

// The site can live under a path (GitHub Pages serves it at /trit), set by
// `base` in astro.config.mjs; every link to a page of the site carries it.
const base = import.meta.env.BASE_URL.replace(/\/$/, "");

export const page = (path: string) => `${base}${path}`;

export const links = {
  repo,
  home: page("/"),
  docs: page("/docs"),
  models: page("/models"),
  benchmarks: page("/benchmarks"),
  compare: page("/compare"),
  pricing: page("/pricing"),
  faq: page("/faq"),
  llms: page("/llms.txt"),
  llmsFull: page("/llms-full.txt"),
  data: page("/benchmarks.json"),
  rawResults: `${repo}/blob/main/bench/runtimes-results.jsonl`,
  issues: `${repo}/issues`,
  license: `${repo}/blob/main/LICENSE`,
  x: "https://x.com/hi9841s",
};
