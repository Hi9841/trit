// The facts on the pages, read from the repository at build time.
import { readRepo } from "./repo";

// ---------- models: trit/src/catalog.rs ----------

export interface Model {
  name: string;
  repo: string;
  kind: string;
  languages: string;
  mb: number;
}

const FAMILY = new Map([
  ["parakeet_tdt", "Parakeet"],
  ["parakeet_ctc", "Parakeet"],
  ["canary", "Canary"],
  ["whisper", "Whisper"],
  ["moonshine", "Moonshine"],
  ["cleanup", "Transcript cleanup"],
]);

export const familyOf = (kind: string) => FAMILY.get(kind) ?? kind;

export const languagesOf = (languages: string) =>
  languages === "en" ? "English" : /^\d+$/.test(languages) ? `${languages} languages` : languages;

export const downloadCommand = (m: Model) =>
  `hf download ${m.repo} --include "*.json" model.safetensors --local-dir models/${m.name}`;

const KNOWN = /Known \{\s*name: "([^"]+)",\s*repo: "([^"]+)",\s*kind: "([^"]+)",\s*languages: "([^"]+)",\s*mb: (\d+),/g;

export const catalog: Model[] = [...readRepo("trit/src/catalog.rs").matchAll(KNOWN)].map((m) => ({
  name: m[1],
  repo: m[2],
  kind: m[3],
  languages: m[4],
  mb: Number(m[5]),
}));

export const DEFAULT_MODEL = "parakeet-redux";

// ---------- benchmarks: desktop/bench/benchmarks.json ----------
// Written by desktop/bench/runtimes_report.py from the raw results: CPU only,
// the test's models, the engines the harness runs today.

export interface Run {
  engine: string;
  runtime: string;
  precision: string;
  trit: boolean;
  /** Trit as it runs when the caller sets nothing. */
  default: boolean;
  ms: number;
  speed: number;
  clean: number;
  other: number;
  /** WER over all 456 clips. */
  all: number;
  errors: number;
  load_mb: number;
  run_mb: number;
}

export interface Bench {
  model: string;
  name: string;
  runs: Run[];
}

export interface Cleanup {
  engine: string;
  runtime: string;
  precision: string;
  trit: boolean;
  default: boolean;
  ms: number;
  wer: number;
  errors: number;
  exact: number;
  cases: number;
}

const data: { date: string; models: Bench[]; cleanup: Cleanup[] } = JSON.parse(
  readRepo("desktop/bench/benchmarks.json"),
);

export const benches: Bench[] = data.models.map((b) => ({ ...b, runs: [...b.runs].sort((x, y) => x.ms - y.ms) }));
export const cleanup: Cleanup[] = [...data.cleanup].sort((x, y) => x.ms - y.ms);
export const benchDate = data.date;

// ---------- places: 1st, 2nd, 3rd per column ----------

/** A column of the tables: its value and whether lower is better. */
export interface Metric<T> {
  name: string;
  value: (r: T) => number;
  lower: boolean;
}

export const speechMetrics: Metric<Run>[] = [
  { name: "Speed", value: (r) => r.speed, lower: false },
  { name: "Clean WER", value: (r) => r.clean, lower: true },
  { name: "Other WER", value: (r) => r.other, lower: true },
  { name: "WER, all clips", value: (r) => r.all, lower: true },
  { name: "Memory after load", value: (r) => r.load_mb, lower: true },
  { name: "Memory after all clips", value: (r) => r.run_mb, lower: true },
];

export const cleanupMetrics: Metric<Cleanup>[] = [
  { name: "Speed", value: (r) => -r.ms, lower: false },
  { name: "Formatted WER", value: (r) => r.wer, lower: true },
  { name: "Exact", value: (r) => r.exact, lower: false },
];

/** The rows that compete for places: every other runtime and Trit's default. */
export const competing = <T extends { trit: boolean; default: boolean }>(rows: T[]) =>
  rows.filter((r) => !r.trit || r.default);

/** 1 + the number of competing rows that are better (a tie shares the place). */
export function place<T extends { trit: boolean; default: boolean }>(rows: T[], metric: Metric<T>, row: T) {
  const v = metric.value(row);

  return 1 + competing(rows).filter((r) => (metric.lower ? metric.value(r) < v : metric.value(r) > v)).length;
}

export const ordinal = (n: number) => `${n}${n === 1 ? "st" : n === 2 ? "nd" : n === 3 ? "rd" : "th"}`;

// ---------- overall: medals per runtime ----------

export interface Medals {
  runtime: string;
  trit: boolean;
  gold: number;
  silver: number;
  bronze: number;
}

/** The medal columns: speed, accuracy over all clips, memory after all clips. */
const MEDAL = new Set(["Speed", "WER, all clips", "Memory after all clips", "Formatted WER"]);

/**
 * Per model and medal column, each runtime's best place among its settings
 * (Trit: its default only), counted as gold, silver and bronze. Sorted as a
 * medal table: gold first, then silver, then bronze.
 */
export function medalTable(): Medals[] {
  const table = new Map<string, Medals>();

  const add = <T extends { runtime: string; trit: boolean; default: boolean }>(rows: T[], metrics: Metric<T>[]) => {
    for (const metric of metrics.filter((m) => MEDAL.has(m.name))) {
      const best = new Map<string, number>();

      for (const row of competing(rows)) {
        const n = place(rows, metric, row);

        best.set(row.runtime, Math.min(n, best.get(row.runtime) ?? n));
      }

      for (const [runtime, n] of best) {
        const m = table.get(runtime) ?? { runtime, trit: runtime === "Trit", gold: 0, silver: 0, bronze: 0 };

        if (n === 1) m.gold += 1;

        if (n === 2) m.silver += 1;

        if (n === 3) m.bronze += 1;

        table.set(runtime, m);
      }
    }
  };

  for (const b of benches) add(b.runs, speechMetrics);
  add(cleanup, cleanupMetrics);

  return [...table.values()].sort((a, b) => b.gold - a.gold || b.silver - a.silver || b.bronze - a.bronze);
}

/** How many times faster Trit's default is than the fastest other runtime of the same model. */
export const lead = (b: Bench) => {
  const trit = b.runs.find((r) => r.default)?.ms ?? Infinity;
  const other = Math.min(...b.runs.flatMap((r) => (r.trit ? [] : [r.ms])));

  return other / trit;
};

/** A run of the benchmark by its engine key; the build fails when it is gone. */
export function runOf(model: string, engine: string) {
  const run = benches.find((b) => b.model === model)?.runs.find((r) => r.engine === engine);

  if (!run) throw new Error(`benchmarks.json has no run "${engine}" of "${model}"`);

  return run;
}

/** The home page's model on Trit (its default) and on the two other runtimes. */
export const redux = {
  trit: runOf("parakeet-redux", "trit"),
  photon: runOf("parakeet-redux", "photon-redux"),
  transformers: runOf("parakeet-redux", "hf-parakeet-redux"),
};

/** A catalog model's size on disk; the build fails when it is gone. */
export function sizeOf(name: string) {
  const model = catalog.find((m) => m.name === name);

  if (!model) throw new Error(`catalog.rs has no model "${name}"`);

  return megabytes(model.mb);
}

/** Megabytes as the tables show them: GB with one decimal from 1000 MB. */
export const megabytes = (mb: number) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${Math.round(mb)} MB`);

// The benchmark's key for a catalog name, where they differ.
const BENCH_KEY = new Map([["parakeet-tdt-0.6b-v3", "parakeet-v3"]]);

/** Trit's default run of the cleanup model. */
export const tritCleanup = cleanup.find((r) => r.default);

// Trit's default run of a catalog model, when it was measured.
export const tritRunOf = (name: string) =>
  benches.find((b) => b.model === (BENCH_KEY.get(name) ?? name))?.runs.find((r) => r.default);
