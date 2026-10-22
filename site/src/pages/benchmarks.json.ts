// The benchmark as published by desktop/bench/runtimes_report.py, for
// programs and agents: the same data as the Benchmarks page.
import { readRepo } from "../lib/repo";

export function GET() {
  return new Response(readRepo("desktop/bench/benchmarks.json"), {
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
}
