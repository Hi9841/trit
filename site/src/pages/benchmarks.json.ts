// The benchmark as published by bench/runtimes_report.py, for
// programs and agents: the same data as the Benchmarks page.
import { readRepo } from "../lib/repo";

export function GET() {
  return new Response(readRepo("bench/benchmarks.json"), {
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
}
