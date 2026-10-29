"""The runtime comparison page, from the results of runtimes.py.

usage: python bench/runtimes_report.py [RESULTS.jsonl] [--out docs/engine-comparison.html]

Run from the repository root. Reads target/runtimes/results.jsonl by default.
Each engine gives the rounds of its newest campaign (runtimes.py --rounds):
the median of each value over the rounds, and the lowest and highest round of
the times. A result without a campaign is one round. Writes one self-contained
HTML file, and the tables between the result markers of
docs/engine-comparison.md.
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The English CPU test: these models, on the CPU only.
SETS = ["clean", "other", "tech", "tts16"]
TEST_MODELS = ["parakeet-redux", "parakeet-v3", "whisper-base.en", "whisper-tiny", "moonshine-tiny", "moonshine-base"]


def newest_campaign(lines: list[dict]) -> dict[str, list[dict]]:
    """The results of each engine's newest campaign, in file order."""
    campaigns = {}
    for i, result in enumerate(lines):
        key = result.get("campaign") or f"line {i}"
        runs = campaigns.setdefault(result["engine"], {})
        runs.setdefault(key, []).append(result)
        runs[key] = runs.pop(key)  # the newest campaign last
    return {engine: list(runs.values())[-1] for engine, runs in campaigns.items()}


def median(values: list[float]) -> float:
    return round(statistics.median(values), 2)


def combine(rounds: list[dict], timed: dict[str, list[dict]]) -> dict:
    """One result from the rounds of a campaign: every number is the median
    over the rounds; each timed value also gets its lowest and highest round
    (`<name>_lo`, `<name>_hi`). `timed` names the lists of rows inside a
    result (key "" for the result itself) and their timed fields."""
    out = dict(rounds[-1], rounds=len(rounds))
    for key, value in rounds[-1].items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = median([r[key] for r in rounds if key in r])
    for field in timed.get("", []):
        out[f"{field}_lo"] = min(r[field] for r in rounds)
        out[f"{field}_hi"] = max(r[field] for r in rounds)
    for key, fields in timed.items():
        if key:
            out[key] = [combine([r[key][i] for r in rounds], {"": fields}) for i in range(len(rounds[-1][key]))]
            for row in out[key]:
                del row["rounds"]
    if "errors" in out and "words" in out:
        out["wer"] = round(100 * out["errors"] / out["words"], 2)
    return out


def load(path: Path) -> list[dict]:
    """One result per engine from its newest campaign (see combine), with the
    model and runtime names of runtimes.py, so older result lines read the
    same as new ones."""
    sys.path.insert(0, str(ROOT / "bench"))
    from runtimes import ENGINES, MODELS

    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result = json.loads(line)
            # Experiments on a single set are not comparable with full runs.
            complete = {s["set"] for s in result["sets"]} >= set(SETS)
            tested = result["engine"] in ENGINES and ENGINES[result["engine"]][0] in TEST_MODELS
            if tested and complete and ENGINES[result["engine"]][3] == "cpu":
                # The sets in one order, so the rounds line up set by set.
                result["sets"] = [next(s for s in result["sets"] if s["set"] == name) for name in SETS]
                lines.append(result)
    latest = {}
    for engine, rounds in newest_campaign(lines).items():
        result = combine(rounds, {"sets": ["median_ms", "rtfx"]})
        model, runtime, precision, device, *_ = ENGINES[engine]
        result.update(model=model, model_name=MODELS[model], runtime=runtime, precision=precision, device=device)
        latest[engine] = result
    order = {name: i for i, name in enumerate(ENGINES)}
    return sorted(latest.values(), key=lambda r: order[r["engine"]])


def load_cleanup(path: Path) -> list[dict]:
    """One result per S1-mini engine of cleanup_runtimes.py, as load does."""
    sys.path.insert(0, str(ROOT / "bench"))
    from cleanup_runtimes import ENGINES

    lines = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line)["engine"] in ENGINES:
                lines.append(json.loads(line))
    latest = {}
    for engine, rounds in newest_campaign(lines).items():
        result = combine(rounds, {"": ["median_ms"]})
        runtime, precision, *_ = ENGINES[engine]
        result.update(runtime=runtime, precision=precision)
        latest[engine] = result
    order = {name: i for i, name in enumerate(ENGINES)}
    return sorted(latest.values(), key=lambda r: order[r["engine"]])


START, END = "<!-- results:start -->", "<!-- results:end -->"
SUMMARY = ("<!-- summary:start -->", "<!-- summary:end -->")
CLEANUP = ("<!-- cleanup:start -->", "<!-- cleanup:end -->")


def is_default(r: dict) -> bool:
    """Trit as it runs when the caller sets nothing: auto precision, the model's own decoding."""
    return r["runtime"] == "Trit" and r["precision"].startswith("auto") and "greedy" not in r["precision"]


def total(r: dict) -> int:
    return sum(s["errors"] for s in r["sets"])


def site(results: list[dict], cleanup: list[dict]) -> dict:
    """The numbers of the site's benchmarks page (site/), already filtered."""
    find = lambda r, name: next(s for s in r["sets"] if s["set"] == name)
    words = lambda r: sum(s["words"] for s in r["sets"])
    models = []
    for model in TEST_MODELS:
        rows = [r for r in results if r["model"] == model]
        if rows:
            models.append({"model": model, "name": rows[0]["model_name"], "runs": [{
                "engine": r["engine"], "runtime": r["runtime"], "precision": r["precision"],
                "trit": r["runtime"] == "Trit", "default": is_default(r),
                "ms": find(r, "clean")["median_ms"], "speed": find(r, "clean")["rtfx"], "rounds": r["rounds"],
                "ms_lo": find(r, "clean")["median_ms_lo"], "ms_hi": find(r, "clean")["median_ms_hi"],
                "speed_lo": find(r, "clean")["rtfx_lo"], "speed_hi": find(r, "clean")["rtfx_hi"],
                "clean": find(r, "clean")["wer"], "other": find(r, "other")["wer"],
                "all": round(100 * total(r) / words(r), 2), "errors": total(r),
                "load_mb": r.get("load_mb", r["rss_mb"]), "run_mb": r["rss_mb"],
            } for r in rows]})
    return {
        "date": max(r["date"] for r in results),
        "models": models,
        "cleanup": [{
            "engine": r["engine"], "runtime": r["runtime"], "precision": r["precision"],
            "trit": r["runtime"] == "Trit", "default": is_default(r), "ms": r["median_ms"], "rounds": r["rounds"],
            "ms_lo": r["median_ms_lo"], "ms_hi": r["median_ms_hi"],
            "wer": r["wer"], "errors": r["errors"], "exact": r["exact"], "cases": r["cases"],
        } for r in cleanup],
    }


def summary(results: list[dict], cleanup: list[dict]) -> str:
    """Errors over all 456 clips: Trit's default against each other runtime's best setting."""
    out = ["| Model | Trit (default) | Other runtimes (best setting each) | Trit time per clip | Fastest other |",
           "|---|---:|---|---:|---:|"]
    clean_ms = lambda r: next(s for s in r["sets"] if s["set"] == "clean")["median_ms"]
    for model in TEST_MODELS:
        rows = [r for r in results if r["model"] == model]
        trit = next((r for r in rows if is_default(r)), None)
        if trit is None:
            continue
        best = {}
        for r in rows:
            if r["runtime"] != "Trit" and (r["runtime"] not in best or total(r) < total(best[r["runtime"]])):
                best[r["runtime"]] = r
        others = sorted(best.values(), key=total)
        lowest = min([total(trit)] + [total(r) for r in others])
        mark = lambda n, lowest=lowest: f"**{n}**" if n == lowest else str(n)
        fastest = min((r for r in rows if r["runtime"] != "Trit"), key=clean_ms)
        out.append(f"| {trit['model_name']} | {mark(total(trit))} | "
                   + ", ".join(f"{r['runtime'].replace(' (PyTorch)', '')} {mark(total(r))}" for r in others)
                   + f" | {clean_ms(trit):.0f} ms | {clean_ms(fastest):.0f} ms ({fastest['runtime'].replace(' (PyTorch)', '')}) |")
    trit = next((r for r in cleanup if is_default(r)), None)
    if trit:
        others = [r for r in cleanup if r["runtime"] != "Trit"]
        lowest = min([trit["errors"]] + [r["errors"] for r in others])
        mark = lambda n, lowest=lowest: f"**{n}**" if n == lowest else str(n)
        fastest = min(others, key=lambda r: r["median_ms"])
        out.append(f"| S1-mini (formatted words, {trit['cases']} cases) | {mark(trit['errors'])} | "
                   + ", ".join(f"{r['runtime'].replace(' (PyTorch)', '')} {r['precision']} {mark(r['errors'])}" for r in others)
                   + f" | {trit['median_ms']:.0f} ms | {fastest['median_ms']:.0f} ms ({fastest['runtime']} {fastest['precision']}) |")
    return "\n".join(out) + "\n"


def cleanup_table(cleanup: list[dict]) -> str:
    out = ["| Runtime | Precision | Formatted WER | Exact sentences | Median per case |", "|---|---|---:|---:|---:|"]
    for r in sorted(cleanup, key=lambda r: r["median_ms"]):
        name = "**Trit**" if r["runtime"] == "Trit" else r["runtime"]
        out.append(f"| {name} | {r['precision']} | {r['wer']:.2f} % ({r['errors']} / {r['words']}) | "
                   f"{r['exact']} / {r['cases']} | {spread(r)} |")
    return "\n".join(out) + "\n"


def spread(row: dict, field: str = "median_ms") -> str:
    """A median time with its lowest and highest round, when there are rounds."""
    lo, hi = row.get(f"{field}_lo", row[field]), row.get(f"{field}_hi", row[field])
    return f"{row[field]:.0f} ms" + (f" ({lo:.0f}–{hi:.0f})" if round(lo) != round(hi) else "")


def between(text: str, marks: tuple[str, str], body: str) -> str:
    if marks[0] not in text or marks[1] not in text:
        return text
    head, rest = text.split(marks[0], 1)
    return f"{head}{marks[0]}\n{body}{marks[1]}{rest.split(marks[1], 1)[1]}"


def markdown(results: list[dict]) -> str:
    """One table per model: its runtimes from fastest to slowest."""
    find = lambda r, name: next(s for s in r["sets"] if s["set"] == name)
    out = []
    for model in dict.fromkeys(r["model"] for r in results):
        rows = sorted((r for r in results if r["model"] == model), key=lambda r: find(r, "clean")["median_ms"])
        out += [f"### {rows[0]['model_name']}", "",
                "| Runtime | Precision | Median per clip | Speed | Clean WER | Other WER | Dev terms WER | Memory after load | Memory after all clips |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            name = "**Trit**" if r["runtime"] == "Trit" else r["runtime"]
            clean = find(r, "clean")
            out.append(f"| {name} | {r['precision']}, {r['device'].upper()} | {spread(clean)} | "
                       f"{clean['rtfx']:.0f}x | {clean['wer']:.2f} % | {find(r, 'other')['wer']:.2f} % | "
                       f"{find(r, 'tech')['wer']:.2f} % | {r.get('load_mb', 0):,} MB | {r['rss_mb']:,} MB |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    args = sys.argv[1:]
    out = ROOT / "docs" / "engine-comparison.html"
    if "--out" in args:
        i = args.index("--out")
        out = Path(args[i + 1])
        del args[i: i + 2]
    source = Path(args[0]) if args else ROOT / "target" / "runtimes" / "results.jsonl"
    results = load(source)
    cleanup = load_cleanup(source.parent / "cleanup.jsonl")
    template = (ROOT / "bench" / "runtimes_report.html").read_text(encoding="utf-8")
    page = template.replace("/*RESULTS*/[]", json.dumps(results, separators=(",", ":")))
    page = page.replace("/*CLEANUP*/[]", json.dumps(cleanup, separators=(",", ":")))
    (ROOT / "bench" / "benchmarks.json").write_text(json.dumps(site(results, cleanup), indent=1), encoding="utf-8")
    out.write_text(page, encoding="utf-8")
    doc = ROOT / "docs" / "engine-comparison.md"
    text = doc.read_text(encoding="utf-8")
    if START in text and END in text:
        head, rest = text.split(START, 1)
        text = f"{head}{START}\n{markdown(results)}{END}{rest.split(END, 1)[1]}"
        text = between(text, SUMMARY, summary(results, cleanup))
        text = between(text, CLEANUP, cleanup_table(cleanup))
        doc.write_text(text, encoding="utf-8", newline="\n")
    print(f"report: {out} and {doc.name} ({len(results)} engines)")


if __name__ == "__main__":
    main()
