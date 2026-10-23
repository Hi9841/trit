"""The runtime comparison page, from the results of runtimes.py.

usage: python bench/runtimes_report.py [RESULTS.jsonl] [--out ../docs/engine-comparison.html]

Run from desktop/. Reads target/runtimes/results.jsonl by default; the last
result of each engine wins. Writes one self-contained HTML file, and the
tables between the result markers of ../docs/engine-comparison.md.
"""

import json
import sys
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[1]
# The English CPU test: these models, on the CPU only.
TEST_MODELS = ["parakeet-redux", "parakeet-v3", "whisper-base.en", "whisper-tiny", "moonshine-tiny", "moonshine-base"]


def load(path: Path) -> list[dict]:
    """The last result of each engine, with the model and runtime names of
    runtimes.py, so older result lines read the same as new ones."""
    sys.path.insert(0, str(DESKTOP / "bench"))
    from runtimes import ENGINES, MODELS

    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result = json.loads(line)
            # Experiments on a single set are not comparable with full runs.
            complete = {s["set"] for s in result["sets"]} >= {"clean", "other", "tech", "tts16"}
            tested = result["engine"] in ENGINES and ENGINES[result["engine"]][0] in TEST_MODELS
            if tested and complete and ENGINES[result["engine"]][3] == "cpu":
                model, runtime, precision, device, *_ = ENGINES[result["engine"]]
                result.update(model=model, model_name=MODELS[model], runtime=runtime, precision=precision,
                              device=device)
                latest[result["engine"]] = result
    order = {name: i for i, name in enumerate(ENGINES)}
    return sorted(latest.values(), key=lambda r: order[r["engine"]])


def load_cleanup(path: Path) -> list[dict]:
    """The last result of each S1-mini engine of cleanup_runtimes.py."""
    sys.path.insert(0, str(DESKTOP / "bench"))
    from cleanup_runtimes import ENGINES

    latest = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result = json.loads(line)
                if result["engine"] in ENGINES:
                    runtime, precision, *_ = ENGINES[result["engine"]]
                    result.update(runtime=runtime, precision=precision)
                    latest[result["engine"]] = result
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
    """The numbers of the site's benchmarks page (trit/site), already filtered."""
    find = lambda r, name: next(s for s in r["sets"] if s["set"] == name)
    words = lambda r: sum(s["words"] for s in r["sets"])
    models = []
    for model in TEST_MODELS:
        rows = [r for r in results if r["model"] == model]
        if rows:
            models.append({"model": model, "name": rows[0]["model_name"], "runs": [{
                "engine": r["engine"], "runtime": r["runtime"], "precision": r["precision"],
                "trit": r["runtime"] == "Trit", "default": is_default(r),
                "ms": find(r, "clean")["median_ms"], "speed": find(r, "clean")["rtfx"],
                "clean": find(r, "clean")["wer"], "other": find(r, "other")["wer"],
                "all": round(100 * total(r) / words(r), 2), "errors": total(r),
                "load_mb": r.get("load_mb", r["rss_mb"]), "run_mb": r["rss_mb"],
            } for r in rows]})
    return {
        "date": max(r["date"] for r in results),
        "models": models,
        "cleanup": [{
            "engine": r["engine"], "runtime": r["runtime"], "precision": r["precision"],
            "trit": r["runtime"] == "Trit", "default": is_default(r), "ms": r["median_ms"],
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
                   f"{r['exact']} / {r['cases']} | {r['median_ms']:.0f} ms |")
    return "\n".join(out) + "\n"


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
            out.append(f"| {name} | {r['precision']}, {r['device'].upper()} | {clean['median_ms']:.0f} ms | "
                       f"{clean['rtfx']:.0f}x | {clean['wer']:.2f} % | {find(r, 'other')['wer']:.2f} % | "
                       f"{find(r, 'tech')['wer']:.2f} % | {r.get('load_mb', 0):,} MB | {r['rss_mb']:,} MB |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    args = sys.argv[1:]
    out = DESKTOP.parent / "docs" / "engine-comparison.html"
    if "--out" in args:
        i = args.index("--out")
        out = Path(args[i + 1])
        del args[i: i + 2]
    source = Path(args[0]) if args else DESKTOP / "target" / "runtimes" / "results.jsonl"
    results = load(source)
    cleanup = load_cleanup(source.parent / "cleanup.jsonl")
    template = (DESKTOP / "bench" / "runtimes_report.html").read_text(encoding="utf-8")
    page = template.replace("/*RESULTS*/[]", json.dumps(results, separators=(",", ":")))
    page = page.replace("/*CLEANUP*/[]", json.dumps(cleanup, separators=(",", ":")))
    (DESKTOP / "bench" / "benchmarks.json").write_text(json.dumps(site(results, cleanup), indent=1), encoding="utf-8")
    out.write_text(page, encoding="utf-8")
    doc = DESKTOP.parent / "docs" / "engine-comparison.md"
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
