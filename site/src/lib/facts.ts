// What the site says about Trit, in one place: the homepage, the FAQ,
// llms.txt and the structured data all read it, so they say the same thing.
// The numbers come from the benchmark and the catalog (data.ts).
import { benchDate, benches, catalog, DEFAULT_MODEL, lead, redux, sizeOf } from "./data";
import { links } from "./links";
import { readRepo } from "./repo";

export const NAME = "Trit";

export const version = /^version = "([^"]+)"/m.exec(readRepo("trit/Cargo.toml"))?.[1] ?? "";

/** One line, for titles and link previews. */
export const tagline = "Speech to text on the CPU. No GPU, no Python.";

/** The "What is Trit" paragraph. */
export const whatIs =
  "Trit is a speech-to-text engine for the CPU. It runs Parakeet, Canary, Whisper and Moonshine models " +
  "from Hugging Face on its own Rust kernels, with no GPU, Python or PyTorch.";

const leads = benches.map(lead);
const runtimes = new Set(benches.flatMap((b) => b.runs.map((r) => r.runtime)));
const clips = 456;
const times = (x: number) => `${x.toFixed(1)}×`;

/** Trit's default has the fewest errors over all clips, or a tie, for every model. */
const fewestErrors = benches.every((b) => {
  const trit = b.runs.find((r) => r.default)?.errors ?? Infinity;

  return b.runs.every((r) => r.trit || r.errors >= trit);
});

/** The stat that only this benchmark has. */
export const stat = {
  text:
    `We ran ${benches.length} speech models on ${runtimes.size} CPU runtimes over the same ${clips} English clips. ` +
    `Trit was the fastest runtime for every model, ${times(Math.min(...leads))} to ${times(Math.max(...leads))} ` +
    `ahead of the next one${fewestErrors ? ", with the fewest errors or a tie" : ""}.`,
  date: benchDate,
};

const speechModels = catalog.filter((m) => m.kind !== "cleanup").length;
const speed = Math.round(redux.trit.speed);

/** An answer without its code marks, for structured data. */
export const plain = (text: string) => text.replaceAll("`", "");

export interface Answer {
  q: string;
  a: string;
  more?: { label: string; href: string };
}

/** The ten questions people ask before they use Trit. Commands are in backticks. */
export const faq: Answer[] = [
  {
    q: "What does Trit cost?",
    a:
      "Nothing for noncommercial use. Trit is under the PolyForm Noncommercial License 1.0.0: you can use, " +
      "change and share it for personal, research, education and other noncommercial work. To use it " +
      "commercially, ask for a license on X (@hi9841s) or in a GitHub issue.",
    more: { label: "Pricing", href: links.pricing },
  },
  {
    q: "Do I need a GPU?",
    a: "No. Trit runs on the CPU only, on its own kernels. It needs no GPU, no CUDA, no Python and no ONNX Runtime.",
  },
  {
    q: "How do I install it?",
    a:
      "Build it from source with Rust: run `cargo build --release` in the `trit` folder. Download a model with the " +
      "Hugging Face CLI (`trit models` prints the command), then run `trit transcribe take.wav`. There is no " +
      "prebuilt download of Trit yet.",
    more: { label: "Docs", href: links.docs },
  },
  {
    q: "Which models can it run?",
    a:
      `${speechModels} speech checkpoints from Hugging Face, as published: Parakeet TDT and CTC, Canary, Whisper ` +
      "and Moonshine. Other checkpoints of the same families load the same way. S1-mini cleans transcripts. " +
      `The default is Parakeet Redux, ${sizeOf(DEFAULT_MODEL)} on disk.`,
    more: { label: "Models", href: links.models },
  },
  {
    q: "Which languages does it understand?",
    a:
      "Parakeet TDT v3, Parakeet Redux and Canary understand 25 European languages. Multilingual Whisper " +
      "models understand 99 and can translate to English. Moonshine, Parakeet CTC and the Whisper .en " +
      "models are English only.",
  },
  {
    q: "How fast is it?",
    a:
      `With Parakeet Redux, Trit takes ${Math.round(redux.trit.ms)} ms for a clip of about 7 s on a Ryzen 9 ` +
      `7950X3D: ${speed}× faster than realtime. In our benchmark it was the fastest CPU runtime for each of ` +
      `the ${benches.length} models, ${times(Math.min(...leads))} to ${times(Math.max(...leads))} ahead of the ` +
      "next one. It transcribes whole clips; it does not stream partial text.",
    more: { label: "Benchmarks", href: links.benchmarks },
  },
  {
    q: "How accurate is it?",
    a:
      `As accurate as the reference runtimes. Parakeet Redux makes ${redux.trit.clean.toFixed(2)} % word errors ` +
      `on LibriSpeech test-clean with Trit, ${redux.transformers.clean.toFixed(2)} % in PyTorch. ` +
      (fewestErrors
        ? `Over all ${clips} test clips, Trit's default made the fewest errors, or tied, for every model we measured.`
        : `The error counts of every runtime over all ${clips} test clips are on the Benchmarks page.`),
    more: { label: "Benchmarks", href: links.benchmarks },
  },
  {
    q: "Does my audio leave my computer?",
    a:
      "No. Trit has no network code: audio goes in and text comes out on your computer. Only the model " +
      "download, with the Hugging Face CLI, uses the network.",
  },
  {
    q: "Which computers does it run on?",
    a:
      "x86-64 CPUs with AVX2 (and F16C) or AVX-512 run the fast kernels; Intel CPUs with AVX-VNNI also get the " +
      "fast int8 path. Other CPUs, ARM included, use a scalar path that gives the same text but is much " +
      "slower. Trit builds for Windows, Linux and macOS; only x86-64 Windows has been measured.",
  },
  {
    q: "Which programming languages can call it?",
    a:
      "Rust (the crate), C and C++ (a C ABI), Python, C# and Node or Bun (one-file bindings), and any language " +
      "with a C FFI, such as Go, Swift and Zig. The `trit` command line gives TOON or JSON to scripts and AI agents.",
    more: { label: "Docs", href: links.docs },
  },
];
