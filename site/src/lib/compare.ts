// Trit against the three runtimes people pick for speech to text on a CPU.
// What each one is comes from its own documentation; what it measured comes
// from the benchmark (data.ts), so the two cannot disagree.
import { benches, type Bench, type Run } from "./data";

/** The facts of one runtime, the rows of the comparison table. */
export interface Facts {
  name: string;
  url: string;
  license: string;
  commercial: string;
  models: string;
  files: string;
  gpu: string;
  streaming: string;
  callFrom: string;
  arm: string;
}

export const ROWS: [keyof Facts, string][] = [
  ["license", "License"],
  ["commercial", "Commercial use"],
  ["models", "Model families"],
  ["files", "Model files"],
  ["gpu", "GPU"],
  ["streaming", "Live streaming"],
  ["callFrom", "Call it from"],
  ["arm", "ARM and Apple Silicon"],
];

export const trit: Facts = {
  name: "Trit",
  url: "https://github.com/Hi9841/tdt",
  license: "PolyForm Noncommercial 1.0.0",
  commercial: "On request",
  models: "Parakeet, Canary, Whisper, Moonshine",
  files: "Hugging Face safetensors, as published",
  gpu: "No, CPU only",
  streaming: "No, whole clips",
  callFrom: "CLI, Rust, C, Python, C#, Node, Bun",
  arm: "Scalar path: same text, much slower",
};

export interface Rival extends Facts {
  /** The runtime's name in benchmarks.json. */
  runtime: string;
  slug: string;
  /** When this runtime is the better choice. */
  pick: string[];
}

export const rivals: Rival[] = [
  {
    runtime: "whisper.cpp",
    slug: "whisper-cpp",
    name: "whisper.cpp",
    url: "https://github.com/ggml-org/whisper.cpp",
    license: "MIT",
    commercial: "Yes",
    models: "Whisper",
    files: "Converted to ggml",
    gpu: "Yes: CUDA, Metal, Vulkan and more",
    streaming: "A real-time microphone example",
    callFrom: "C, C++, community bindings",
    arm: "Yes, tuned for Apple Silicon",
    pick: [
      "You have a GPU, a Mac with Apple Silicon, or an ARM board.",
      "You ship on phones or in the browser.",
      "You need commercial use without asking for a license.",
    ],
  },
  {
    runtime: "faster-whisper (CTranslate2)",
    slug: "faster-whisper",
    name: "faster-whisper",
    url: "https://github.com/SYSTRAN/faster-whisper",
    license: "MIT",
    commercial: "Yes",
    models: "Whisper",
    files: "Converted to CTranslate2",
    gpu: "Yes: CUDA",
    streaming: "No",
    callFrom: "Python",
    arm: "Yes",
    pick: [
      "Your code is Python and you have an NVIDIA GPU.",
      "You need word-level timestamps or its built-in voice activity filter.",
      "You transcribe long files in batches on a GPU, with its batched pipeline.",
    ],
  },
  {
    runtime: "sherpa-onnx",
    slug: "sherpa-onnx",
    name: "sherpa-onnx",
    url: "https://github.com/k2-fsa/sherpa-onnx",
    license: "Apache-2.0",
    commercial: "Yes",
    models: "Many: Zipformer, Paraformer, Whisper, Moonshine, Parakeet and more",
    files: "Exported to ONNX",
    gpu: "Optional: CUDA",
    streaming: "Yes, with streaming models",
    callFrom: "C, C++, Python, Java, Swift, C#, Go, JavaScript and more",
    arm: "Yes, also Android and iOS",
    pick: [
      "You need live text while the speaker talks.",
      "You ship on Android, iOS or a Raspberry Pi.",
      "You also need text to speech, speaker labels or keyword spotting.",
    ],
  },
];

/** One model that both Trit and a rival ran: Trit's default and every setting of the rival. */
export interface Shared {
  bench: Bench;
  trit: Run;
  runs: Run[];
}

export const sharedOf = (rival: Rival): Shared[] =>
  benches.flatMap((bench) => {
    const trit = bench.runs.find((r) => r.default);
    const runs = bench.runs.filter((r) => r.runtime === rival.runtime);

    return trit && runs.length ? [{ bench, trit, runs }] : [];
  });

const min = (runs: Run[], value: (r: Run) => number) =>
  runs.reduce((best, r) => (value(r) < value(best) ? r : best));

/** What the benchmark found for one model, in sentences. */
export function findings({ bench, trit, runs }: Shared, rival: Rival) {
  const fastest = min(runs, (r) => r.ms);
  const fewest = min(runs, (r) => r.errors);
  const leanest = min(runs, (r) => r.run_mb);

  const out = [
    `Trit's default takes ${Math.round(trit.ms)} ms a clip; the fastest ${rival.name} setting ` +
      `(${fastest.precision}) takes ${Math.round(fastest.ms)} ms: ` +
      (fastest.ms > trit.ms
        ? `Trit is ${(fastest.ms / trit.ms).toFixed(1)}× faster.`
        : `${rival.name} is ${(trit.ms / fastest.ms).toFixed(1)}× faster.`),
    trit.errors <= fewest.errors
      ? `Trit makes ${trit.errors} errors over all clips; ${rival.name} at best ${fewest.errors} (${fewest.precision}).`
      : `${rival.name} makes fewer errors: ${fewest.errors} (${fewest.precision}) against Trit's ${trit.errors}.`,
  ];

  if (leanest.run_mb < trit.run_mb) {
    out.push(
      `${rival.name} holds less memory: ${leanest.run_mb} MB after all clips (${leanest.precision}) against ` +
        `Trit's ${trit.run_mb} MB.`,
    );
  }

  return { model: bench.name, lines: out };
}
