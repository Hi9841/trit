// Hand-drawn line art for the page, built as SVG strings at build time.
// The style is Xiaohei's: black wobbly lines on blank paper, a solid black
// creature with dot eyes doing the work, a few short handwritten notes.

import { DROP } from "./crank";
import { megabytes, redux, sizeOf } from "./data";

const r1 = (n: number) => +n.toFixed(1);

// The measured numbers the drawings write (desktop/bench/benchmarks.json).
const ms = (x: number) => `${Math.round(x)} ms`;
const time = { trit: ms(redux.trit.ms), photon: ms(redux.photon.ms), transformers: ms(redux.transformers.ms) };
const memory = { trit: megabytes(redux.trit.run_mb), transformers: megabytes(redux.transformers.run_mb) };

// Displaces clean vector strokes by a little noise, so they read as pen.
const wobble = (id: string, seed: number) =>
  `<filter id="${id}" x="-5%" y="-5%" width="110%" height="110%" color-interpolation-filters="sRGB">` +
  `<feTurbulence type="fractalNoise" baseFrequency="0.022" numOctaves="2" seed="${seed}"/>` +
  `<feDisplacementMap in="SourceGraphic" scale="3.2"/></filter>`;

const BODY =
  "M2 -52 C24 -53 37 -30 37 -2 C37 26 22 40 0 40 " +
  "C-22 40 -37 27 -36 -3 C-35 -31 -21 -51 2 -52Z";

// A gentle curve with an open arrowhead, in one path.
function arrow(x1: number, y1: number, x2: number, y2: number, bend = 0.15, cls = "ln") {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const cx = (x1 + x2) / 2 - dy * bend;
  const cy = (y1 + y2) / 2 + dx * bend;
  const a = Math.atan2(y2 - cy, x2 - cx);
  const head = (w: number) => `${r1(x2 - 10 * Math.cos(a + w))} ${r1(y2 - 10 * Math.sin(a + w))}`;

  return (
    `<path class="${cls}" d="M${x1} ${y1} Q${r1(cx)} ${r1(cy)} ${x2} ${y2} ` +
    `M${head(-0.5)} L${x2} ${y2} L${head(0.5)}"/>`
  );
}

// A speech waveform: quiet at both ends, loud in the middle.
function wave(x0: number, x1: number, y: number, amp: number, n: number) {
  const step = (x1 - x0) / n;
  let d = `M${x0} ${y}`;

  for (let i = 0; i < n; i++) {
    const env = Math.sin((Math.PI * (i + 0.5)) / n);
    const a = amp * env * (0.45 + 0.55 * Math.abs(Math.sin(i * 1.9))) * (i % 2 ? -1 : 1);
    const xa = x0 + step * i;
    const xb = xa + step;
    d += ` C${r1(xa + step * 0.3)} ${r1(y - a)} ${r1(xb - step * 0.3)} ${r1(y - a)} ${r1(xb)} ${y}`;
  }

  return d;
}

// Standing legs, in Xiaohei's own units; scenes.ts redraws them as he walks.
const STAND = "M-11 38 L-12 72 L-20 73 M10 38 L11 72 L19 73";

// Xiaohei, feet on the ground at y + 72 * s, named for scenes.ts. `back` is
// drawn behind him (arms), `front` in front (what he carries); both move with
// him. data-a is where his feet touch the ground: the pivot for lean, squash
// and stretch.
function xiaohei(name: string, x: number, y: number, s: number, back = "", front = "") {
  return `<g data-p="${name}" data-a="${x} ${y + 72 * s}">${back}
    <g transform="translate(${x} ${y}) scale(${s})">
      <path class="ln" vector-effect="non-scaling-stroke" data-p="${name}-legs" d="${STAND}"/>
      <path class="xh" vector-effect="non-scaling-stroke" d="${BODY}"/>
      <g data-p="${name}-gaze"><g class="eyes"><circle class="eye" cx="-8" cy="-24" r="3.4"/><circle class="eye" cx="9" cy="-25" r="3.4"/></g></g>
    </g>${front}</g>`;
}

// An accent copy of a path, which draws on: a cable going live.
const live = (name: string, d: string) => `<path class="live" data-p="${name}" pathLength="1" d="${d}"/>`;

// Every stroke gets pathLength 1, so the draw-in can run on any path length.
const drawable = (svg: string) => svg.replace(/class="ln/g, 'pathLength="1" class="ln');

// Three steps, left to right: audio goes into a machine marked CPU, Xiaohei
// turns the crank, text comes out with its time. Each step has a numbered
// label on one line. Groups carry --d, the order in which they draw in.
const step = (n: number, x: number, label: string) =>
  `<path class="ln" d="M${x + 11} 103 a11 11 0 1 0 0.1 0"/>` +
  `<text class="hand fade step-n" x="${x + 11}" y="121" text-anchor="middle">${n}</text>` +
  `<text class="hand fade step" x="${x + 32}" y="121">${label}</text>`;

// The crank pulls three times; each pull turns every drum one value.
const VALS = ["−1", "0", "+1"];
const TICKS = 3;

// One drum of the machine's window: a stack of the three values that turns
// up one value per tick. It rests on the value the page has always shown.
function drum(shown: number, x: number) {
  const off = (((shown - TICKS) % 3) + 3) % 3;

  const glyphs = Array.from(
    { length: TICKS + 2 },
    (_, k) => `<text class="hand fade fit" x="${x}" y="${212 + DROP * k}" text-anchor="middle">${VALS[(k + off) % 3]}</text>`,
  );

  return `<g data-p="drum" transform="translate(0 ${-DROP * TICKS})">${glyphs.join("")}</g>`;
}

// Xiaohei on the stool; the eyes sit in their own group so they can look.
const worker = `<g data-p="worker"><g transform="translate(170 300) scale(1.2)">
  <path class="ln legs" vector-effect="non-scaling-stroke" d="M-11 38 L-12 72 L-20 73 M10 38 L11 72 L19 73"/>
  <path class="xh fade" vector-effect="non-scaling-stroke" d="${BODY}"/>
  <g data-p="gaze"><g class="eyes"><circle class="eye fade" cx="-8" cy="-24" r="3.4"/><circle class="eye fade" cx="9" cy="-25" r="3.4"/></g></g>
</g></g>`;

// The underline that marks the step being told.
const underline = (x: number, length: number) =>
  `<path class="uline" data-p="uline" pathLength="1" d="M${x + 32} 128 L${x + 32 + length} 127"/>`;

const TAPE = "M524 204 C594 201 674 207 776 203 L770 214 L777 225 L770 236 L776 247 C674 251 594 245 525 250 Z";

const path = wave(24, 278, 200, 30, 13);

// The hero drawing. Each moving part carries data-p, for the story in
// crank.ts; the markup is its last frame, so without JavaScript it is complete.
export const crank = drawable(`<svg viewBox="0 84 800 362" role="img" aria-label="Three steps. 1: a sound wave, audio at 16 kHz, goes into a slot in a machine. 2: the machine, marked CPU and 1.58-bit, shows minus one, zero and plus one in its window while Xiaohei, a small black creature on a stool, turns its crank. 3: a paper strip that reads 'Restart the pod, then…' comes out of the other side, marked ${time.trit}.">
  <defs>${wobble("wob-crank", 4)}
    <clipPath id="win-crank"><rect x="333" y="178" width="161" height="50"/></clipPath>
    <clipPath id="exit-crank"><rect x="524" y="140" width="300" height="140"/></clipPath>
    <clipPath id="ink-crank"><rect data-p="ink" x="524" y="190" width="270" height="70"/></clipPath>
  </defs>
  <g filter="url(#wob-crank)">
    <g style="--d:0">
      ${step(1, 24, "audio, 16 kHz")}${underline(24, 118)}
      <path class="ln" d="${path}"/>
      <path class="pulse" data-p="pulse" pathLength="1" d="${path}"/>
      <path class="pulse" data-p="pulse" pathLength="1" d="${path}"/>
      <path class="ln" d="M300 180 L278 172 L278 228 L300 220"/>
    </g>
    <g style="--d:1">
      ${step(2, 300, "1.58-bit, CPU")}${underline(300, 118)}
      <g data-p="machine">
        <path class="ln paper" d="M302 146 L518 143 Q524 143 524 149 L526 354 Q526 360 520 360 L307 362 Q300 362 300 356 Z"/>
        <path class="ln" d="M330 176 L496 174 L497 230 L331 232 Z"/>
        <path class="ln" d="M372 272 L452 270 L453 306 L373 308 Z"/>
        <path class="ln" d="M318 362 L318 374 L342 374 L342 362 M486 361 L486 373 L510 373 L510 361"/>
        <g clip-path="url(#win-crank)">${drum(0, 358)}${drum(1, 414)}${drum(2, 470)}</g>
        <text class="hand fade" x="413" y="297" text-anchor="middle">CPU</text>
      </g>
    </g>
    <g style="--d:2">
      <path class="ln" d="M126 387 L214 385 L215 396 L127 398 Z M138 398 L132 436 M204 396 L210 434"/>
      <path class="ln" data-p="arm2" d="M210 322 Q232 334 256 324"/>
      <path class="ln" data-p="arm" d="M210 292 Q236 312 262 304"/>
      ${worker}
      <path class="ln" d="M300 289 a11 11 0 1 0 0.1 0"/>
      <g data-p="crank">
        <path class="ln acc" d="M300 300 L274 300"/>
        <path class="ln acc" d="M269 300 a5 5 0 1 0 10 0 a5 5 0 1 0 -10 0"/>
      </g>
    </g>
    <g style="--d:3">
      ${step(3, 590, "text")}${underline(590, 40)}
    </g>
    <g style="--d:3" clip-path="url(#exit-crank)">
      <g data-p="tape">
        <path class="ln paper" data-p="paper" d="${TAPE}"/>
        <g clip-path="url(#ink-crank)"><text class="hand fade fit" x="536" y="234">Restart the pod, then…</text></g>
      </g>
    </g>
    <g style="--d:4" data-p="result">
      <text class="hand lg acc-t fade" x="640" y="318">${time.trit}</text>
      ${arrow(684, 290, 676, 260, 0.15, "ln acc")}
    </g>
  </g></svg>`);

// Xiaohei alone, for the logo mark and the favicon.
export const mark = `<svg viewBox="-40 -56 80 100" aria-hidden="true"><path fill="currentColor" d="${BODY}"/><circle cx="-8" cy="-24" r="5" fill="var(--paper)"/><circle cx="9" cy="-25" r="5" fill="var(--paper)"/></svg>`;

// A roller suitcase too full to close, and Xiaohei already walking off.

export const suitcase = `<svg viewBox="0 0 800 380" role="img" aria-label="A roller suitcase marked ${memory.transformers} is so full that its sides bulge, its strap strains, a sock hangs out and a cable trails to the floor. Xiaohei is already walking away, carrying a tiny box marked ${memory.trit}. Notes: still packing; already leaving.">
  <defs>${wobble("wob-case", 11)}</defs>
  <g filter="url(#wob-case)">
    <path class="ln" d="M20 348 L780 346"/>
    <g data-p="case" data-o="211 334">
      <path class="ln" d="M176 122 L176 70 M246 122 L246 70 M166 70 L256 70 L256 56 L166 56 Z"/>
      <path class="ln paper" d="M118 132 C110 210 110 262 118 326 Q120 334 130 334 L292 334 Q302 334 304 326 C312 262 312 200 304 132 Q302 122 292 122 L130 122 Q120 122 118 132 Z"/>
      <path class="ln" d="M140 136 L140 320 M282 136 L282 320"/>
      <path class="ln" d="M112 230 L310 228 M196 219 L226 219 L226 239 L196 239 Z"/>
      <path class="ln" data-p="strain" d="M232 212 l9 -7 M234 246 l9 6 M188 212 l-9 -7"/>
      <path class="ln" d="M140 341 a7 7 0 1 0 0.1 0 M282 341 a7 7 0 1 0 0.1 0"/>
      <path class="ln paper" data-p="sock" data-o="308 234" d="M306 232 L322 236 L326 268 C328 278 344 280 346 288 C348 296 334 300 322 294 C312 290 312 280 310 270 Z"/>
      <path class="ln" d="M306 306 C330 308 350 322 360 340 M356 336 L368 334 L370 344 L358 346 Z M368 336 l8 -1 M369 342 l8 0"/>
    </g>
    <text class="hand lg" data-p="case-mb" x="211" y="296" text-anchor="middle">${memory.transformers}</text>
    <g data-p="note1"><text class="hand" x="20" y="92">still packing…</text>${arrow(80, 104, 108, 156, 0.2)}</g>
    <path class="ln" data-p="speed" d="M516 254 L548 254 M508 276 L544 276 M520 298 L542 298"/>
    ${xiaohei(
      "who",
      600,
      275,
      1,
      `<path class="ln" d="M626 258 C636 256 640 258 642 262 M626 292 C634 294 638 292 642 288"/>`,
      `<path class="ln paper" d="M638 260 L676 259 L677 290 L639 291 Z"/>
      <text class="hand sm" data-p="box-mb" x="657" y="280" text-anchor="middle" style="font-size:12px">${memory.trit}</text>`,
    )}
    <g data-p="note2"><text class="hand lg acc-t" x="560" y="176">already leaving</text>${arrow(640, 188, 628, 220, -0.15, "ln acc")}</g>
  </g></svg>`;

// Benchmarks: three runtimes, one tape measure, Xiaohei pulling it out. Each
// bar carries its runtime's name above it. The bars are to scale (the slowest
// is 533 long); data-k is the three times, 8x slower, for scenes.ts.

const slowest = redux.transformers.ms;
const end = (x: number) => r1(131 + (533 * x) / slowest);
const SLOW = 8;

export const measure = `<svg viewBox="0 0 800 400" data-k="${[redux.trit.ms, redux.photon.ms, slowest].map((x) => Math.round(x * SLOW)).join(" ")}" role="img" aria-label="Three hand-drawn bars start at the same line: a short one for Trit, marked ${time.trit}, a longer one for Photon, marked ${time.photon}, and a long one for Transformers, marked ${time.transformers}. Under them, Xiaohei pulls a tape measure out from the start line. Notes: same clips; same timer.">
  <defs>${wobble("wob-measure", 3)}</defs>
  <g filter="url(#wob-measure)">
    <path class="ln" d="M130 70 L131 330"/>
    <text class="hand sm" x="138" y="97">Trit</text>
    <text class="hand sm" x="138" y="167">Photon</text>
    <text class="hand sm" x="138" y="237">Transformers</text>
    <path class="ln acc paper" data-p="bar1" d="M131 106 L${end(redux.trit.ms)} 105 L${end(redux.trit.ms) + 1} 132 L131 133 Z"/>
    <text class="hand acc-t" data-p="lab1" x="${end(redux.trit.ms) + 16}" y="127">${time.trit}</text>
    <path class="ln paper" data-p="bar2" d="M131 176 L${end(redux.photon.ms)} 175 L${end(redux.photon.ms) + 1} 202 L131 203 Z"/>
    <text class="hand" data-p="lab2" x="${end(redux.photon.ms) + 16}" y="197">${time.photon}</text>
    <path class="ln paper" data-p="bar3" d="M131 246 L664 244 L665 271 L131 273 Z"/>
    <text class="hand" data-p="lab3" x="560" y="266">${time.transformers}</text>
    <path class="ln paper" d="M96 324 L130 323 L131 352 L97 353 Z"/>
    <g data-p="tape">
      <path class="ln" d="M130 338 L684 336"/>
      <path class="ln" d="${Array.from({ length: 14 }, (_, i) => `M${170 + i * 38} 338 L${170 + i * 38} ${i % 2 ? 331 : 327}`).join(" ")}"/>
    </g>
    ${xiaohei("who", 722, 278, 1, `<path class="ln" d="M690 326 C680 330 678 336 684 338"/>`)}
    <path class="ln" d="M20 392 L780 390"/>
    <text class="hand" x="150" y="40">same clips</text>
    ${arrow(196, 50, 136, 74, -0.2)}
    <text class="hand" x="250" y="384">same timer</text>
    ${arrow(330, 366, 350, 346, 0.15)}
  </g></svg>`;

// Models: a shelf of checkpoints as published; Xiaohei carries off the small default.

const SHELF: [string, number, number, number, number, string][] = [
  ["canary-1b-v2", 560, 190, 174, 104, "lg"],
  ["parakeet-tdt-0.6b-v3", 420, 120, 175, 82, "sm"],
  ["whisper-large-v3-turbo", 320, 86, 175, 62, "sm"],
  ["whisper-base.en", 330, 70, 329, 40, "sm"],
  ["whisper-tiny", 420, 52, 329, 32, "sm"],
  ["moonshine-tiny", 560, 44, 328, 28, "sm"],
];

// A cardboard box standing on y; it grows from its base, its label does not.
const box = ([name, x, w, y, h, size]: (typeof SHELF)[number], i: number) =>
  `<g data-p="box${i}" data-o="${x + w / 2} ${y}">` +
  `<path class="ln paper" d="M${x} ${y} L${x} ${y - h} L${x + w} ${y - h - 1} L${x + w} ${y} Z"/>` +
  `<path class="ln" d="M${x + w * 0.35} ${y - h} L${x + w * 0.35} ${y - h + h * 0.25}"/></g>` +
  `<text class="hand ${size}" data-p="mb${i}" x="${x + w / 2}" y="${y - h / 2 + 6}" text-anchor="middle">${sizeOf(name)}</text>`;

export const shelf = `<svg viewBox="0 0 800 420" role="img" aria-label="A shelf of boxes of very different sizes, marked ${SHELF.map(([name]) => sizeOf(name)).join(", ").replace(/, ([^,]+)$/, " and $1")}. One place on the lower shelf is empty: Xiaohei walks away carrying that box, marked ${sizeOf("parakeet-redux")}. Notes: as published; the default.">
  <defs>${wobble("wob-shelf", 7)}</defs>
  <g filter="url(#wob-shelf)">
    <path class="ln" d="M20 402 L780 400"/>
    <path class="ln" d="M300 56 L298 400 M770 54 L772 400"/>
    <path class="ln" d="M300 176 L770 174 M300 186 L770 184 M300 330 L770 328 M300 340 L770 338"/>
    ${SHELF.map(box).join("")}
    <path class="ln dash" data-p="slot" d="M492 328 L492 296 L540 295 L540 328"/>
    <g data-p="note1"><text class="hand" x="610" y="36">as published</text>${arrow(660, 46, 650, 66, 0.15)}</g>
    <path class="ln" data-p="speed" d="M240 266 L268 266 M246 288 L276 288"/>
    ${xiaohei(
      "who",
      130,
      328,
      1,
      `<path class="ln" d="M156 252 C168 248 174 252 178 258 M156 290 C166 292 172 290 178 286"/>`,
      `<path class="ln paper" d="M174 254 L224 253 L225 292 L175 293 Z"/>
      <text class="hand sm" data-p="box-mb" x="199" y="278" text-anchor="middle" style="font-size:13px">${sizeOf("parakeet-redux")}</text>`,
    )}
    <g data-p="note2"><text class="hand lg acc-t" x="60" y="190">the default</text>${arrow(150, 202, 190, 246, 0.2, "ln acc")}</g>
  </g></svg>`;

// Docs: one C ABI, a plug for every language; Xiaohei plugs in the last one.

const CABLES = [165, 200, 235].map((y, i) => `M40 ${y - 40 + i * 25} C150 ${y - 40 + i * 25} 210 ${y} 298 ${y}`);
const NODE = "M511 208 C560 190 640 160 700 186 C740 204 760 236 790 246";

export const plugs = `<svg viewBox="0 0 800 380" role="img" aria-label="A box marked trit.h has sockets on both sides. Cables marked Python, C# and Go, Zig, Swift are plugged in on the left. Xiaohei plugs a cable marked Node, Bun into the right side. Note: one C ABI.">
  <defs>${wobble("wob-plugs", 9)}</defs>
  <g filter="url(#wob-plugs)">
    <g data-p="box" data-o="401 264">
      <path class="ln paper" d="M330 140 L470 138 L472 262 L332 264 Z"/>
      <text class="hand lg" x="401" y="212" text-anchor="middle">trit.h</text>
      <path class="ln" d="M472 200 L482 200 M472 216 L482 216"/>
    </g>
    ${[165, 200, 235]
      .map(
        (y, i) =>
          `<path class="ln" d="${CABLES[i]}"/>` +
          `<path class="ln paper" d="M298 ${y - 9} L322 ${y - 9} L322 ${y + 9} L298 ${y + 9} Z M322 ${y - 4} L332 ${y - 4} M322 ${y + 4} L332 ${y + 4}"/>`,
      )
      .join("")}
    <text class="hand" x="36" y="114">Python</text>
    <text class="hand" x="36" y="174">C#</text>
    <text class="hand" x="36" y="234">Go, Zig, Swift</text>
    <path class="ln spark" data-p="spark" d="M478 186 l-7 -9 M486 183 l0 -11 M494 186 l7 -9 M478 232 l-7 9 M494 232 l7 9"/>
    <g data-p="plug">
      <path class="ln paper" d="M486 199 L510 198 L511 217 L487 218 Z"/>
      <path class="ln" d="${NODE}"/>
      ${live("live", NODE)}
    </g>
    ${xiaohei("who", 624, 270, 1, `<path class="ln" d="M590 252 C568 246 540 228 514 214"/>`)}
    <path class="ln" d="M540 344 L780 342"/>
    <text class="hand acc-t" data-p="lab" x="680" y="160">Node, Bun</text>
    <text class="hand lg" x="330" y="96">one C ABI</text>
    ${arrow(400, 106, 402, 132, 0.1)}
  </g></svg>`;

// Docs, last: the work is done. The cable is out, the engine idles, and
// Xiaohei sleeps on his feet against it ("an idle engine uses no CPU").

export const idle = `<svg viewBox="0 0 800 300" role="img" aria-label="The box marked trit.h stands on the floor with its cable unplugged beside it. Xiaohei leans against the box, asleep, eyes closed, while small z letters drift up. Note: idle, no CPU used.">
  <defs>${wobble("wob-idle", 5)}</defs>
  <g filter="url(#wob-idle)">
    <path class="ln" d="M20 262 L780 260"/>
    <path class="ln paper" d="M300 152 L420 150 L422 261 L302 262 Z"/>
    <text class="hand lg" x="361" y="216" text-anchor="middle">trit.h</text>
    <path class="ln" d="M422 196 L432 196 M422 212 L432 212"/>
    <path class="ln" d="M486 250 C560 246 620 238 700 248 C740 253 760 250 790 252"/>
    <path class="ln paper" d="M462 243 L486 242 L487 258 L463 259 Z M462 247 L452 247 M462 254 L452 254"/>
    <g data-p="sleeper" data-o="552 261">
      <g transform="translate(540 189) rotate(-10)">
        <path class="ln" vector-effect="non-scaling-stroke" d="M-11 38 L-12 72 L-20 73 M10 38 L11 72 L19 73"/>
        <g data-p="chest" data-o="0 40">
          <path class="xh" vector-effect="non-scaling-stroke" d="${BODY}"/>
          <path class="shut" d="M-14 -22 q6 5 12 0 M3 -23 q6 5 12 0"/>
        </g>
      </g>
    </g>
    <text class="hand" style="font-size:26px" data-p="z0" x="586" y="124">z</text>
    <text class="hand" style="font-size:32px" data-p="z1" x="604" y="98">z</text>
    <text class="hand" style="font-size:40px" data-p="z2" x="626" y="68">z</text>
    <text class="hand" x="96" y="96">idle, no CPU used</text>
    ${arrow(200, 108, 296, 156, 0.2)}
  </g></svg>`;
