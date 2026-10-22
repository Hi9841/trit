// Hand-drawn line art for the page, built as SVG strings at build time.
// The style is Xiaohei's: black wobbly lines on blank paper, a solid black
// creature with dot eyes doing the work, a few short handwritten notes.

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

// Xiaohei, feet on the ground at y + 72 * s. A walker has two leg frames
// (stride and passing); global.css swaps them while the drawing acts.
function xiaohei(x: number, y: number, s = 1, legs: "stand" | "walk" = "stand") {
  const leg = (d: string, frame = "") =>
    `<path class="ln legs${frame}" vector-effect="non-scaling-stroke" d="${d}"/>`;

  const drawn =
    legs === "walk"
      ? leg("M-10 38 L-17 72 L-25 72 M10 38 L18 70 L26 69", " step-a") +
        leg("M-8 38 L-6 72 L-14 72 M8 38 L7 71 L15 70", " step-b")
      : leg("M-11 38 L-12 72 L-20 73 M10 38 L11 72 L19 73");

  return `<g transform="translate(${x} ${y}) scale(${s})">${drawn}
    <path class="xh fade" vector-effect="non-scaling-stroke" d="${BODY}"/>
    <g class="eyes"><circle class="eye fade" cx="-8" cy="-24" r="3.4"/>
    <circle class="eye fade" cx="9" cy="-25" r="3.4"/></g></g>`;
}

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

// Every stroke gets pathLength 1, so the draw-in can run on any path length.
const drawable = (svg: string) => svg.replace(/class="ln/g, 'pathLength="1" class="ln');

const EM = "  ";

// Sound goes into a machine marked CPU, Xiaohei turns the crank, text comes out.
// Groups carry --d, the order in which they draw in.
export const crank = drawable(`<svg viewBox="0 0 800 460" role="img" aria-label="Xiaohei, a small black creature on a stool, turns the crank of a machine marked CPU. A sound wave goes into a funnel on top, and a paper strip that says 'restart the pod, then' comes out of the side. Notes: your voice, 16 kHz; 1.58-bit weights; ${time.trit}; no GPU.">
  <defs>${wobble("wob-crank", 4)}</defs>
  <g filter="url(#wob-crank)">
    <g style="--d:0">
      <path class="ln" d="${wave(24, 330, 78, 30, 15)} Q386 80 398 112"/>
      <path class="ln" d="M340 112 Q400 98 460 108 M340 112 L366 182 M460 108 L436 182"/>
    </g>
    <g style="--d:1">
      <path class="ln paper" d="M306 184 L536 180 Q543 180 543 187 L545 372 Q545 379 538 379 L311 381 Q304 381 304 374 Z"/>
      <path class="ln" d="M344 218 L502 216 L503 276 L345 278 Z"/>
      <path class="ln" d="M380 314 L470 312 L471 348 L381 350 Z"/>
      <path class="ln" d="M322 381 L322 394 L346 394 L346 381 M502 380 L502 393 L526 393 L526 380"/>
      <path class="ln" d="M325 200 a3 3 0 1 0 0.1 0 M524 198 a3 3 0 1 0 0.1 0"/>
      <path class="ln" d="M545 290 L545 330"/>
      <text class="hand fade digits" x="424" y="256" text-anchor="middle">−1${EM}0${EM}+1</text>
      <text class="hand fade" x="426" y="340" text-anchor="middle">CPU</text>
    </g>
    <g style="--d:2">
      <path class="ln" d="M146 380 L228 377 L229 388 L147 391 Z M158 391 L152 430 M218 388 L224 428"/>
      <g class="worker">
        <path class="ln" d="M226 268 C242 270 252 286 259 297"/>
        <path class="ln" d="M226 302 C240 307 250 305 258 301"/>
        ${xiaohei(186, 275, 1.4)}
      </g>
      <path class="ln" d="M304 271 a11 11 0 1 0 0.1 0"/>
      <g class="crank">
        <path class="ln acc" d="M304 282 L268 301"/>
        <path class="ln acc" d="M261 298 a6 6 0 1 0 0.1 0"/>
      </g>
    </g>
    <g style="--d:3" class="tape">
      <path class="ln" d="M549 296 C600 292 642 300 690 296 C722 293 744 306 746 322 C747 334 734 341 724 334"/>
      <path class="ln" d="M549 324 C600 320 642 328 690 324 C708 322 718 328 724 334"/>
      <text class="hand sm fade" x="560" y="316" transform="rotate(-1.2 560 316)">restart the pod, then…</text>
    </g>
    <g style="--d:4">
      <text class="hand fade" x="24" y="30">your voice, 16 kHz</text>
      ${arrow(104, 40, 120, 58, 0.2)}
      <text class="hand fade" x="590" y="196">1.58-bit weights</text>
      ${arrow(612, 206, 512, 226, -0.2)}
      <g class="result">
        <text class="hand lg acc-t fade" x="660" y="408">${time.trit}</text>
        ${arrow(690, 378, 700, 344, 0.15, "ln acc")}
      </g>
      <text class="hand fade" x="386" y="448">no GPU</text>
      ${arrow(420, 428, 424, 404, 0.1)}
    </g>
  </g></svg>`);

// A roller suitcase too full to close, and Xiaohei already walking off.
export const suitcase = `<svg viewBox="0 0 800 380" role="img" aria-label="A roller suitcase marked ${memory.transformers} is so full that its sides bulge, its strap strains, a sock hangs out and a cable trails to the floor. Xiaohei is already walking away, carrying a tiny box marked ${memory.trit}. Notes: still packing; already leaving.">
  <defs>${wobble("wob-case", 11)}</defs>
  <g filter="url(#wob-case)">
    <path class="ln" d="M20 348 L780 346"/>
    <g class="case">
    <path class="ln" d="M176 122 L176 70 M246 122 L246 70 M166 70 L256 70 L256 56 L166 56 Z"/>
    <path class="ln paper" d="M118 132 C110 210 110 262 118 326 Q120 334 130 334 L292 334 Q302 334 304 326 C312 262 312 200 304 132 Q302 122 292 122 L130 122 Q120 122 118 132 Z"/>
    <path class="ln" d="M140 136 L140 320 M282 136 L282 320"/>
    <path class="ln" d="M112 230 L310 228 M196 219 L226 219 L226 239 L196 239 Z"/>
    <path class="ln strain" d="M232 212 l9 -7 M234 246 l9 6 M188 212 l-9 -7"/>
    <path class="ln" d="M140 341 a7 7 0 1 0 0.1 0 M282 341 a7 7 0 1 0 0.1 0"/>
    <path class="ln paper" d="M306 232 L322 236 L326 268 C328 278 344 280 346 288 C348 296 334 300 322 294 C312 290 312 280 310 270 Z"/>
    <path class="ln" d="M306 306 C330 308 350 322 360 340 M356 336 L368 334 L370 344 L358 346 Z M368 336 l8 -1 M369 342 l8 0"/>
    <text class="hand lg" x="211" y="296" text-anchor="middle">${memory.transformers}</text>
    </g>
    <text class="hand" x="20" y="92">still packing…</text>
    ${arrow(80, 104, 108, 156, 0.2)}
    <path class="ln speed" d="M516 254 L548 254 M508 276 L544 276 M520 298 L542 298"/>
    <g class="walking">
      <path class="ln" d="M626 258 C636 256 640 258 642 262 M626 292 C634 294 638 292 642 288"/>
      ${xiaohei(600, 275, 1, "walk")}
      <path class="ln paper" d="M638 260 L676 259 L677 290 L639 291 Z"/>
      <text class="hand sm" x="657" y="280" text-anchor="middle" style="font-size:12px">${memory.trit}</text>
    </g>
    <text class="hand lg acc-t" x="560" y="176">already leaving</text>
    ${arrow(640, 188, 628, 220, -0.15, "ln acc")}
  </g></svg>`;

// Xiaohei alone, for the logo mark and the favicon.
export const mark = `<svg viewBox="-40 -56 80 100" aria-hidden="true"><path fill="currentColor" d="${BODY}"/><circle cx="-8" cy="-24" r="5" fill="var(--paper)"/><circle cx="9" cy="-25" r="5" fill="var(--paper)"/></svg>`;

// A cardboard box standing on y, with its size written on the front.
function box(x: number, w: number, y: number, h: number, label: string, size = "sm") {
  return (
    `<path class="ln paper" d="M${x} ${y} L${x} ${y - h} L${x + w} ${y - h - 1} L${x + w} ${y} Z"/>` +
    `<path class="ln" d="M${x + w * 0.35} ${y - h} L${x + w * 0.35} ${y - h + h * 0.25}"/>` +
    `<text class="hand ${size}" x="${x + w / 2}" y="${y - h / 2 + 6}" text-anchor="middle">${label}</text>`
  );
}

// The boxes on the shelf, largest first, for the description.
const SHELF = [
  "canary-1b-v2",
  "parakeet-tdt-0.6b-v3",
  "whisper-large-v3-turbo",
  "whisper-base.en",
  "whisper-tiny",
  "moonshine-tiny",
];

// Models: a shelf of checkpoints as published; Xiaohei carries off the small default.
export const shelf = `<svg viewBox="0 0 800 420" role="img" aria-label="A shelf of boxes of very different sizes, marked ${SHELF.map((name) => sizeOf(name)).join(", ").replace(/, ([^,]+)$/, " and $1")}. One place on the lower shelf is empty: Xiaohei walks away carrying that box, marked ${sizeOf("parakeet-redux")}. Notes: as published; the default.">
  <defs>${wobble("wob-shelf", 7)}</defs>
  <g filter="url(#wob-shelf)">
    <path class="ln" d="M20 402 L780 400"/>
    <path class="ln" d="M300 56 L298 400 M770 54 L772 400"/>
    <path class="ln" d="M300 176 L770 174 M300 186 L770 184 M300 330 L770 328 M300 340 L770 338"/>
    ${box(560, 190, 174, 104, sizeOf("canary-1b-v2"), "lg")}
    ${box(420, 120, 175, 82, sizeOf("parakeet-tdt-0.6b-v3"))}
    ${box(320, 86, 175, 62, sizeOf("whisper-large-v3-turbo"))}
    ${box(330, 70, 329, 40, sizeOf("whisper-base.en"))}
    ${box(420, 52, 329, 32, sizeOf("whisper-tiny"))}
    ${box(560, 44, 328, 28, sizeOf("moonshine-tiny"))}
    <path class="ln dash" d="M492 328 L492 296 L540 295 L540 328"/>
    <text class="hand" x="610" y="36">as published</text>
    ${arrow(660, 46, 650, 66, 0.15)}
    <path class="ln speed" d="M76 266 L48 266 M70 288 L40 288"/>
    <g class="walking">
      <path class="ln" d="M156 252 C168 248 174 252 178 258 M156 290 C166 292 172 290 178 286"/>
      ${xiaohei(130, 328, 1, "walk")}
      <path class="ln paper" d="M174 254 L224 253 L225 292 L175 293 Z"/>
      <text class="hand sm" x="199" y="278" text-anchor="middle" style="font-size:13px">${sizeOf("parakeet-redux")}</text>
    </g>
    <text class="hand lg acc-t" x="60" y="190">the default</text>
    ${arrow(150, 202, 190, 246, 0.2, "ln acc")}
  </g></svg>`;

// Benchmarks: three runtimes, one tape measure, Xiaohei holding the end. The
// bars are to scale (the slowest is 533 px long) and grow 8x slower than measured.
const slowest = redux.transformers.ms;
const end = (x: number) => r1(131 + (533 * x) / slowest);
const grow = (x: number) => `${Math.round(x * 8)}ms`;

export const measure = `<svg viewBox="0 0 800 400" style="--t1:${grow(redux.trit.ms)};--t2:${grow(redux.photon.ms)};--t3:${grow(slowest)}" role="img" aria-label="Three hand-drawn bars start at the same line: a short one marked ${time.trit}, a longer one marked ${time.photon} and a long one marked ${time.transformers}. Under them, Xiaohei pulls a tape measure out from the start line. Notes: same clips; same timer.">
  <defs>${wobble("wob-measure", 3)}</defs>
  <g filter="url(#wob-measure)">
    <path class="ln" d="M130 70 L131 330"/>
    <path class="ln acc paper bar bar-1" d="M131 106 L${end(redux.trit.ms)} 105 L${end(redux.trit.ms) + 1} 132 L131 133 Z"/>
    <text class="hand acc-t lab lab-1" x="${end(redux.trit.ms) + 16}" y="127">${time.trit}</text>
    <path class="ln paper bar bar-2" d="M131 176 L${end(redux.photon.ms)} 175 L${end(redux.photon.ms) + 1} 202 L131 203 Z"/>
    <text class="hand lab lab-2" x="${end(redux.photon.ms) + 16}" y="197">${time.photon}</text>
    <path class="ln paper bar bar-3" d="M131 246 L664 244 L665 271 L131 273 Z"/>
    <text class="hand lab lab-3" x="560" y="266">${time.transformers}</text>
    <path class="ln paper" d="M96 324 L130 323 L131 352 L97 353 Z"/>
    <g class="tape">
      <path class="ln" d="M130 338 L684 336"/>
      <path class="ln" d="${Array.from({ length: 14 }, (_, i) => `M${170 + i * 38} 338 L${170 + i * 38} ${i % 2 ? 331 : 327}`).join(" ")}"/>
    </g>
    <g class="walker">
      <g class="walking">
        <path class="ln" d="M690 326 C680 330 678 336 684 338"/>
        ${xiaohei(722, 278, 1, "walk")}
      </g>
    </g>
    <path class="ln" d="M20 392 L780 390"/>
    <text class="hand" x="150" y="40">same clips</text>
    ${arrow(196, 50, 140, 92, -0.2)}
    <text class="hand" x="250" y="384">same timer</text>
    ${arrow(330, 366, 350, 346, 0.15)}
  </g></svg>`;

// Docs: one C ABI, a plug for every language; Xiaohei plugs in the last one.
export const plugs = `<svg viewBox="0 0 800 380" role="img" aria-label="A box marked trit.h has sockets on both sides. Cables marked Python, C# and Go, Zig, Swift are plugged in on the left. Xiaohei plugs a cable marked Node, Bun into the right side. Note: one C ABI.">
  <defs>${wobble("wob-plugs", 9)}</defs>
  <g filter="url(#wob-plugs)">
    <path class="ln paper" d="M330 140 L470 138 L472 262 L332 264 Z"/>
    <text class="hand lg" x="401" y="212" text-anchor="middle">trit.h</text>
    ${[165, 200, 235]
      .map(
        (y, i) =>
          `<path class="ln" d="M40 ${y - 40 + i * 25} C150 ${y - 40 + i * 25} 210 ${y} 298 ${y}"/>` +
          `<path class="ln paper" d="M298 ${y - 9} L322 ${y - 9} L322 ${y + 9} L298 ${y + 9} Z M322 ${y - 4} L332 ${y - 4} M322 ${y + 4} L332 ${y + 4}"/>`,
      )
      .join("")}
    <text class="hand" x="36" y="114">Python</text>
    <text class="hand" x="36" y="174">C#</text>
    <text class="hand" x="36" y="234">Go, Zig, Swift</text>
    <path class="ln" d="M472 200 L482 200 M472 216 L482 216"/>
    <path class="ln spark" d="M478 186 l-7 -9 M486 183 l0 -11 M494 186 l7 -9 M478 232 l-7 9 M494 232 l7 9"/>
    <g class="pusher">
      <path class="ln paper" d="M486 199 L510 198 L511 217 L487 218 Z"/>
      <path class="ln acc cable" d="M511 208 C560 190 640 160 700 186 C740 204 760 236 790 246"/>
      <g class="walking">
        <path class="ln" d="M590 252 C568 246 540 228 514 214"/>
        ${xiaohei(624, 270, 1, "walk")}
      </g>
    </g>
    <path class="ln" d="M540 344 L780 342"/>
    <text class="hand acc-t lab" x="680" y="160">Node, Bun</text>
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
    <g>
      <g transform="translate(540 189) rotate(-10)">
        <path class="ln" vector-effect="non-scaling-stroke" d="M-11 38 L-12 72 L-20 73 M10 38 L11 72 L19 73"/>
        <g class="torso">
          <path class="xh" vector-effect="non-scaling-stroke" d="${BODY}"/>
          <path class="shut" d="M-14 -22 q6 5 12 0 M3 -23 q6 5 12 0"/>
        </g>
      </g>
    </g>
    <text class="hand zz z1" x="586" y="124">z</text>
    <text class="hand zz z2" x="604" y="98">z</text>
    <text class="hand zz z3" x="626" y="68">z</text>
    <text class="hand" x="96" y="96">idle, no CPU used</text>
    ${arrow(200, 108, 296, 156, 0.2)}
  </g></svg>`;
