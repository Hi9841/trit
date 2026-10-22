// Home, crank: the story of the hero drawing (the drawing is `crank` in
// sketch.ts). Audio goes into the machine; Xiaohei listens, winds up, pulls
// the crank three times and catches his breath; each pull turns the drums and
// prints a third of the text; he hops as the time lands. The score is a pure
// function from time to a Pose; one renderer applies it to the drawing.

import { bell, easeOut, flow, mix, run, seg, wave as swing } from "./engine";
import type { Keys } from "./engine";

// ---------- the pose a score returns ----------

interface Pose {
  /** Crank angle in degrees, clockwise, 0 = handle pointing left. */
  phi: number;
  /** Xiaohei: shift, lift off the stool, lean (degrees), squash and stretch. */
  wx: number;
  wy: number;
  lift: number;
  lean: number;
  sx: number;
  sy: number;
  /** Where the eyes look, in body units. */
  gx: number;
  gy: number;
  /** 1 = the hand is on the crank. */
  grip: number;
  /** The free arm: 0 hangs, 1 is up. wag waves it. */
  cheer: number;
  wag: number;
  /** Three pulses of signal on the way into the slot: 0 to 1, or 0 = none. */
  pulses: number[];
  /** How many steps each of the three drums has turned. */
  drums: number[];
  /** The paper unrolled, the ink printed, the loose end swaying (degrees). */
  paper: number;
  ink: number;
  sway: number;
  result: number;
  /** The underline of each numbered step, 0 to 1. */
  steps: number[];
  /** The machine takes a gulp: 0 to 1. */
  gulp: number;
}

const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);
const pull = (phi: number) => 1 - Math.cos((phi * Math.PI) / 180);

// One tick of drum column c, at each time in `at`.
const turned = (t: number, at: number[], c: number) =>
  sum(at.map((when) => easeOut(seg(t, when + 70 * c, when + 70 * c + 220))));

// ---------- the score ----------
// Times in ms from the start of the story. Three pulls of the crank with a
// breath after each (HOLDS); a drum tick and a third of the text per pull;
// then a crouch, a hop, a squash on landing, and the time.

const TURNS: Keys = [[0, 0], [1100, 0], [1900, 360], [2300, 360], [3000, 720], [3350, 720], [4000, 1080], [4300, 1080]];
const TICKS = [1950, 3050, 4050];
const HOLDS = [[1900, 2300], [3000, 3350], [4000, 4300]];
const SQUASH: Keys = [[0, 1], [650, 1], [1050, 0.96], [4300, 0.96], [4480, 0.93], [4620, 1.07], [4980, 1.05], [5060, 0.84], [5240, 1.04], [5500, 1]];

function performer(t: number, idle: number): Pose {
  const body = pull(flow(t - 110, TURNS));
  const sy = flow(t, SQUASH);
  const cheer = flow(t, [[0, 0], [4650, 0], [4880, 1], [5400, 1], [5900, 0.1]]);

  return {
    phi: flow(t, TURNS),
    wx: 6.5 * body,
    wy: 1.1 * swing(t, 3200) * idle + 1.6 * sum(HOLDS.map(([a, b]) => bell(seg(t, a!, b!)))),
    lift: flow(t, [[0, 0], [4560, 0], [4800, 18], [5040, 0]]),
    lean: 2.2 * body + flow(t, [[0, 0], [650, 0], [1050, -3.2], [1500, 0]]) + 0.7 * swing(t, 4300, 0.3) * idle,
    sx: 1 / sy ** 0.6,
    sy,
    gx: flow(t, [[0, -2], [600, -2], [900, 3], [3300, 3], [3700, 5], [4600, 5], [4800, 0], [5100, 5]]),
    gy: flow(t, [[0, -4], [600, -4], [900, 1], [3300, 1], [3700, 0], [4600, 0], [4800, -4], [5100, 2]]),
    grip: flow(t, [[0, 0], [650, 0], [1000, 1], [4250, 1], [4450, 0]]),
    cheer,
    wag: 5 * swing(t, 650) * cheer * seg(t, 4900, 5100),
    pulses: [0, 1].map((i) => seg(t, 450 + 600 * i, 1350 + 600 * i)),
    drums: [0, 1, 2].map((c) => turned(t, TICKS, c)),
    paper: easeOut(seg(t, 1900, 2400)),
    ink: sum(TICKS.map((when) => easeOut(seg(t, when + 40, when + 440)) / 3)),
    sway: 0.7 * swing(t, 3600, 0.1) * idle * seg(t, 5200, 6000),
    result: easeOut(seg(t, 5000, 5600)),
    steps: [seg(t, 300, 800), seg(t, 1100, 1600), seg(t, 1950, 2450)],
    gulp: sum([0, 1].map((i) => bell(seg(t, 1350 + 600 * i, 1590 + 600 * i)))),
  };
}

/** Height of one value in a drum. */
export const DROP = 46;

// ---------- the renderer ----------

const n = (x: number) => x.toFixed(2);

function rig(figure: HTMLElement) {
  const one = (name: string) => figure.querySelector<SVGElement>(`[data-p="${name}"]`)!;
  const all = (name: string) => [...figure.querySelectorAll<SVGElement>(`[data-p="${name}"]`)];

  const parts = {
    worker: one("worker"),
    gaze: one("gaze"),
    arm: one("arm"),
    arm2: one("arm2"),
    crank: one("crank"),
    machine: one("machine"),
    tape: one("tape"),
    paper: one("paper"),
    ink: one("ink"),
    result: one("result"),
    drums: all("drum"),
    pulses: all("pulse"),
    lines: all("uline"),
  };

  // A point of the drawing of Xiaohei, moved by where he stands now.
  const onWorker = (p: Pose, x: number, y: number) => {
    const dx = (x - 170) * p.sx;
    const dy = (y - 386) * p.sy;
    const a = (p.lean * Math.PI) / 180;

    return [
      170 + dx * Math.cos(a) - dy * Math.sin(a) + p.wx,
      386 + dx * Math.sin(a) + dy * Math.cos(a) + p.wy - p.lift,
    ] as const;
  };

  // An arm is one curve that sags, like the hand-drawn ones.
  const arm = (el: SVGElement, from: readonly [number, number], to: readonly [number, number], bowX: number, bowY: number) => {
    const cx = (from[0] + to[0]) / 2 + bowX;
    const cy = (from[1] + to[1]) / 2 + bowY;

    el.setAttribute("d", `M${n(from[0])} ${n(from[1])} Q${n(cx)} ${n(cy)} ${n(to[0])} ${n(to[1])}`);
  };

  return (p: Pose) => {
    parts.worker.setAttribute(
      "transform",
      `translate(${n(p.wx)} ${n(p.wy - p.lift)}) translate(170 386) rotate(${n(p.lean)}) scale(${n(p.sx)} ${n(p.sy)}) translate(-170 -386)`,
    );
    parts.gaze.setAttribute("transform", `translate(${n(p.gx)} ${n(p.gy)})`);
    parts.crank.setAttribute("transform", `rotate(${n(p.phi)} 300 300)`);

    const a = (p.phi * Math.PI) / 180;
    const knob = [300 - 26 * Math.cos(a), 300 - 26 * Math.sin(a)] as const;
    const rest = onWorker(p, 255, 318);

    arm(parts.arm, onWorker(p, 210, 292), [mix(rest[0], knob[0], p.grip), mix(rest[1], knob[1], p.grip)], 0, 10);
    arm(parts.arm2, onWorker(p, 210, 322), onWorker(p, mix(256, 234 + p.wag, p.cheer), mix(324, 262, p.cheer)), 10 * p.cheer, 8 * (1 - p.cheer));

    parts.machine.setAttribute("transform", `translate(0 ${n(1.6 * Math.min(p.gulp, 1))})`);
    parts.drums.forEach((d, c) => d.setAttribute("transform", `translate(0 ${n(-DROP * p.drums[c]!)})`));

    parts.pulses.forEach((el, i) => {
      const k = p.pulses[i]!;

      el.style.visibility = k > 0 && k < 1 ? "visible" : "hidden";
      el.style.strokeDashoffset = n(0.14 - 1.14 * k);
    });

    parts.paper.setAttribute("transform", `translate(524 0) scale(${n(0.06 + 0.94 * p.paper)} 1) translate(-524 0)`);
    parts.ink.setAttribute("width", n(270 * p.ink));
    parts.tape.setAttribute("transform", `rotate(${n(p.sway)} 524 226)`);
    parts.result.style.opacity = n(p.result);
    parts.result.setAttribute("transform", `translate(0 ${n(10 * (1 - p.result))})`);
    parts.lines.forEach((line, i) => (line.style.strokeDashoffset = n(1 - p.steps[i]!)));
  };
}

const END = 6000;

/**
 * Starts the story once the drawing has drawn itself in. While it plays, the
 * lines boil as in the other drawings (motion.ts). Returns stop.
 */
export function mount(figure: HTMLElement) {
  const draw = rig(figure);
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const noise = figure.querySelector("feTurbulence");
  const seed = Number(noise?.getAttribute("seed") ?? "1");
  let shown = seed;

  return run(
    figure,
    END,
    1700,
    (t, idle) => {
      const boil = seed + (t > 0 && t < END ? Math.floor(t / 110) % 3 : 0);

      draw(performer(t, idle));

      if (boil !== shown) noise?.setAttribute("seed", String(boil));
      shown = boil;
    },
    still,
  );
}
