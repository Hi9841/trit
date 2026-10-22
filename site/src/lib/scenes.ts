// The stories of the site's drawings (sketch.ts), all but the hero's
// (crank.ts). Xiaohei acts each scene: he walks, glances, leans and hops; the
// drawing reacts. A score is a pure function of time that returns a Frame
// (where Xiaohei is and what each named part does); one renderer applies it.
// motion.ts starts each story when its drawing comes into view.

import { bell, clamp01, easeInOut, easeOut, flow, mix, run, seg, wave as swing } from "./engine";
import type { Keys } from "./engine";

// ---------- what a score returns ----------

/** A named part. Transforms turn and scale about the part's data-o. */
interface Part {
  x?: number;
  y?: number;
  r?: number;
  sx?: number;
  sy?: number;
  o?: number;
  /** Shown from the left, 0 to 1. */
  reveal?: number;
  /** An accent path drawn on, 0 to 1. */
  draw?: number;
}

/** Xiaohei: offset from where he is drawn, lean, squash, gaze, and his walk. */
interface Actor {
  name: string;
  x?: number;
  y?: number;
  lean?: number;
  sx?: number;
  sy?: number;
  gx?: number;
  gy?: number;
  /** Walk cycles, from the ground he has covered, so the feet never skate. */
  walk?: number;
  /** 0 stands, 1 strides. */
  step?: number;
  dir?: number;
}

interface Frame {
  actors: Actor[];
  parts: [string, Part][];
}

// ---------- shared moves ----------

const TAU = 2 * Math.PI;
const breath = (t: number, idle: number) => 1.1 * swing(t, 3200) * idle;
const loop = (t: number, period: number, phase = 0) => (((t / period + phase) % 1) + 1) % 1;

// Units of ground per walk cycle. A planted foot moves back at the body's
// speed: its swing amplitude is STRIDE / 2π.
const STRIDE = 40;

/** A walk along keys of x: position, how much it strides, its cycle, its bob. */
function walking(t: number, keys: Keys) {
  const x = flow(t, keys);
  const speed = Math.abs(flow(t + 16, keys) - flow(t - 16, keys)) / 32;
  const step = clamp01(speed / 0.07);

  return { x, step, walk: x / STRIDE, bob: -1.5 * step * Math.abs(Math.sin((TAU * x) / STRIDE)) };
}

// Turn toward something and back: 0, eased up to 1 over [a, a+in], held,
// eased down over [b, b+out].
const glance = (t: number, a: number, b: number, inn = 300, out = 300) =>
  easeInOut(seg(t, a, a + inn)) * (1 - easeInOut(seg(t, b, b + out)));

// ---------- Home: the suitcase ----------
// He walks out with the small box, looks back as the case strains twice and
// the sock swings, then shrugs.

function suitcase(t: number, idle: number): Frame {
  const w = walking(t, [[0, -130], [2000, 0]]);
  const bump = bell(seg(t, 2450, 2700)) + bell(seg(t, 2750, 3050));
  const since = t - 2450;
  const sock = since > 0 ? 9 * Math.sin((TAU * since) / 700) * Math.exp(-since / 900) : 0;
  const back = glance(t, 2100, 3300);
  const note = easeOut(seg(t, 1600, 2000));

  return {
    actors: [
      {
        name: "who",
        x: w.x,
        y: w.bob + breath(t, idle) - 3 * bell(seg(t, 3500, 3900)),
        lean: 3 * w.step,
        gx: mix(3, -5, back),
        walk: w.walk,
        step: w.step,
        dir: 1,
      },
    ],
    parts: [
      ["case", { sx: 1 + 0.03 * bump + 0.006 * swing(t, 2600) * idle, sy: 1 - 0.015 * bump }],
      ["strain", { o: easeOut(seg(t, 2450, 2600)) }],
      ["sock", { r: sock + 2 * swing(t, 3100) * idle }],
      ["speed", { o: w.step }],
      ["note2", { o: note, y: 8 * (1 - note) }],
    ],
  };
}

// ---------- Benchmarks: the tape measure ----------
// k: the three measured times, 8x slower. The bars grow linearly: that is the
// data, so it gets no easing. He pulls the tape out as they race, glances up
// when the first bar ends, and turns to the bars at the end.

function measure(t: number, idle: number, [t1 = 0, t2 = 0, t3 = 1]: number[]): Frame {
  const w = walking(t, [[0, -540], [t3, 0]]);
  const up = glance(t, t1, t1 + 900);
  const turn = easeInOut(seg(t, t3 + 150, t3 + 500));

  return {
    actors: [
      {
        name: "who",
        x: w.x,
        y: w.bob + breath(t, idle) + 2 * bell(seg(t, t3 + 600, t3 + 1000)),
        lean: 4 * w.step,
        gx: mix(mix(3, -3, up), -5, turn),
        gy: mix(mix(0, -4, up), -3, turn),
        walk: w.walk,
        step: w.step,
        dir: 1,
      },
    ],
    parts: [
      ["bar1", { reveal: seg(t, 0, t1) }],
      ["bar2", { reveal: seg(t, 0, t2) }],
      ["bar3", { reveal: seg(t, 0, t3) }],
      ["lab1", { o: easeOut(seg(t, t1, t1 + 250)) }],
      ["lab2", { o: easeOut(seg(t, t2, t2 + 250)) }],
      ["lab3", { o: easeOut(seg(t, t3, t3 + 250)) }],
      ["tape", { reveal: clamp01((554 + w.x) / 554) }],
    ],
  };
}

// ---------- Models: the shelf ----------
// He walks off from the empty place with the default, and glances back.

function shelf(t: number, idle: number): Frame {
  const w = walking(t, [[0, 330], [300, 330], [2600, 0]]);
  const back = glance(t, 2700, 3300);
  const note = easeOut(seg(t, 2400, 2800));

  return {
    actors: [
      {
        name: "who",
        x: w.x,
        y: w.bob + breath(t, idle),
        lean: -3 * w.step,
        gx: mix(-3, 4, back),
        gy: mix(0, -3, back),
        walk: w.walk,
        step: w.step,
        dir: -1,
      },
    ],
    parts: [
      ["speed", { o: w.step }],
      ["note2", { o: note, y: 8 * (1 - note) }],
    ],
  };
}

// ---------- Docs: the plugs ----------
// He walks in with the Node plug, winds up and pushes it home. It sparks, the
// cable goes live, and he hops.

const PUSH: Keys = [[0, 120], [900, 40], [1300, 46], [1600, 0]];

function plugs(t: number, idle: number): Frame {
  const w = walking(t, PUSH);
  const look = easeInOut(seg(t, 2300, 2600));
  const lab = easeOut(seg(t, 2100, 2500));

  return {
    actors: [
      {
        name: "who",
        x: w.x,
        y: w.bob + breath(t, idle) - 8 * bell(seg(t, 2300, 2700)),
        lean: flow(t, [[0, -2], [900, -2], [1250, 4], [1550, -5], [1900, 0]]),
        gx: mix(-4, 0, look),
        gy: mix(1, -1, look),
        walk: w.walk,
        step: w.step,
        dir: -1,
      },
    ],
    parts: [
      ["plug", { x: w.x }],
      ["box", { x: -2 * bell(seg(t, 1600, 1750)) }],
      ["spark", { o: bell(seg(t, 1600, 2000)) }],
      ["live", { draw: easeOut(seg(t, 1650, 2300)) }],
      ["lab", { o: lab, y: 8 * (1 - lab) }],
    ],
  };
}

// ---------- Docs, last: asleep ----------
// Loops only: there is no story in sleep. He breathes, the z's drift up, and
// every 7 s he slips slowly down the box and catches himself.

function asleep(t: number, idle: number): Frame {
  const c = loop(t, 7000) * 7000;
  const slip = easeInOut(seg(c, 3000, 5200)) * (1 - easeOut(seg(c, 5200, 5500)));

  const z = [0, 1, 2].map((i): [string, Part] => {
    const p = loop(t, 4800, -i / 3);

    return idle > 0
      ? [`z${i}`, { x: -8 + 14 * p, y: 14 - 28 * p, o: Math.min(1, 2.5 * bell(p)) }]
      : [`z${i}`, {}];
  });

  return {
    actors: [],
    parts: [
      ["sleeper", { r: -4 * slip * idle, y: -2 * bell(seg(c, 5200, 5600)) * idle }],
      ["chest", { sx: 1 + 0.02 * (0.5 + 0.5 * swing(t, 3400)) * idle, sy: 1 + 0.035 * (0.5 + 0.5 * swing(t, 3400)) * idle }],
      ...z,
    ],
  };
}

// ---------- the scenes: a score, and when its story ends ----------

const SCENES = new Map<string, [(t: number, idle: number, k: number[]) => Frame, (k: number[]) => number]>([
  ["suitcase", [suitcase, () => 4200]],
  ["measure", [measure, ([, , t3 = 0]) => t3 + 1200]],
  ["shelf", [shelf, () => 3600]],
  ["plugs", [plugs, () => 3000]],
  ["idle", [asleep, () => 0]],
]);

// ---------- the renderer ----------

const n = (x: number) => x.toFixed(2);

// Xiaohei's legs in his own units, from the walk: each foot swings about its
// hip and lifts while it travels forward; standing, the toes splay out.
function legs(walk: number, step: number, dir: number) {
  return [0, 1]
    .map((i) => {
      const ph = TAU * walk + i * Math.PI;
      const hip = i ? 10 : -10;
      const fx = (i ? 11 : -12) - (STRIDE / TAU) * step * Math.sin(ph);
      const fy = 72 - 5 * step * Math.max(0, -Math.cos(ph));
      const toe = mix(i ? 8 : -8, 8 * dir, step);

      return `M${hip} 38 L${n(fx)} ${n(fy)} l${n(toe)} 1`;
    })
    .join(" ");
}

function rig(figure: HTMLElement) {
  const els = new Map<string, SVGElement>();

  for (const el of figure.querySelectorAll<SVGElement>("[data-p]")) els.set(el.dataset.p ?? "", el);

  const pivot = (el: SVGElement, key: string) => (el.dataset[key] ?? "0 0").split(" ").map(Number);

  const part = (name: string, p: Part) => {
    const el = els.get(name);

    if (!el) return;

    if (p.x !== undefined || p.y !== undefined || p.r !== undefined || p.sx !== undefined || p.sy !== undefined) {
      const [ox = 0, oy = 0] = pivot(el, "o");

      el.setAttribute(
        "transform",
        `translate(${n(p.x ?? 0)} ${n(p.y ?? 0)}) translate(${ox} ${oy}) rotate(${n(p.r ?? 0)}) ` +
          `scale(${n(p.sx ?? 1)} ${n(p.sy ?? 1)}) translate(${-ox} ${-oy})`,
      );
    }

    if (p.o !== undefined) el.style.opacity = n(p.o);

    if (p.reveal !== undefined) el.style.clipPath = `inset(-20px ${n((1 - p.reveal) * 100)}% -20px -20px)`;

    if (p.draw !== undefined) el.style.strokeDashoffset = n(1 - p.draw);
  };

  const actor = (a: Actor) => {
    const el = els.get(a.name);

    if (!el) return;

    const [fx = 0, fy = 0] = pivot(el, "a");

    el.setAttribute(
      "transform",
      `translate(${n(a.x ?? 0)} ${n(a.y ?? 0)}) translate(${fx} ${fy}) rotate(${n(a.lean ?? 0)}) ` +
        `scale(${n(a.sx ?? 1)} ${n(a.sy ?? 1)}) translate(${-fx} ${-fy})`,
    );
    els.get(`${a.name}-legs`)?.setAttribute("d", legs(a.walk ?? 0, a.step ?? 0, a.dir ?? 1));
    els.get(`${a.name}-gaze`)?.setAttribute("transform", `translate(${n(a.gx ?? 0)} ${n(a.gy ?? 0)})`);
  };

  return (frame: Frame) => {
    frame.actors.forEach(actor);
    frame.parts.forEach(([name, p]) => part(name, p));
  };
}

/** Plays the story of a drawing when it comes into view. Returns stop. */
export function mount(figure: HTMLElement, scene: string) {
  const entry = SCENES.get(scene);

  if (!entry) return () => {};

  const [score, ends] = entry;
  const k = (figure.querySelector("svg")?.dataset.k ?? "").split(" ").flatMap((x) => (x ? [Number(x)] : []));
  const end = ends(k);
  const draw = rig(figure);
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const noise = figure.querySelector("feTurbulence");
  const seed = Number(noise?.getAttribute("seed") ?? "1");
  let shown = seed;

  return run(
    figure,
    end,
    300,
    (t, idleness) => {
      // While a story plays, the lines boil, as in motion.ts.
      const boil = seed + (t > 0 && t < end ? Math.floor(t / 110) % 3 : 0);

      draw(score(t, idleness, k));

      if (boil !== shown) noise?.setAttribute("seed", String(boil));
      shown = boil;
    },
    still,
  );
}
