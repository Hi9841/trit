// A small motion engine for the drawings: the web port of the clocks, tracks
// and shaping helpers of rossark-motion-skill. A scene is a pure function of
// time, `pose(t, idle)`; one clock calls it every frame, and only while the
// drawing is on screen.

export type Keys = ReadonlyArray<readonly [number, number]>;

export const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

/** A window of the scene clock, as 0 to 1. */
export const seg = (t: number, from: number, to: number) => clamp01((t - from) / (to - from));

/** A smooth -1 to 1, for sway, bob and breathing. */
export const wave = (t: number, period: number, phase = 0) => Math.sin(2 * Math.PI * (t / period + phase));

/** A bump: 0, up to 1, and back to 0 as p runs 0 to 1. */
export const bell = (p: number) => Math.sin(Math.PI * clamp01(p));

export const mix = (a: number, b: number, k: number) => a + (b - a) * k;

function bezier(x1: number, y1: number, x2: number, y2: number) {
  const cx = 3 * x1;
  const bx = 3 * (x2 - x1) - cx;
  const ax = 1 - cx - bx;
  const cy = 3 * y1;
  const by = 3 * (y2 - y1) - cy;
  const ay = 1 - cy - by;
  const along = (s: number, a: number, b: number, c: number) => ((a * s + b) * s + c) * s;

  return (x: number) => {
    if (x <= 0) return 0;

    if (x >= 1) return 1;

    let lo = 0;
    let hi = 1;

    for (let i = 0; i < 18; i++) {
      const mid = (lo + hi) / 2;

      if (along(mid, ax, bx, cx) < x) lo = mid;
      else hi = mid;
    }

    return along((lo + hi) / 2, ay, by, cy);
  };
}

/** Things that open or arrive. */
export const easeOut = bezier(0.23, 1, 0.32, 1);

/** Things that travel. */
export const easeInOut = bezier(0.77, 0, 0.175, 1);

/** The slope at key k that never overshoots a key (monotone cubic). */
function tangent(keys: Keys, k: number) {
  const last = keys.length - 1;

  if (k <= 0 || k >= last) return 0;

  const before = (keys[k]![1] - keys[k - 1]![1]) / (keys[k]![0] - keys[k - 1]![0]);
  const after = (keys[k + 1]![1] - keys[k]![1]) / (keys[k + 1]![0] - keys[k]![0]);

  return before * after <= 0 ? 0 : (2 * before * after) / (before + after);
}

/**
 * A chain of poses with continuous speed. It passes through every key and
 * never overshoots one, so a body does not stop dead at each pose. Repeat a
 * value in two keys to hold it.
 */
export function flow(t: number, keys: Keys) {
  const last = keys.length - 1;

  if (t <= keys[0]![0]) return keys[0]![1];

  if (t >= keys[last]![0]) return keys[last]![1];

  let i = 0;

  while (t > keys[i + 1]![0]) i++;

  const [t0, v0] = keys[i]!;
  const [t1, v1] = keys[i + 1]!;
  const h = t1 - t0;
  const u = (t - t0) / h;
  const u2 = u * u;
  const u3 = u2 * u;

  return (
    (2 * u3 - 3 * u2 + 1) * v0 +
    (u3 - 2 * u2 + u) * h * tangent(keys, i) +
    (-2 * u3 + 3 * u2) * v1 +
    (u3 - u2) * h * tangent(keys, i + 1)
  );
}

/**
 * Runs `frame(t, idle)` while the figure is on screen and the tab is shown.
 * t starts at -lead, so a scene can wait for the drawing to draw itself in.
 * `idle` is 1 for the small loops that say "alive", 0 for a still page.
 * With reduced motion the scene shows its last frame, once. Clicking the
 * figure plays the scene again from its start.
 */
export function run(
  figure: HTMLElement,
  end: number,
  lead: number,
  frame: (t: number, idle: number) => void,
  still: boolean,
) {
  if (still) {
    frame(end, 0);

    return () => {};
  }

  let t = -lead;
  let prev = 0;
  let raf = 0;
  let onScreen = false;

  frame(t, 1);

  const step = (now: number) => {
    // A long gap (a stalled tab) counts as one frame, not a jump.
    t += prev ? Math.min(now - prev, 64) : 0;
    prev = now;
    frame(t, 1);
    raf = requestAnimationFrame(step);
  };

  const sync = () => {
    const want = onScreen && !document.hidden;

    figure.classList.toggle("visible", want);

    if (want && !raf) {
      prev = 0;
      raf = requestAnimationFrame(step);
    } else if (!want && raf) {
      cancelAnimationFrame(raf);
      raf = 0;
    }
  };

  const watcher = new IntersectionObserver(
    (entries) => {
      onScreen = entries.some((e) => e.isIntersecting);
      sync();
    },
    { threshold: 0.3 },
  );

  const again = () => {
    t = 0;
  };

  watcher.observe(figure);
  document.addEventListener("visibilitychange", sync);
  figure.addEventListener("click", again);

  return () => {
    cancelAnimationFrame(raf);
    watcher.disconnect();
    document.removeEventListener("visibilitychange", sync);
    figure.removeEventListener("click", again);
  };
}
