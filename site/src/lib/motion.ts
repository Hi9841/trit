// Plays the action of each drawing (a figure with data-act) when it comes into
// view. The motion itself is CSS (global.css, "Drawing actions"); this script
// only sets the classes: `visible` while on screen (Xiaohei blinks), `act`
// while the action plays, `done` for its last frame. While a drawing acts, its
// lines boil: the wobble filter changes its noise a few times a second, like
// a hand-drawn animation. Clicking a drawing plays its action again.

interface Act {
  delay: number; // ms after the drawing comes into view
  length: number; // ms the action plays
  again: boolean; // plays each time it comes into view, not only the first
}

const ACTS = new Map<string, Act>([
  ["crank", { delay: 1500, length: 1300, again: false }], // after the draw-in
  ["walk", { delay: 200, length: 3000, again: true }],
  ["measure", { delay: 300, length: 2800, again: false }], // more than the slowest bar (--t3)
  ["plug", { delay: 400, length: 700, again: false }],
  ["sleep", { delay: 0, length: 0, again: false }], // only breathes while visible
]);

const still = matchMedia("(prefers-reduced-motion: reduce)").matches;

// The crank's window shows the three values of a trit, turning over.
const turnOver = (text: string, k: number) => {
  const values = text.split(/\s+/);

  return [...values.slice(k % values.length), ...values.slice(0, k % values.length)].join("  ");
};

function play(figure: HTMLElement, act: Act) {
  if (figure.classList.contains("act")) return;

  const noise = figure.querySelector("feTurbulence");
  const seed = noise?.getAttribute("seed") ?? "1";
  const digits = figure.querySelector(".digits");
  const shown = digits?.textContent ?? "";
  let frame = 0;

  figure.classList.remove("done");
  figure.getBoundingClientRect(); // commit the first frame before the action
  figure.classList.add("act");

  const tick = window.setInterval(() => {
    frame += 1;
    noise?.setAttribute("seed", String(Number(seed) + (frame % 3)));

    if (digits) digits.textContent = turnOver(shown, frame);
  }, 110);

  window.setTimeout(() => {
    window.clearInterval(tick);
    noise?.setAttribute("seed", seed);

    if (digits) digits.textContent = shown;
    figure.classList.remove("act");
    figure.classList.add("done");
  }, act.length);
}

for (const figure of document.querySelectorAll<HTMLElement>(".art[data-act]")) {
  const act = ACTS.get(figure.dataset.act ?? "");

  if (!act) continue;

  if (still) {
    figure.classList.add("done");
    continue;
  }

  let played = false;

  figure.style.setProperty("--blink", `${(1 + Math.random() * 4).toFixed(2)}s`);
  figure.addEventListener("click", () => play(figure, act));

  new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        figure.classList.toggle("visible", entry.isIntersecting);

        if (!entry.isIntersecting || (played && !act.again)) continue;
        played = true;
        window.setTimeout(() => play(figure, act), act.delay);
      }
    },
    { threshold: 0.3 },
  ).observe(figure);
}
