// Starts the story of each drawing (a figure with data-scene). The hero's
// crank has its own story (crank.ts); every other drawing's is in scenes.ts.
// Each story plays when its drawing comes into view, pauses off screen, and
// plays again on a click. With reduced motion, each shows its last frame.

import { mount as crank } from "./crank";
import { mount } from "./scenes";

for (const figure of document.querySelectorAll<HTMLElement>(".art[data-scene]")) {
  const scene = figure.dataset.scene ?? "";

  // Each drawing blinks on its own beat, never in step with another.
  figure.style.setProperty("--blink", `${(1 + Math.random() * 4).toFixed(2)}s`);

  if (scene === "crank") crank(figure);
  else mount(figure, scene);
}
