// Every rendered code block (src/lib/markdown.ts) gets a copy button in its
// header. The label confirms through a short blur instead of jumping.
for (const block of document.querySelectorAll<HTMLElement>(".codeblock")) {
  const head = block.querySelector(".code-head");
  const code = block.querySelector("pre");

  if (!head || !code) continue;

  const button = document.createElement("button");

  const swap = (text: string) => {
    button.classList.add("swap");
    window.setTimeout(() => {
      button.textContent = text;
      button.classList.remove("swap");
    }, 120);
  };

  button.type = "button";
  button.className = "code-copy";
  button.textContent = "Copy";
  button.setAttribute("aria-label", `Copy the ${head.textContent ?? ""} code`);
  button.addEventListener("click", async () => {
    let copied = true;

    try {
      await navigator.clipboard.writeText(code.innerText);
    } catch {
      copied = false;
    }

    swap(copied ? "Copied" : "Select it");
    window.setTimeout(() => swap("Copy"), 1600);
  });
  head.append(button);

  // A line too long for the block fades at the right edge until scrolled to its end.
  const edge = () => {
    block.classList.toggle("overflows", code.scrollWidth > code.clientWidth + 1);
    block.classList.toggle("at-end", code.scrollLeft + code.clientWidth >= code.scrollWidth - 2);
  };

  edge();
  code.addEventListener("scroll", edge, { passive: true });
  new ResizeObserver(edge).observe(code);
}
