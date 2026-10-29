// Markdown from the repository to HTML for the site: GitHub-style heading
// ids, links rewritten for the site, and code highlighted for both themes.
import { Marked, type Token, type Tokens } from "marked";
import { codeToHtml, createCssVariablesTheme } from "shiki";
import { docsHref, landingHref, README } from "./docs";
import { links } from "./links";
import { blobUrl } from "./repo";
import { slug } from "./sections";

export { sections, slug, type Section } from "./sections";

// Pages on this site that replace a file of the repository. The README is
// many pages: a link to one of its headings goes to the page of that heading.
const onSite = new Map([["docs/engine-comparison.md", links.benchmarks]]);

// A link relative to `from` (a file of the repository) becomes a site page,
// a GitHub URL, or stays as it is.
function rewrite(href: string, from: string) {
  if (href.startsWith("#") && from === README) return docsHref(href.slice(1));

  if (/^([a-z]+:|#|\/)/i.test(href)) return href;
  const [path, hash = ""] = href.split("#");
  const parts = from.split("/").slice(0, -1);

  for (const part of path.split("/")) {
    if (part === "..") parts.pop();
    else if (part !== ".") parts.push(part);
  }

  const target = parts.join("/");

  if (target === README) return hash ? docsHref(hash) : landingHref;

  const page = onSite.get(target);

  if (page) return hash ? `${page}#${hash}` : page;

  return blobUrl(target) + (hash ? `#${hash}` : "");
}

const isCode = (token: Token): token is Tokens.Code => token.type === "code";
const isStrong = (token?: Token): token is Tokens.Strong => token?.type === "strong";

// Marked hands renderers raw text; `<dir or name>` must not become an element.
const escapeHtml = (text: string) =>
  text.replace(/&(?!(amp|lt|gt|quot|#\d+);)/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// Code takes its colours from the site's palette (global.css), so it follows
// the theme and stays in two tones: ink, and muted comments.
const theme = createCssVariablesTheme({ name: "site", variablePrefix: "--code-" });

// The name a code block shows in its header. "text" blocks in the README are program output.
const LANGUAGE = new Map([
  ["powershell", "PowerShell"],
  ["bash", "Shell"],
  ["sh", "Shell"],
  ["rust", "Rust"],
  ["toml", "Cargo.toml"],
  ["c", "C"],
  ["python", "Python"],
  ["csharp", "C#"],
  ["js", "JavaScript"],
  ["javascript", "JavaScript"],
  ["text", "Output"],
]);

// Code blocks become highlighted HTML under a header with their language;
// src/lib/codecopy.ts adds a copy button to the header.
async function highlight(tokens: Token[]) {
  for (const [i, token] of tokens.entries()) {
    if (!isCode(token)) continue;

    const lang = token.lang || "text";
    const html = await codeToHtml(token.text, { lang, theme });
    const name = LANGUAGE.get(lang) ?? lang;

    tokens[i] = {
      type: "html",
      block: true,
      pre: true,
      raw: token.raw,
      text: `<div class="codeblock"><div class="code-head"><span>${name}</span></div>${html}</div>\n`,
    };
  }
}

// Renders `md`, a part of the file `from`, to HTML. `raise` moves headings
// up that many levels (a section on its own page: `###` becomes h2).
export async function render(md: string, from: string, raise = 0) {
  const marked = new Marked({
    renderer: {
      heading({ tokens, depth, text }) {
        const level = Math.max(2, depth - raise);

        return `<h${level} id="${slug(text)}">${this.parser.parseInline(tokens)}</h${level}>\n`;
      },
      // Short code and flags stay on one line; long code may wrap between words.
      codespan({ text }) {
        if (text.length <= 24 || text.startsWith("-")) return `<code class="whole">${escapeHtml(text)}</code>`;

        // Long code may break only after a path slash, a comma or a closing tag: never inside "</".
        return `<code class="long">${escapeHtml(text).replace(/((?<!&lt;)\/|, |&gt;)/g, "$1<wbr>")}</code>`;
      },
      // An item that opens with a bold lead word ("**Input:** ...") becomes a
      // label cell and a text cell; the label stays in view beside long text.
      listitem(item) {
        const first = item.tokens[0];
        const inline = first && (first.type === "text" || first.type === "paragraph") && "tokens" in first ? first.tokens : undefined;
        const lead = inline?.[0];

        if (!inline || !isStrong(lead)) return false;

        const label = this.parser.parseInline(lead.tokens).replace(/:\s*$/, "");

        // Without its label's colon the text reads as its own sentence: a
        // lowercase first word becomes capitalized. Never code, and never a
        // term with digits or hyphens ("x86-64").
        const text = (this.parser.parseInline(inline.slice(1)).replace(/^\s+/, "") + this.parser.parse(item.tokens.slice(1))).replace(
          /^[a-z](?=[a-z]*[\s,.;:])/,
          (c) => c.toUpperCase(),
        );

        return `<li class="term"><strong>${label}</strong><div>${text}</div></li>\n`;
      },
      link({ href, tokens }) {
        const url = rewrite(href, from);
        const external = url.startsWith("http") ? ' rel="noopener"' : "";

        return `<a href="${url}"${external}>${this.parser.parseInline(tokens)}</a>`;
      },
      // Each cell carries its column's name, so a phone can show rows as blocks.
      table(token) {
        const names = token.header.map((c) => c.text.replace(/[`"<>]/g, ""));
        const align = (a: string | null) => (a ? ` class="${a}"` : "");
        const head = token.header.map((c) => `<th scope="col"${align(c.align)}>${this.parser.parseInline(c.tokens)}</th>`);

        const cell = (c: Tokens.TableCell, k: number) =>
          `<td data-label="${names[k]}"${align(c.align)}>${this.parser.parseInline(c.tokens)}</td>`;

        const rows = token.rows.map((row) => `<tr>${row.map(cell).join("")}</tr>`);
        const wide = names.length > 2 ? " wide" : "";

        return `<div class="table${wide}"><table><thead><tr>${head.join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>\n`;
      },
    },
  });

  const tokens = marked.lexer(md);

  await highlight(tokens);

  return marked.parser(tokens);
}
