// The docs: README.md cut at its `## ` headings, one page a section. The
// landing page (/docs/) has the part before the first heading and "Start here".
import { links, page } from "./links";
import { readRepo } from "./repo";
import { sections, subheadings, type Section } from "./sections";

export const README = "README.md";

export const landingHref = `${links.docs}/`;

// Sections the site shows elsewhere: the docs index replaces Contents, and
// the Benchmarks page replaces Performance.
const ELSEWHERE = new Map([
  ["contents", landingHref],
  ["performance", links.benchmarks],
]);

const ON_LANDING = new Set(["start-here"]);

// The facts to look up, after the pages that get Trit running.
const REFERENCE = new Set([
  "languages",
  "transcript-cleanup",
  "behavior-and-limits",
  "precision",
  "environment-variables",
  "layout-of-the-code",
  "license",
]);

export interface DocPage extends Section {
  href: string;
}

const [intro, ...rest] = sections(readRepo(README));

/** The landing page's parts: the introduction, then "Start here". */
export const landing: Section[] = [
  { title: "What Trit is", id: "what-trit-is", body: intro!.body.replace(/^# .*$/m, "") },
  ...rest.filter((s) => ON_LANDING.has(s.id)),
];

const own = rest.filter((s) => !ELSEWHERE.has(s.id) && !ON_LANDING.has(s.id));
const pageOf = (s: Section): DocPage => ({ ...s, href: page(`/docs/${s.id}/`) });

/** The index of every docs page, in reading order; the landing page first. */
export const groups = [
  {
    name: "Use it",
    pages: [
      { title: landing[0]!.title, id: "", body: "", href: landingHref },
      ...own.filter((s) => !REFERENCE.has(s.id)).map(pageOf),
    ],
  },
  { name: "Reference", pages: own.filter((s) => REFERENCE.has(s.id)).map(pageOf) },
];

/** Every page in reading order, for Previous and Next. */
export const ordered: DocPage[] = groups.flatMap((g) => g.pages);

/** The section pages, without the landing page. */
export const docPages = ordered.slice(1);

// Where each heading of the README is now: its page, and its place on it.
const anchors = new Map<string, string>(ELSEWHERE);

for (const s of landing) {
  anchors.set(s.id, s === landing[0] ? landingHref : `${landingHref}#${s.id}`);

  for (const id of subheadings(s.body)) if (!anchors.has(id)) anchors.set(id, `${landingHref}#${id}`);
}

for (const p of docPages) {
  anchors.set(p.id, p.href);

  for (const id of subheadings(p.body)) if (!anchors.has(id)) anchors.set(id, `${p.href}#${id}`);
}

/** The site address of `#id`, a heading of the README. */
export const docsHref = (id: string) => anchors.get(id) ?? `${landingHref}#${id}`;

/** Every README heading and its address, for old `/docs/#id` links. */
export const anchorMap = Object.fromEntries(anchors);

/** A page's first sentences as plain text, for its description. */
export function summaryOf(body: string) {
  const prose = body
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .find((block) => /^[A-Z`*[]/.test(block) && !block.startsWith("|"));

  const text = (prose ?? "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[`*]/g, "")
    .replace(/\s+/g, " ");

  const cut = text.length > 160 ? text.slice(0, text.lastIndexOf(". ", 160) + 1) || text.slice(0, 157) + "…" : text;

  return cut;
}
