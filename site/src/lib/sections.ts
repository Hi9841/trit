// Markdown files of the repository, cut at their headings. No rendering
// here, so docs.ts and markdown.ts can both use it.

// The ids GitHub gives headings, so "#c-and-c" in the README still works.
export function slug(text: string) {
  return text
    .toLowerCase()
    .replace(/<[^>]+>/g, "")
    .replace(/[^\p{L}\p{N}\s-]/gu, "")
    .trim()
    .replace(/\s/g, "-");
}

export interface Section {
  title: string;
  id: string;
  body: string;
}

// Splits a file at its `## ` headings. The part before the first one has the title "".
export function sections(md: string): Section[] {
  const out: Section[] = [];
  let current: Section = { title: "", id: "", body: "" };

  for (const line of md.split(/\r?\n/)) {
    const h2 = /^## (.+)$/.exec(line);

    if (h2) {
      out.push(current);
      current = { title: h2[1].trim(), id: slug(h2[1]), body: "" };
    } else {
      current.body += `${line}\n`;
    }
  }

  out.push(current);

  return out;
}

/** The ids of the `###` and deeper headings of a section's body. */
export const subheadings = (body: string) => [...body.matchAll(/^#{3,6} (.+)$/gm)].map((m) => slug(m[1]!));
