# Critique: Docs page

One critic for the whole job (Sonnet, high effort), briefed as in the
web-designer skill's `critique.md`. Renders: `shoot.mjs` at 390, 768, 1280 and
1440, light and dark, `matrix.mjs`, and the head drawing's filmstrip.

## Round 1

| # | Line | Score |
|---|---|---|
| 1 | Concept on the page | 4 |
| 2 | Not the category average | 4 |
| 3 | Not a slop face | 4 |
| 4 | First screen | 4 |
| 5 | Typography | 4 |
| 6 | Colour | 4 |
| 7 | Richness | 3 |
| 8 | Structure and rhythm | 3 |
| 9 | Craft | 3 |
| 10 | Responsive | 4 |
| 11 | Everyone's computer | 4 |
| 12 | Signature | 4 |
| 13 | Screenshot test | 3 |

Not done: 9 of 13 lines at 4 or above (10 needed), none below 3.

Asks and what was done:

1. Craft: a gap after inline code, flags split at the hyphen, long code lines
   clipped with no cue, the head command splitting at 320px. Done: tighter
   code chips; short code and every `--flag` never wrap; overflowing code
   blocks fade at the right edge until scrolled to the end, with a thin
   scrollbar; the head command scrolls instead of wrapping.
   Declined: pull the drawing's cable inside the content edge. The drawing
   running off the right edge is the page's one deliberate grid break.
2. Structure: split the page into "Use it" and "Reference", larger h2.
   Done, at site level: two parts with a full-ink rule and a label between
   them, two groups in the index, h2 at 36px. Declined for now: h3 groups
   inside "Behavior and limits". That changes `trit/README.md`, which GitHub
   also shows; it is the owner's decision.
3. Richness: a closing drawing. Done: Xiaohei asleep by an unplugged cable,
   from the README's "an idle engine uses no CPU"; he breathes and the "z"s
   float while it is on screen.
4. Navigation: the index marked a section at the top of the page; phones had
   no way back to the index. Done: a stricter band for the current section;
   on phones the "On this page" menu sticks to the top and names the
   current section.
5. Phone scanning: stacked tables too tall, drawing labels small. Done: on
   phones, the cells after the first run on one line with their names;
   handwriting in drawings at 12px or more. Also: the "DOCS" eyebrow now
   names the version, and the lede says what Trit is.

## Round 2

| # | Line | Round 1 | Round 2 |
|---|---|---|---|
| 1 | Concept on the page | 4 | 4 |
| 2 | Not the category average | 4 | 4 |
| 3 | Not a slop face | 4 | 4 |
| 4 | First screen | 4 | 4 |
| 5 | Typography | 4 | 4 |
| 6 | Colour | 4 | 4 |
| 7 | Richness | 3 | 4 |
| 8 | Structure and rhythm | 3 | 4 |
| 9 | Craft | 3 | 3 |
| 10 | Responsive | 4 | 4 |
| 11 | Everyone's computer | 4 | 4 |
| 12 | Signature | 4 | 4 |
| 13 | Screenshot test | 3 | 3 |

Done by the stop rule: 11 of 13 lines at 4 or above, none below 3, no FAILs
in the scan or the matrix. Asks 2, 3 and 5 landed; 1 and 4 mostly. The
critic accepted the three declines.

### Fixed after the last round, not rescored

Verified with the scan, the matrix and by eye; no FAILs.

1. The index and the phone bar showed a current section at the top of the
   page. Above the first section nothing is current now.
2. The lede lost the space before its link; code chips left a gap before
   punctuation. Fixed; chips are tighter (0.05em 0.18em).
3. Long inline code broke mid-token on phones. Long code now breaks only
   after a slash, a comma or a closing tag.
4. The closing drawing: less space above it, wider, and its caption is
   balanced (no one-word last line).
5. "Behavior and limits" read as one wall of bullets. At site level, without
   changing the README, a list whose items open with a bold lead word shows
   the lead word in its own column on a hairline (also on Benchmarks).

## Round 3

Average 4.0. Done by the stop rule (12 of 13 at 4 or above). The owner asked
for 4.5 or more, so the job continued.

| # | Line | R1 | R2 | R3 | R4 |
|---|---|---|---|---|---|
| 1 | Concept on the page | 4 | 4 | 4 | 4 |
| 2 | Not the category average | 4 | 4 | 4 | 4 |
| 3 | Not a slop face | 4 | 4 | 4 | 4 |
| 4 | First screen | 4 | 4 | 4 | 5 |
| 5 | Typography | 4 | 4 | 4 | 4 |
| 6 | Colour | 4 | 4 | 4 | 5 |
| 7 | Richness | 3 | 4 | 4 | 4 |
| 8 | Structure and rhythm | 3 | 4 | 4 | 4 |
| 9 | Craft | 3 | 3 | 4 | 4 |
| 10 | Responsive | 4 | 4 | 4 | 4 |
| 11 | Everyone's computer | 4 | 4 | 5 | 5 |
| 12 | Signature | 4 | 4 | 4 | 4 |
| 13 | Screenshot test | 3 | 3 | 3 | 4 |
| | Average | 3.7 | 3.8 | 4.0 | 4.2 |

Round 3 asks, done for round 4: the head drawing larger, its floor on the
last rule of the words column (every inner page); long code breaks only
after a path slash, ", " or ">"; full-width hairlines on lead-word lists;
the lead word in a sticky label cell (renderer, README text unchanged); both
parts open with an ink rule and a 20px name; stronger breathing and z
drift; less space before the footer. Declined: "Whisper `.en`" keeps the
README's own space.

## Round 4

Average 4.2. Done by the stop rule (13 of 13 at 4 or above, three at 5).
The budget of four rounds is spent; 4.5 was not reached.

### Fixed after the last round, not rescored

Verified with the scan, the matrix, computed styles and by eye.

1. Lead-word text rendered at 20px: the list item's class `lead` collided
   with the hero's `.lead`. Renamed `term`; the text is 17px (16px on
   phones) like the rest, the label 15px/600.
2. Text after a dropped colon read as a fragment ("windows cut at…"). A
   plain lowercase first word is capitalized; code and terms with digits
   ("x86-64") are left alone.
3. The part name sat far above its first heading: now 16px.
4. Code in table cells: 0.95em of the cell.
5. The sleeper's breathing was invisible: now on the torso only, at 4–6%,
   feet fixed.
6. The band above the closing drawing: 32px; the drawing keeps its own air.

What stands between this page and 4.5, in the critic's words: h3 levels
inside the README's long sections (the owner's call, since it changes
trit/README.md), and the ten screens of reference between the two drawings.
