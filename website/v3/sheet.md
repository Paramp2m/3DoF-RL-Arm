# DIRECTION SHEET — Colour chapters
seed 2101867040 · drawn 2026-10-22 · sector ai-product · tone bold
Brief: 3-DOF robot arm learns reach, hold, track, avoid and push with reinforcement learning in MuJoCo; honest ablations

| | |
| --- | --- |
| Direction | Colour chapters |
| Hero | Waterline headline |
| Type | Hanken Grotesk |
| Colour | Ultramarine and yellow |
| Sections | intro (Statement tiles) → work (Stacked folders with tabs) → stats (Bars standing on the edge) → showcase (Headline you can type) → about (Inset panel, shaped portrait) → list (Cards that fold into spines) → cta (A sentence with a blank) → footer (The mark, built large) |
| Motion | Masked headline |
| Photographs | the "Water" set, as atmosphere |

This is the direction for this build. Adopt it. Do not blend it with the layout you would have made anyway,
and do not redraw because it is unfamiliar: unfamiliar is the point. Redraw a slot only when it conflicts
with a stated brand constraint or needs an asset that does not exist, and say why in your report.

The parts of this sheet were written separately and will disagree in small ways. Settle every disagreement in this order:
**house rules** (including type no smaller than 13px and body text of 16px or more) → **the direction** (the idea, the composition habits, where the
one accent goes) → **the type system and the palette** (which face does which job, the five size steps, which colour is the ground and where the
accent goes) → **the hero blueprint** → **the section blueprints**. Under each blueprint below, "Adjusted for this page" lists where it disagrees.
In practice: take geometry from blueprints (columns, proportions, what sits where) and take everything else from above them. Map every size a
blueprint names onto one of your five type steps instead of adding a sixth. Use one button shape, one corner radius and one pair of page margins on
the whole page, whatever individual blueprints say. If a blueprint names a kind of typeface (a light serif, a heavy grotesk), use the dealt face.

## 1. The idea
Every section is a chapter with its own flat colour, floor to ceiling. The page reads like a set of posters hung in a row, held together by one typeface and one grid. The colours are not decoration: they come from the subject (the brand's packaging, the artist's record sleeves, the product's own colour ways, the categories of the content) and each chapter's colour means that chapter. Decide the sequence like a designer orders a book: a strong opener, a quiet second, a surprise in the middle, a dark or very light closer.

Before anything else, write one sentence in your notes: "On this page, <the subject's own material> becomes <the device>."
Fill the first blank from your inventory of what only this client has (its name and what it means, its street, its numbers, its
constraint), then put the nearest competitor in the sentence: if it still works, the idea is the category's, not the client's.
Then name the one live thing the visitor can do on the page (SKILL.md step 4).

## 2. Composition
Sections are full-bleed fields, each at least 90svh tall, butted directly against each other with no gap and no divider. Inside every field the same 12-column grid and the same margins apply, and each field holds one thing: a huge heading (8–12vw) with a small paragraph, or one photograph with a caption, or one list. Text colour is chosen per field for strong contrast (ink or paper, never grey). Headings start on the same left edge in every chapter, so the eye feels the system as colours change. The navigation and the footer take the colours of the first and last chapter.

## 3. Type — Hanken Grotesk
Medium-weight grotesk statements at one large size against caption-size text: a two-size system with no bold. Suits portfolios, agencies and tools that speak in short declarative lines.
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500&display=swap" rel="stylesheet">
```
- Display: **Hanken Grotesk** — weight 500 · tracking -0.05em · leading 0.9 · sentence case; one statement size (about 6vw) used for every heading
- Text: **Hanken Grotesk** — 400 for body, 500 for emphasis · leading 1.45 · tracking -0.005em
```css
:root {
  --step-small: clamp(0.8125rem, 0.79rem + 0.1vw, 0.875rem);
  --step-body: clamp(1.0625rem, 1rem + 0.25vw, 1.1875rem);
  --step-lead: clamp(1.25rem, 1.1rem + 0.7vw, 1.625rem);
  --step-head: clamp(2rem, 1.3rem + 3.2vw, 4rem);
}
```
One typeface in two weights. Headings very large and tight; body 18–20px. Because colour is loud, the type stays plain: no italics, no second face.
The direction decides sizes and how many of them; the type system decides how this face is set. Where the two give different tracking or leading, use the values of the type system: they are tuned to this face.
The display size itself comes from the hero blueprint below.

## 4. Colour — Ultramarine and yellow
```css
:root {
  --bg: #1423b8;
  --surface: #1220a6;
  --ink: #ffffff;
  --muted: #b4bcf0;
  --accent: #1fc27a;
  --accent-ink: #0a0a0a;
  --field-1: #1423b8;  --field-1-ink: #ffffff;
  --field-2: #000000;  --field-2-ink: #ffffff;
  --field-3: #f7dd47;  --field-3-ink: #000000;
  --field-4: #e9479f;  --field-4-ink: #000000;
}
```
Each section is one flat field in sequence: ultramarine, black, yellow, then ultramarine again, with white type on blue and black, and black type on yellow. Green is the only small accent and never grows larger than a button.
Four to six flat fields used at full strength, one per section, each with its assigned text colour. No tints, no gradients between them: the hard edge where two fields meet is the look.

## 5. Imagery
**This subject is software and there are no screenshots, so its pictures are its own interface.** Wherever this sheet says photograph, image or object, build a believable fragment of the product in HTML at real size (real labels, numbers that add up, one item hovered or selected) by the app-screen rules (references/app-screens.md: deal its style once with `node <skill>/scripts/draw.mjs --app --kind <kind> --keep palette,type --out .taste-skill/app`, and build it with assets/app.css and assets/charts.js), put the attribute `data-ui` on its root, and crop or scale it the way the blueprint treats a photograph: a full-bleed close crop, a tile, the fixed half of a split. Where a fragment makes no sense, use a flat colour field with type on it. Never a fake dashboard of grey bars, never a device frame. If the product works on something physical (ships, plants, buildings, a city), one real picture of that world can carry a section: find it with `node ~/.claude/skills/design-taste-frontend/scripts/assets.mjs "<that world>"`, download it with `--get`, credit it on the page, and never let it pass for the product or its customers.

Sparing. One photograph or object per chapter at most, either cut out and sitting directly on the colour, or a plain rectangle with generous colour around it. Photographs should be simple enough not to fight the field.

**Floor (the inspector checks it):** at least four real pictures on the page, usually six to twelve; SVG drawings do not count. One moving picture where an honest clip exists: `node <skill>/scripts/assets.mjs --video "<query>"`, used as a muted loop with a poster frame. Whatever this direction says about pictures, it is carried out with real ones.

Optional atmosphere, two at most, from the "Water" set (lakes, rivers, waterfalls and water surfaces). The page does not need them.
Order of preference for imagery: (1) the project's own photographs and product shots; (2) images you generate, if an image tool is
available; (3) real pictures of the subject's world found in the open collections (assets.mjs, downloaded and credited on the page);
(4) drawings of the subject in code; (5) these. They are real photographs (Unsplash licence, served by picsum.photos) of places, light and
texture. They are atmosphere, never a stand-in for a specific product, dish, person or interface. List the ones you use in your report so
they can be replaced.
- `https://picsum.photos/id/1051/1600/1000` — Weathered wooden jetty reaching into a misty mountain lake · mid tone · pale blue, weathered grey, slate · calm area: top · native landscape · photo: Ales Krivec
- `https://picsum.photos/id/124/1600/1000` — Small red boat floating on clear turquoise water, seen from above · mid tone · turquoise, teal, red · calm area: right · native landscape · photo: Anton Sulsky
- `https://picsum.photos/id/1038/1600/1000` — Small red boat on a wide expanse of dark blue rippling sea · dark tone · deep blue, slate, red · calm area: bottom · native portrait · photo: Marcelo Quinan
- `https://picsum.photos/id/469/1600/1000` — Two people paddling a canoe across a misty green lake · mid tone · emerald green, pale grey, teal · calm area: bottom · native landscape · photo: Matthew Clark
The last two path numbers are width and height: request the exact crop you render (`/1200/1500` for a 4:5 portrait, `/2400/1350` for a hero) and
add `?grayscale` to desaturate. Use at most two. Always set `width`, `height`, `alt` and `loading="lazy"` (not on the hero image).
Other sets that fit this brief: fog-and-quiet, architecture, sky-and-space — `node draw.mjs --photos <set>` prints one in full; `--photos list` shows all sets. Taking photographs from another set is your call and is not a redraw; keep them to one light and one treatment.

## 6. Hero — Waterline headline
Two flat fields fill the viewport and meet at exactly half height: the upper one paper or white, the lower one a saturated palette colour. No image. The headline is two lines of capitals in a black-weight grotesk at clamp(72px, 9vw, 140px), centred, leading 0.86, about 65% of the viewport wide, positioned so the seam cuts through the middle of the second line. Above the seam the letters take the accent ink; below it they switch to a deep ink, as if under water. Build it with two stacked copies of the headline in the same position (the lower copy aria-hidden), each clipped to its own field with clip-path: inset(), or with mix-blend-mode: difference on a single copy if the colours allow. The wordmark sits top-left and the single action, a solid rectangle in the accent ink, top-right, both on 28px margins; there are no other links. Near the bottom, 40px above the edge, one supporting sentence at 20px is centred in the paper colour on the saturated field. Everything else is empty colour.

On a phone: The same two fields, meeting at 46% of the height. The headline sets at 15.5vw on two lines, filling the width inside 16px margins, the seam still cutting the second line. The wordmark sits top-left and the action top-right as a small solid rectangle. The supporting sentence sits bottom-left at 17px on the saturated field, on up to three lines.
Headline: A blunt statement of 2-4 words in capitals, at most 20 characters, split so each line is one or two words.

Adjusted for this page:
- It names a kind of typeface: use the dealt faces (display Hanken Grotesk, text Hanken Grotesk) and their settings; capitals only where the type system sets them.
- It names colours (white, black): map them onto this palette's roles (ground `--bg`, panels `--surface` and the fields, text `--ink`/`--muted`, the accent `--accent` on the primary action only).
- It blends text with `difference`: over anything but black, white or grey that gives random hues. Use a solid ink chosen for each ground instead.

## 7. Sections, in order
Each is a composition to rebuild with this project's content. Keep the geometry; change the subject. Re-order or drop a
section when the argument of the page needs it (see references/copy.md), never to fall back to a card grid.

### 1. intro — Statement tiles (mosaic)
A board of filled tiles with 8px gaps and 8px outer margins: 12 columns by two rows of 38svh each. Tile A, columns 1–8 across both rows, is the darkest tile and carries a slogan of two to four words in capitals at clamp(64px, 9vw, 132px), leading 0.86, on two lines, set hard against the tile's 20px left padding and bottom-aligned. Tile B, columns 9–12 of row one, is a pale tile with one word or figure at the same size, bottom-aligned, cropped by `overflow: hidden` if it runs long. Tile C, columns 9–12 of row two, is a photograph covering the tile. Below the board a fourth tile spans all 12 columns in the page ground colour: its left half is empty and a text of 90–130 words at 22–24px with leading 1.35 runs in columns 7–12 with 40px padding. No separate heading: the slogan in tile A is the heading.
On a phone: A two-column board with 6px gaps. The slogan tile spans both columns at 46svh with type at 17vw on two lines; the word tile and the photograph tile sit side by side as squares; the long text tile spans both columns with 20px padding at 18px.

Adjusted for this page:
- It names a kind of typeface: use the dealt faces (display Hanken Grotesk, text Hanken Grotesk) and their settings; capitals only where the type system sets them.

### 2. work — Stacked folders with tabs (band)
Projects are filed as folders: four to six full-width flat colour bands stacked so that each shows a 64px strip above the next, built with `position: sticky` and increasing `top` offsets (0, 64, 128px) so the bands pile up as the page scrolls. From the top-left edge of each band one or two tabs rise: filled shapes 44px tall and 200–320px wide with shouldered corners cut by `clip-path: polygon()` or an inline SVG path, side by side with 6px gaps, each in a flat colour that contrasts with its band. The project name sits inside the tab at 26–28px; the category sits right-aligned in the band's visible strip at 15px. An open band is 80svh: a 16:10 photograph in columns 1–7 and, top-aligned in columns 9–12, a 60-word description with a link. A two-word heading at 32px precedes the stack.
On a phone: Bands do not stick; they stack in normal flow, each led by one tab 40px tall with the name at 20px and the category beside it. Inside a band the photograph is full width at 4:3 with the description beneath at 16px and 20px padding.

### 3. stats — Bars standing on the edge (full-bleed)
The product's own chart, drawn in HTML, is the whole section. A full-bleed field 90svh tall. Along its bottom edge stand ten to fourteen solid bars, one per true value (months, releases, cohorts), in a CSS grid spanning the full viewport width with 8px gaps and no side margins; heights are the values scaled so the tallest is 58vh. No axes, no grid, no legend; the chart's root is marked data-ui. Every bar is one tone two steps from the field except the one the page is about, which takes the accent. Three bars print their value directly above themselves at clamp(20px, 2vw, 30px), tabular. The headline figure sits top-left on the page margin at clamp(88px, 10vw, 168px), and its claim, two to five words at 28px, sits beneath it on the same margin. Hovering or focusing a bar shows its value and period above it.
On a phone: Bars run sideways: each a full-width row 28px tall growing from the left edge, 6px gaps, eight values at most. The headline figure sits above at 26vw with its claim at 22px. Three rows print their value at 17px inside the bar's right end. The accent row stays.

### 4. showcase — Headline you can type (statement)
The section's headline is an input. One line at clamp(48px, 8vw, 132px), tight tracking, flush left across columns 1–12, is a real `<input>` (root marked data-ui) with no border and no fill, prefilled with a believable two-to-five-word example of what a user gives the product: a search, a question, a date in words, a colour name. A solid caret block 0.08em wide blinks after the last character. 48px beneath, on the same left edge, the product's answer appears at true interface size, at most 560px wide: three result rows, one parsed value, or a swatch, updating as the visitor types. Bottom-left, three more examples at 17px act as plain text buttons that refill the line. The rest of the 88svh field is empty. If the text outgrows the line it scrolls inside the input; it never wraps.
On a phone: The input is set at 11vw on one line and scrolls sideways inside itself when the text runs long. The answer follows 28px below at full width and true size. The three examples become a wrapping row of filled blocks 44px tall beneath the answer; tapping one refills the line.

### 5. about — Inset panel, shaped portrait (split)
A deep-toned panel inset 20px from the page edges with a 12px radius, about 110svh tall, `overflow: hidden`. Left half: a two-line statement of four to seven words at clamp(40px, 4.4vw, 64px) in a pale ink, top-left with 56px padding; it is the heading and has nothing above it. Below it a photograph of the person or the workshop, masked with `clip-path` or an SVG `mask` to one bold shape taken from the subject (a four-lobed flower, a circle, an arch, a chamfered block), 38vw across, positioned so the panel's bottom edge crops its lowest fifth. Right side, starting at 63% of the width and 56px from the top: two text blocks 300px wide and 64px apart, each a subhead at 22px over a four-line paragraph at 15px; then one solid button in the accent colour. Between the statement and the text blocks the panel is empty. One photograph only.
On a phone: The panel is inset 10px and stacks: the statement at 8.5vw, then the two text blocks at full width, the button, and last the shaped photograph at 86vw across, centred and cropped by the panel's bottom edge by a quarter of its height.

Adjusted for this page:
- It names its own corners: use the one radius of this page (square unless the direction says otherwise) and one button shape.
- It sets a sentence large over several lines: jurors read that as a long headline. Either cut it to a two-line headline at display size, or keep it a statement at 36px or less on three lines (24px, four lines on a phone).

### 6. list — Cards that fold into spines (pinned)
For a journal, a run of notes or a list of chapters without pictures. A wrapper about 400svh tall holds a sticky stage of 100svh. The heading, two to four words at clamp(48px, 4.2vw, 64px), sits top-left on the 32px margin, 120px from the top. Under it a band 360px tall spans the content width and holds a horizontal deck of text cards, each about 620px wide, in three neighbouring flat tones of one hue taken in turn. On a card, with 32px padding: its title at 30px on up to two lines, leading 1.1; an excerpt of two lines at 16px; and along the bottom edge the date on the left and a text link to read it on the right, both at 16px. As the page scrolls, each step slides the next card in from the right edge and squeezes the earlier ones into slivers 120px wide stacked edge to edge at the left, like book spines, where only the first letters of their titles show, clipped by the sliver. Clicking a sliver brings its card back. 48px under the band, the number of entries and a link to the full list sit at 17px on the left margin, so the stage does not end on an empty floor.
On a phone: No pin and no slivers. The cards stack at full width with 12px gaps, each in its tone, the title at 24px, the excerpt at 16px, the date and link at the foot. The heading sits above them at 10vw.

Adjusted for this page:
- It pins something for three screens or more: hold a pin for two or three screens at most, give every screen under it its own content, and let nothing pass over the pinned words (the inspector checks all three).

### 7. cta — A sentence with a blank (statement)
The closing sentence has a gap in it, and the gap is the form. A sentence of five to eight words is set at about 6vw on two lines, flush left across columns 1–11; there is no separate heading. Where the reader's email (or name, or city) belongs, an `<input>` sits inline on the baseline: a filled block of the one-step tone, exactly one line-height tall and about 7em wide, with the placeholder in the same face and size in the softer ink, no border, and the page's radius or motif corner. The sentence's last word is the submit `<button>`, set in the same type on a block of the accent colour of the same height. Field and button use `em` units so they scale with the sentence. One line at 16px under the sentence, in the softer ink, says what happens after sending. Nothing else is in the section.
On a phone: The sentence is set at 10vw on two lines; the input drops to a line of its own between them at full width and one line-height tall, and the final word, as the button, ends the second line. The explaining line follows at 15px.

Adjusted for this page:
- It sets a sentence large over several lines: jurors read that as a long headline. Either cut it to a two-line headline at display size, or keep it a statement at 36px or less on three lines (24px, four lines on a phone).

### 8. footer — The mark, built large (band)
On a flat colour field, the brand closes the page as a construction rather than as set type. Across the full content width and about 34svh tall, the wordmark is built from blocks: each letter is knocked out of its own solid rectangle in a palette colour, the rectangles differ in height (60–100% of the band) and butt together with 4px gaps, bottoms on one line. Where the project has a simple geometric mark or motif, enlarge that instead, drawn as inline SVG from flat rectangles, circles and quarter-arcs with the same 4px gaps. Below, 56px lower: four columns of plain links at 15px across columns 1–8, and two addresses or contact lines in columns 9–12. A last row carries the copyright at the left and one small filled button at the right. Space and the block construction do all the separating.
On a phone: The block wordmark spans the full width at about 22svh tall; a long name breaks into two rows of blocks. Link columns go two across at 15px, contact lines beneath at full width, then copyright and the small button on one row.

## 8. Motion — Masked headline
On load, each line of the headline rises from behind its own baseline, about 900ms with a strong ease-out, the second line 110ms after the first; the supporting sentence and the action then fade up together. Nothing else on the page performs. Because headlines are two lines at most, you author the line breaks and wrap each line yourself. Recipe: 'Masked line reveal' in references/motion.md.
This is the signature move, the one loud moment. Around it the page still needs the rest of the motion layer (SKILL.md step 9): an entrance for the parts of every section, feedback on every control, and one moving picture. The inspector fails a page that barely moves. Recipes: references/motion.md.

## 10. Where the richness comes from
Colour at scale. A whole screen of one saturated colour is rich on its own, so a chapter needs few elements; it still needs its content at full size, because a coloured screen with one small line on it reads as empty.

## 11. How this direction goes wrong
Check contrast on every field with the palette's assigned ink. Do not put cards on the fields. Do not use more than six colours. Two neighbouring fields should differ clearly in lightness as well as hue.

## 12. Real references
Studied sites that work this way. Take the principle, never the specifics: no names, copy, images or signature assets of theirs.
- **Ai in Design Report 2026** (Awwwards Site of the Day, studied score 8.4) `aww-ai-in-design-report-2026`
  Idea: A research report laid out like a colour-blocked print annual: a thermal-blurred video collage on top, an oversized tight grotesk title, and flat orange, lilac, sage and black blocks carrying stats, quotes and chapters.
  Hero: The top 53% is a full-width video band: blurred, heat-map-like abstract footage in orange, lilac and sage, overlaid with four or five hard-edged rectangles of the same footage in grey or green, two tagged with tiny mono labels in white boxes. The header sits flush on top: a black logo block at far left, an orange link tab running to 40% of the width, two plain links at right. Below the band, on white: a two-line deck at about 1.4vw on a 1% left margin, and from 42% across a two-line title at 8.3vw, medium weight, tracking -0.06em. Bottom row: mono credit left, scroll cue and arrow aligned to the title column.
  Worth taking: Start most headings and text at a fixed column line about 42% across, leaving the left for small decks, credits and chapter numbers.
- **Warm & Fuzzy** (Awwwards Site of the Day, studied score 7.7) `aww-warm-fuzzy`
  Idea: A showreel-first studio site in which every section takes one flat, loud colour (blue, black, yellow) under heavy grotesk type and a chunky custom wordmark.
  Hero: The showreel plays full-bleed with a visible film-grain overlay and no scrim. A chunky hand-cut wordmark sits top left, 16px from the edges; the same top line continues with a one-sentence status link, a tiny thumbnail beside the latest project's name, and three caps links at the far right, all white over the footage. A small green caps play label floats at dead centre. The tagline is anchored bottom left, 16px from both edges, in white grotesk caps at 47px (3.3vw) on one line spanning 56% of the width. A tiny square pause control sits bottom right.
  Worth taking: Give every section one flat saturated background and change it at each section break, keeping type strictly black or white.
- **Heron AI** (Awwwards Site of the Day, studied score 7.7) `aww-heron-ai`
  Idea: An AI agent that checks architects' models for code problems, shown as a drafting sheet: elevations and floor plans on warm paper, where the product's job is acted out with flat orange flags on the drawings instead of being described with abstract tech imagery.
  Hero: Warm drafting paper inside a 7px inset frame. Between two full-width rules at y≈78 and y≈748, one contained drawing box (about 95% of the width and 72% of the height) holds a stippled, hatched concrete house elevation about 960px wide with bare trees, standing on a drawn ground line, with two tiny flat-orange squares on the facade as hotspots. Nothing else shows in the captured fold: the two-line 47px (3.2vw) uppercase bold title is docked under the lower rule at y≈775 and reveals only after the first scroll, beside a 13px intro box and an orange link. Before that a 4.5s loader draws the house as line art stroke by stroke while a spec table fills in bottom-right.
  Worth taking: Draw the pictures in the product's own medium: for a tool that works on architectural models, plans and elevations that build in stages (structure, then detail, then dimensions, then the flagged problems) instead of abstract 3D or glowing shapes.
Full notes on any of them: `node draw.mjs --site <id>`.

## Always
Headlines two lines or fewer at every width. No eyebrow labels, no monospace, no 01/02/03, no thin rules, no gradients, no shadows at rest,
no cream-and-terracotta, no violet. The frame: one h1 on the first screen that says what this is, equal outer margins, no
section huddled on one side with the other half empty, no blank stretch over half a screen, a heading on every section.
`inspect.mjs` measures all of it.

Saved to website/v3/.taste-skill/direction.json · reprint with `node draw.mjs --show`
