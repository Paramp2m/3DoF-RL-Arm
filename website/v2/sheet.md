# DIRECTION SHEET — The name as a monument
seed 1909920412 · drawn 2026-10-22 · sector studio-portfolio · tone editorial
Brief: 3-DOF robot arm learns reach, hold, track, avoid and push with reinforcement learning in MuJoCo; honest ablations

| | |
| --- | --- |
| Direction | The name as a monument |
| Hero | Stacked name, two inks |
| Type | Chivo |
| Colour | Warm black, gold |
| Sections | intro (Narrow column, empty field) → work (Names as one paragraph) → stats (Quiet figure lines) → showcase (Band of unequal panels) → about (Side facts, wide paragraph) → list (One item per poster line) → cta (One word is the button) → footer (Columns above, wordmark below) |
| Motion | Masked headline |
| Photographs | the "Hands and making" set, as atmosphere |

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
The name is the headline. Set it so large that it becomes architecture: as wide as the viewport, cropped by an edge, or stacked to fill the screen. Nothing else on the first screen competes; the description of what this is drops to a single small sentence. The idea comes from treating the name's letters as material. Ask what the subject can do to its own name: a photograph of the work sits inside a counter or replaces a letter, the name repeats down the page like a reel, the letters are split to opposite corners and the page happens between them. In the study this was the single most common device among sites scoring 7.5 or more.

Before anything else, write one sentence in your notes: "On this page, <the subject's own material> becomes <the device>."
Fill the first blank from your inventory of what only this client has (its name and what it means, its street, its numbers, its
constraint), then put the nearest competitor in the sentence: if it still works, the idea is the category's, not the client's.
Then name the one live thing the visitor can do on the page (SKILL.md step 4).

## 2. Composition
First screen: the wordmark spans 88–100% of the viewport width (font-size 16–24vw, set with a clamp so it never wraps), anchored either to the bottom edge with its descenders nearly touching it, or dead centre with everything else pushed to the corners. One line of supporting text at 16–20px sits in a corner on a shared margin; navigation is three or four words in the opposite corner. After the hero the page is deliberately small and calm: a narrow text column or a simple two-column grid, so the contrast with the monument holds. The name returns once more at full width as the footer, in the inverse colours. Margins are tight (20–32px) so the type reaches the edges. Two type sizes do nearly all the work: the monument and a small text size, with one mid size for section headings.

## 3. Type — Chivo
A black grotesque in capitals with letters nearly colliding, and only two sizes on the page: enormous and small. Suits agencies, city and culture projects and musicians who want the type to land like a headline.
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Chivo:wght@400;700;900&display=swap" rel="stylesheet">
```
- Display: **Chivo** — weight 900 · tracking -0.05em · leading 0.88 · uppercase
- Text: **Chivo** — 400 for paragraphs, 700 for navigation, tags and links · leading 1.35 · tracking -0.01em
```css
:root {
  --step-small: clamp(0.8125rem, 0.79rem + 0.1vw, 0.875rem);
  --step-body: clamp(1.0625rem, 1rem + 0.25vw, 1.1875rem);
  --step-lead: clamp(1.25rem, 1.1rem + 0.7vw, 1.625rem);
  --step-head: clamp(2.5rem, 1.2rem + 5vw, 5.5rem);
}
```
One family. The display cut must survive enormous sizes: tight tracking (-0.04 to -0.07em), leading 0.8–0.9, optical kerning on. Section headings are modest (28–40px) on purpose.
The direction decides sizes and how many of them; the type system decides how this face is set. Where the two give different tracking or leading, use the values of the type system: they are tuned to this face.
The display size itself comes from the hero blueprint below.

## 4. Colour — Warm black, gold
```css
:root {
  --bg: #0c0a08;
  --surface: #181512;
  --ink: #ffffff;
  --muted: #8f8a84;
  --accent: #c9a45a;
  --accent-ink: #0c0a08;
}
```
Warm near-black fills the page and photographs sit bright against it, with pure white small type; panels warm to a brown-black. Gold is for one text link or small button per screen, never a field.
Two values carry the page (ground and ink). The accent, if used at all, appears on one thing only: the single action or the inset image frame. The footer inverts ground and ink.

## 5. Imagery
**The project has no photographs of its own yet.** First choice: **real pictures of the subject's world from the open collections**: its city and street in old postcards and photographs, an old map of the neighbourhood, its materials up close, its trade in engravings, museum objects of its kind (references/imagery.md §3). Start with `node ~/.claude/skills/design-taste-frontend/scripts/assets.mjs "mujoco postcard"`, then "mujoco old map" and "learns reach", then queries from your inventory (its street, its materials, its trade's tools and history). Download what you use into the project with `--get` and credit every one on the page (`--credits` prints the footer line). They show the world around the client, never the client's own room, dish, product, person or team. Second, for what no honest picture exists of: draw the subject's own things in code, large, in one flat graphic language and the palette's real colours (its tiles, its windows, its objects; references/imagery.md §8). Third: the dealt photographs, only where an honestly generic picture fits (the sea, the weather, a material), never one a visitor could take for the thing itself and never an unrelated scene. One treatment for all of them. If this direction is built on photographs, plan the set before building (four to six pictures of one family) and reuse them at different crops and scales instead of finding or drawing a new picture for every slot; keep drawings as `<symbol>` + `<use>` or generate them with a small script so the page stays light. List every found picture (with its source and licence), every dealt photograph and every drawing that stands in for photography in your report.

Few images, used as inserts in the typography rather than as illustrations beside it: one photograph inside or behind the name (a 4:5 or 1:1 frame replacing a letter, or tucked between two words), then a short run of large full-width or half-width photographs further down. Square corners. No collage of small thumbnails.

**Floor (the inspector checks it):** at least four real pictures on the page, usually six to twelve; SVG drawings do not count. One moving picture where an honest clip exists: `node <skill>/scripts/assets.mjs --video "<query>"`, used as a muted loop with a poster frame. Whatever this direction says about pictures, it is carried out with real ones.

Photographs dealt for this build, from the "Hands and making" set (hands at work, craft and making)
Order of preference for imagery: (1) the project's own photographs and product shots; (2) images you generate, if an image tool is
available; (3) real pictures of the subject's world found in the open collections (assets.mjs, downloaded and credited on the page);
(4) drawings of the subject in code; (5) these. They are real photographs (Unsplash licence, served by picsum.photos) of places, light and
texture. They are atmosphere, never a stand-in for a specific product, dish, person or interface. List the ones you use in your report so
they can be replaced.
- `https://picsum.photos/id/785/1600/1000` — Two cupped hands holding a pile of green spruce tips · mid tone · fresh green, dark green, warm skin · native landscape · photo: Jean Lakosnyk
- `https://picsum.photos/id/319/1600/1000` — Hands holding a vintage rangefinder camera against a denim jacket · mid tone · denim blue, mustard, silver · native landscape · photo: Kristian Karlsson
- `https://picsum.photos/id/491/1600/1000` — Axe, hammer, pliers and leather gloves laid out on dark wood · dark tone · dark brown, tan, steel grey · native landscape · photo: Todd Quackenbush
- `https://picsum.photos/id/304/1600/1000` — Hand adjusting faders on an audio mixing console · mid tone · warm white, navy, amber · native landscape · photo: Drew Patrick Miller
- `https://picsum.photos/id/1060/1600/1000` — Kettle pouring water over a pour-over coffee dripper on a dark counter · dark tone · dark teal, black, warm brown · native landscape · photo: Karl Fredrickson
- `https://picsum.photos/id/674/1600/1000` — Cupped hands holding a heap of dark blue grapes · mid tone · pale blue, dark purple, tan · native landscape · photo: Maja Petric
- `https://picsum.photos/id/407/1600/1000` — Hands lighting a sparkler that throws bright sparks in the dark · dark tone · charcoal, warm white, amber · calm area: right · native landscape · photo: Kelley Bozarth
The last two path numbers are width and height: request the exact crop you render (`/1200/1500` for a 4:5 portrait, `/2400/1350` for a hero) and
add `?grayscale` to desaturate. Use three to six of them, not all eight. Always set `width`, `height`, `alt` and `loading="lazy"` (not on the hero image).
Other sets that fit this brief: fog-and-quiet, architecture, texture, people — `node draw.mjs --photos <set>` prints one in full; `--photos list` shows all sets. Taking photographs from another set is your call and is not a redraw; keep them to one light and one treatment.

## 6. Hero — Stacked name, two inks
One saturated flat field covers the whole 1440×900 viewport and there is no image. The name is broken into two lines and set flush left from a 40px margin at clamp(110px, 15vw, 230px), heavy weight, tracking -0.04em, line-height 0.82 so the lines almost touch. The block starts 14% down and fills the top 58% of the screen; the longer line reaches at least column 11. Line one is set in the light ink and line two in the dark ink, so the break reads as a change of voice: split a two-word name between the lines, or break one long name with a hyphen at a syllable. The top edge carries only small text at 15px, 24px in: a monogram top-left, and two or three text links plus the single action as an underlined link top-right. The supporting sentence, 18px in the light ink and max 30ch, is right-aligned in the bottom-right corner on columns 9–12, its last line 40px above the bottom edge. The bottom-left corner stays empty. The next section begins as a different flat field exactly at the fold.

On a phone: The field fills 100svh. The name keeps its two lines and two inks at 21vw, flush left at a 16px margin, starting 18% down; a long name hyphenates rather than dropping below 18vw. The supporting sentence moves to the bottom-left at 16px across the full width, with the action as an underlined link beneath it, 24px above the bottom edge. The top-right links reduce to one word.
Headline: The name on exactly two lines, one per ink, each line up to about 8 characters at 15vw on desktop and 21vw on a phone.

Adjusted for this page:
- Its action is a text link: the primary action on the first screen is a real button (filled or solid-bordered, 44px or taller), placed where the blueprint puts the link. A second action may stay a link.

## 7. Sections, in order
Each is a composition to rebuild with this project's content. Keep the geometry; change the subject. Re-order or drop a
section when the argument of the page needs it (see references/copy.md), never to fall back to a card grid.

### 1. intro — Narrow column, empty field (stack)
A single text column 300–340px wide, its left edge at 15% of the viewport width, starting 64px below the section top. It holds four or five short paragraphs at 15–16px with leading 1.5 and one blank line between them; the first paragraph is in medium weight and full ink, the rest in regular weight and the soft ink. No heading, no image, no button; one underlined text link may end the last paragraph. The remaining three-quarters of the 1440 screen is untouched ground, and the section is only as tall as the text plus 160px of padding below. The column's left edge must match a line used elsewhere on the page (the wordmark's left edge or a later column). Where the page has a large object or a fixed photograph, this column is the small text beside it. Holds 90–150 words comfortably.
On a phone: The column becomes 78% of the width, left-aligned at the 20px margin, so a strip of empty ground remains on the right as a deliberate margin. Text stays at 15–16px with the first paragraph in medium weight. Padding is 72px above and 96px below.

### 2. work — Names as one paragraph (statement)
All project or client names run together as one paragraph at poster scale: 10–18 names at clamp(40px, 5.2vw, 78px), leading 1.1, flowing inline across the full width between 24px margins and wrapping where they fall, five to seven lines in all. Names are separated by a comma, or by a small inline frame: an `inline-block` 0.8em tall and 1.2em wide, aligned to the cap height, holding that project's photograph and placed directly after its name. Every name is a link; all names rest in the soft ink, and the hovered one takes full ink while its inline frame scales to 1.6 times from its left edge. A heading of two words opens the paragraph as its first words in full ink, on the same line and at the same size, followed by a colon. Below the paragraph at the left margin, one line at 15px gives the count and the years covered.
On a phone: The paragraph is set at 9vw, left-aligned, about twelve lines, with names separated by commas. Inline frames are removed except after the first three names, where they stay at cap height. All names are full ink; each remains a link with 6px of extra line spacing for touch.

Adjusted for this page:
- It sets a sentence large over several lines: jurors read that as a long headline. Either cut it to a two-line headline at display size, or keep it a statement at 36px or less on three lines (24px, four lines on a phone).

### 3. stats — Quiet figure lines (list)
Numbers stated without raising the voice. In columns 2–7, a definition-list layout: three to five rows, each a two-track grid. The first track is 132px wide and holds the figure, right-aligned, at 28–32px in medium weight and full ink with tabular numerals. The second track, after a 24px gap, holds one sentence at 16–17px in the soft ink, at most 40ch, whose first baseline matches the figure's baseline. Sentences share one left edge and figures one right edge, so a clean channel of space runs down between them. 28px between rows. The heading, two to four words at 24–28px, sits 56px above on the sentences' left edge, not the figures'. Optionally, a single small square photograph occupies columns 9–11, its top 80px lower than the first row, with no caption. Leave at least 18vh of empty ground above and below the block.
On a phone: The grid stays two-track, with the figure track narrowed to 88px and figures at 24px; sentences at 16px wrap beside them. If the photograph is used it comes after the rows at 56% width, flush right. The heading is 22px on the sentences' left edge.

### 4. showcase — Band of unequal panels (strip)
A modest heading, then one wide band of pictures. The heading, two to four words at 36–44px, starts at column 3, left of centre, with 64px beneath it. The band spans the full viewport width edge to edge and is 68svh tall: four photographs standing side by side with no gaps and deliberately unequal widths, a flex row with flex-grow values of 5, 2, 3 and 2. Each panel is cropped with object-fit: cover to a tall slice, so the band reads as one folded view. The first panel carries a two-line title at 28px in paper colour at its top-left, 24px in, over a neutral scrim that covers only that corner. Hovering a panel raises its flex-grow to 6 over 500ms while the others give way; without hover the widths simply stay. One caption line at 16px sits under the band at the left margin.
On a phone: The heading sits on the margin at 28px. The band becomes a two-by-two block with no gaps, 84svh tall in total: the first row split 64/36 and the second 36/64, so the slices stay unequal. The title stays on the first panel. No hover behaviour.

### 5. about — Side facts, wide paragraph (table)
Two rows share one vertical line at 34% of the viewport width. Row one: in a 160px column at the left margin, three label and value pairs stack with 20px between them, each a label in the soft ink with its value beneath in full ink, all at 13px (founded, based in, team size). From the 34% line a five-line paragraph at clamp(20px, 1.7vw, 26px), leading 1.35, runs to 90% of the width. Row two starts 120px lower and repeats the logic for the record of the practice: one plain word at 13px at the far left (awards, press, clients), then from the 34% line two list columns, each 260px wide with a 64px gutter, at 13px with 10px between lines: name at the left, year at the right in the soft ink. No heading; the paragraph opens the section. Each list comfortably holds 6–12 lines. No photographs are needed.
On a phone: Everything aligns to the left margin. The paragraph comes first at 19px. The three label and value pairs follow as a two-column grid at 14px. The two record lists stack with 40px between them, each opened by its plain word in full ink.

### 6. list — One item per poster line (statement)
A short list set as a poster. Four to seven items (services, headline acts, courses, opening days) stack one per line at clamp(64px, 10vw, 150px) in the display weight, all capitals or all lowercase, leading 0.92, flush left at a 20px margin on a flat field. Its one detail (a price, a date, a time range) sits at the right margin on the same baseline at 18–20px with tabular figures, so the page has two alignments only: giant words left, small facts right. Items too long for one line at this size are shortened, not shrunk. Between items there is only the tight leading. Hovering an item swaps its ink for the accent. There is no heading: the first item starts 80px below the section top and the last ends 80px above the bottom. The list reuses the hero title's exact type size, so it reads as another poster.
On a phone: Items at 15vw, one per line, flush left with 16px margins. Each detail moves beneath its item at 15px, left-aligned, with 6px above and 22px below. If any single word is wider than the screen at 15vw, drop the whole list to 12vw rather than breaking a word.

Adjusted for this page:
- It names a kind of typeface: use the dealt faces (display Chivo, text Chivo) and their settings; capitals only where the type system sets them.

### 7. cta — One word is the button (band)
The entire band is one link. A full-bleed field in the accent colour, about 60svh tall, contains a single verb of four to eight letters set at 24–30vw in the heaviest weight, tight tracking, sized with `clamp()` so it spans the content width on one line with its baseline 24px above the bottom edge. The `<a>` wraps the whole band; on hover and focus the field and the word swap colours instantly. In the top-left corner, at 20px, one sentence says what happens next; in the top-right, an arrow drawn as a simple inline SVG, 48px, points up and right. There is no other button anywhere in the section. In a page with a motif, the band's top corners take the motif's shape (a large radius, or a chamfer cut with `clip-path`). Keep the word honest: the action itself, not a slogan.
On a phone: The band is 46svh tall. The word is set at whatever size fills the width on one line; choose a shorter verb rather than going below 22vw. Sentence top-left at 17px across two lines, arrow top-right at 32px. The whole band stays tappable.

### 8. footer — Columns above, wordmark below (stack)
The footer inverts ground and ink and fills a whole viewport (min-height 100svh), arranged as three rows in a grid (auto 1fr auto). Top row, 40px from the edge: four small columns on the 12-column grid (address, navigation, social, hours or contact), each a plain list at 15–16px with its first line in the full ink and the rest in the softer ink. Middle row: empty. Bottom row: the wordmark alone, sized to run from the left margin to the right margin on one line. Set it as inline SVG text with a `viewBox` fitted to the word and `width: 100%`, or tune `font-size` in `vw` to the letter count, so it never wraps; its baseline sits 20px above a single line of legal text at 13px. With `position: sticky; bottom: 0` and a lower `z-index` than the page, the last section slides away to uncover it.
On a phone: Columns become two by two at 15px with 32px between rows, at the top. A gap of about 20svh follows. The wordmark spans the full width at the bottom; a name longer than ten letters stacks on two lines, each fitted to the width. Legal line beneath at 12px.

Adjusted for this page:
- It sets text at 12px: use your small step (13px or more) instead.

## 8. Motion — Masked headline
On load, each line of the headline rises from behind its own baseline, about 900ms with a strong ease-out, the second line 110ms after the first; the supporting sentence and the action then fade up together. Nothing else on the page performs. Because headlines are two lines at most, you author the line breaks and wrap each line yourself. Recipe: 'Masked line reveal' in references/motion.md.
This is the signature move, the one loud moment. Around it the page still needs the rest of the motion layer (SKILL.md step 9): an entrance for the parts of every section, feedback on every control, and one moving picture. The inspector fails a page that barely moves. Recipes: references/motion.md.

## 10. Where the richness comes from
Scale. The size of the name against the smallness of everything else is the entire visual event, so protect both ends: do not shrink the name to make room, and do not let body text creep above 20px.

## 11. How this direction goes wrong
A long name set small is just a logo. If the name has more than about 12 characters, stack it on two lines or crop it with the viewport edge; never drop below 12vw. Do not add a second big headline under the name. Do not decorate the letters.

## 12. Real references
Studied sites that work this way. Take the principle, never the specifics: no names, copy, images or signature assets of theirs.
- **House of Honey** (Awwwards Site of the Day, studied score 8.4) `aww-house-of-honey-1`
  Idea: A block-heavy wordmark pierced by a hairline script: the bold-against-delicate contrast is repeated at every scale, with giant pink script behind quiet serif statements.
  Hero: Top 42% is a flat blush-pink band; the bottom 58% is a full-bleed photograph of a bright living room, arched window on the centre axis, butting against the band with no margin. A tiny nav row sits at the very top with 8px side margins: block logo glyph and five links left, contact link with a small square arrow box far right. The wordmark fills the width edge to edge: two words in ultra-black extended caps, about 120px cap height, dark aubergine, with a hairline script 'of' 230px tall between them, overshooting above and below. Directly beneath, two two-line serif uppercase taglines at 31px, one flush left, one flush right.
  Worth taking: Set a wordmark edge to edge in ultra-heavy extended caps and thread a hairline script connecting word through it that overshoots the cap height both ways.
- **Paul Kalkbrenner** (Awwwards Site of the Day, studied score 8.1) `aww-paul-kalkbrenner`
  Idea: Swiss poster typography for a techno artist: the name as one black line with a photo set inside it like a glyph, then every chapter gets its own flat colour and visible grid.
  Hero: Pure white viewport. The artist's name sits on one line at exact vertical centre in bold black neo-grotesk, about 10.4vw, spanning roughly 78% of the width with equal side margins. The space between first and last name is replaced by a small landscape black-and-white photo, x-height tall and resting on the baseline. Top-left is a tiny circular monogram; eight nav links in 13px bold are spread at even intervals across the right half of the top edge. Bottom edge: equaliser bars with a grey label over a track title at left, a three-line bold blurb starting just left of centre, a sound toggle hard right. About 85% is empty.
  Worth taking: Replace the space between the two words of a giant name with a small photo cropped to x-height.
- **Tiny Computer Co.** (One Page Love, studied score 7.2) `opl-tiny-computer-co`
  Idea: A software studio presented as a plain text log: one serif sentence, then monospace notes with dates in a narrow left column, and nothing else.
  Hero: Near-black viewport. Everything hangs on a left margin of about 5%. A small serif wordmark sits 10% from the top; there is no navigation at all. The headline, two lines of off-white serif at 7.5vw with tight leading, starts about 27% down and spans only the left 44% of the width. One line of 18px monospace follows a full headline-line below it. The right half and the bottom 40% of the viewport are completely empty: no button, no image, no rule.
  Worth taking: Confine the whole page to a left column under half the viewport and leave the rest empty, with no nav or buttons.
Full notes on any of them: `node draw.mjs --site <id>`.

## Always
Headlines two lines or fewer at every width. No eyebrow labels, no monospace, no 01/02/03, no thin rules, no gradients, no shadows at rest,
no cream-and-terracotta, no violet. The frame: one h1 on the first screen that says what this is, equal outer margins, no
section huddled on one side with the other half empty, no blank stretch over half a screen, a heading on every section.
`inspect.mjs` measures all of it.

Saved to website/v2/.taste-skill/direction.json · reprint with `node draw.mjs --show`
