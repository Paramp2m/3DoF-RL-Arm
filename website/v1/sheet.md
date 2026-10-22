# DIRECTION SHEET — The product performs
seed 1424786875 · drawn 2026-10-22 · sector dev-tool · tone technical
Brief: 3-DOF robot arm learns reach, hold, track, avoid and push with reinforcement learning in MuJoCo; honest ablations

| | |
| --- | --- |
| Direction | The product performs |
| Hero | Controls on the sentence |
| Type | Work Sans |
| Colour | Green-black and bone |
| Sections | intro (Statement with inline thumbnails) → work (Quiet ledger, side preview) → stats (Figure set by a slider) → showcase (Output index, live sheet) → about (Founder's letter) → list (Day tabs, banded agenda) → cta (First step, already open) → footer (A footer of tiles) |
| Motion | Masked headline |
| Photographs | the "Sky and space" set, as atmosphere |

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
Software is most convincing when it is doing its job in front of you. Instead of describing features, put small, true, working fragments of the product on the page: the actual control, with real content, responding to the pointer. A colour picker's loupe reads the page's own background and prints its value. A list tool lets you tick an item right there. A type-scale plugin sets the page's own headings. Find the one interaction that is the product and make the hero that interaction; then show each further capability as another fragment at real size. No dashboard collage, no browser frame, no screenshots shrunk until unreadable.

Before anything else, write one sentence in your notes: "On this page, <the subject's own material> becomes <the device>."
Fill the first blank from your inventory of what only this client has (its name and what it means, its street, its numbers, its
constraint), then put the nearest competitor in the sentence: if it still works, the idea is the category's, not the client's.
Then name the one live thing the visitor can do on the page (SKILL.md step 4).

## 2. Composition
Hero: a short plain headline (4–6vw, left-aligned, upper-left third) and one sentence; the rest of the first screen is the main fragment at actual interface size, sitting directly on the page or on one flat stage colour, large enough to read every label. The fragment is built in HTML and works. Below, capabilities are a single-column sequence, each a row of two parts on a 5/7 split: a two-line statement on one side and the relevant fragment on the other, with the fragments varying in size and shape so the column never becomes a zigzag of equal blocks. Fragments are cropped pieces of interface (a menu, a row, a control, a panel edge), not whole windows. One section pins a fragment while three short statements change it.

## 3. Type — Work Sans
A light grotesque with open, slightly old-fashioned shapes; large headings look airy and the text weight beneath stays easy to read. Suits AI and research products, fashion houses and calm brand sites.
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Work+Sans:wght@300;400;500&display=swap" rel="stylesheet">
```
- Display: **Work Sans** — weight 300 · tracking -0.035em · leading 1.0 · sentence case
- Text: **Work Sans** — 400 for body, 500 for emphasis · leading 1.55 · tracking 0
```css
:root {
  --step-small: clamp(0.8125rem, 0.79rem + 0.1vw, 0.875rem);
  --step-body: clamp(1.0625rem, 1rem + 0.25vw, 1.1875rem);
  --step-lead: clamp(1.25rem, 1.1rem + 0.7vw, 1.625rem);
  --step-head: clamp(2rem, 1.3rem + 3.2vw, 4rem);
}
```
The marketing type matches the product's interface type so the fragments and the page feel like one thing. Headline modest; interface text at its true size (13–15px) inside fragments.
The direction decides sizes and how many of them; the type system decides how this face is set. Where the two give different tracking or leading, use the values of the type system: they are tuned to this face.
The display size itself comes from the hero blueprint below.

## 4. Colour — Green-black and bone
```css
:root {
  --bg: #161c1a;
  --surface: #1f2725;
  --ink: #e3e2da;
  --muted: #8e948d;
  --accent: #9fc7b8;
  --accent-ink: #161c1a;
}
```
Bone type on green-black from top to bottom, with panels and drawn details barely lighter than the ground so photographs supply the colour. Pale jade is for the booking or contact button alone.
Neutral ground. The product's own interface colours provide the colour. At most one flat stage colour behind fragments.

## 5. Imagery
**This subject is software and there are no screenshots, so its pictures are its own interface.** Wherever this sheet says photograph, image or object, build a believable fragment of the product in HTML at real size (real labels, numbers that add up, one item hovered or selected) by the app-screen rules (references/app-screens.md: deal its style once with `node <skill>/scripts/draw.mjs --app --kind <kind> --keep palette,type --out .taste-skill/app`, and build it with assets/app.css and assets/charts.js), put the attribute `data-ui` on its root, and crop or scale it the way the blueprint treats a photograph: a full-bleed close crop, a tile, the fixed half of a split. Where a fragment makes no sense, use a flat colour field with type on it. Never a fake dashboard of grey bars, never a device frame. If the product works on something physical (ships, plants, buildings, a city), one real picture of that world can carry a section: find it with `node ~/.claude/skills/design-taste-frontend/scripts/assets.mjs "<that world>"`, download it with `--get`, credit it on the page, and never let it pass for the product or its customers.

The interface fragments are the main imagery: build them carefully with real labels, real numbers and sensible states (one hovered, one selected). Around them, two or three real pictures of the customer's world (found with scripts/assets.mjs), never stock of people at laptops.

**Floor (the inspector checks it):** at least four real pictures on the page, usually six to twelve; SVG drawings do not count. One moving picture where an honest clip exists: `node <skill>/scripts/assets.mjs --video "<query>"`, used as a muted loop with a poster frame. Whatever this direction says about pictures, it is carried out with real ones.

Optional atmosphere, two at most, from the "Sky and space" set (skies, clouds, stars and night skies). The page does not need them.
Order of preference for imagery: (1) the project's own photographs and product shots; (2) images you generate, if an image tool is
available; (3) real pictures of the subject's world found in the open collections (assets.mjs, downloaded and credited on the page);
(4) drawings of the subject in code; (5) these. They are real photographs (Unsplash licence, served by picsum.photos) of places, light and
texture. They are atmosphere, never a stand-in for a specific product, dish, person or interface. List the ones you use in your report so
they can be replaced.
- `https://picsum.photos/id/505/1600/1000` — Tiny silhouetted figure on a dark ridge under a blue and orange twilight sky · dark tone · deep blue, orange, black · calm area: top · native landscape · photo: lee Scott
- `https://picsum.photos/id/974/1600/1000` — Person standing beneath the Milky Way with a warm glow on the horizon · dark tone · deep navy, violet, amber · calm area: left · native landscape · photo: Greg Rakozy
- `https://picsum.photos/id/607/1600/1000` — Soft streaked clouds in a blue sky above a calm sea · mid tone · sky blue, white, teal · calm area: all · native portrait · photo: Grégoire Hervé-Bazin
- `https://picsum.photos/id/833/1600/1000` — Tiny aeroplane crossing a wide, softly overcast sky · light tone · warm grey, off-white · calm area: all · native portrait · photo: Paul Green
The last two path numbers are width and height: request the exact crop you render (`/1200/1500` for a 4:5 portrait, `/2400/1350` for a hero) and
add `?grayscale` to desaturate. Use at most two. Always set `width`, `height`, `alt` and `loading="lazy"` (not on the hero image).
Other sets that fit this brief: architecture, fog-and-quiet, water — `node draw.mjs --photos <set>` prints one in full; `--photos list` shows all sets. Taking photographs from another set is your call and is not a redraw; keep them to one light and one treatment.

## 6. Hero — Controls on the sentence
A dark flat ground with no logo bar: a 14px wordmark top-left and the single action as an underlined text link top-right, both 28px in. The headline is a two-line sentence, centred, at clamp(96px, 9vw, 150px), line-height 1.0, occupying the middle 64% of the width and sitting slightly above the midline. It is set in a variable font and it is the demonstration. Three small control chips from the product sit on top of the sentence, each about 300 by 80px with the product's true radius, a text label and a numeric readout: one above the first line near its left end, one overlapping the start of the second line, and a two-state switch at the lower right. They are tilted 2 to 4 degrees and filled solid, one step lighter than the ground. Each one works: dragging the first changes the headline's weight through `font-variation-settings`, the second changes its letter-spacing, the third toggles uppercase. Mark each chip's root with data-ui. One supporting sentence at 17px, max 40ch, is centred 72px under the headline. The bottom third of the screen stays empty.

On a phone: The sentence stays on two lines at 13.5vw, centred at 38% of the height. The three chips shrink to 200 by 60px and move off the letters: one above the sentence and two side by side beneath it, untilted and still live. The supporting sentence follows at 16px. The wordmark stays top-left and the action link top-right.
Headline: A two-line sentence of three to five words, up to about 11 characters per line at 9vw, that still reads when its weight and spacing change.

Adjusted for this page:
- It names a kind of typeface: use the dealt faces (display Work Sans, text Work Sans) and their settings; capitals only where the type system sets them.
- It sets a sentence large over several lines: jurors read that as a long headline. Either cut it to a two-line headline at display size, or keep it a statement at 36px or less on three lines (24px, four lines on a phone).
- Its action is a text link: the primary action on the first screen is a real button (filled or solid-bordered, 44px or taller), placed where the blueprint puts the link. A second action may stay a link.

## 7. Sections, in order
Each is a composition to rebuild with this project's content. Keep the geometry; change the subject. Re-order or drop a
section when the argument of the page needs it (see references/copy.md), never to fall back to a card grid.

### 1. intro — Statement with inline thumbnails (statement)
A centred statement of 18–28 words, four lines at clamp(32px, 4.2vw, 62px), medium or bold weight, leading 1.12, in a block 60–66% of the viewport wide with 22vh of empty ground above and below. Three to five small frames sit inline between words: each is an `inline-block` 0.85em tall and 0.85–1.3em wide, `vertical-align: -0.1em`, radius 0.14em, holding a tightly cropped photograph (or a crop of real interface) of the thing the neighbouring word names. Frames are part of the line, never floating over it, with 0.12em of space either side. No heading above and no paragraph below; the statement is set as a `p`. Directly under the section the page may split into two flat colour halves or begin the first product row, so the thumbnails read as a preview of what follows. On a motif-driven page every frame takes the motif's shape.
On a phone: Left-aligned instead of centred, at 8vw with leading 1.15, about seven lines inside 20px margins. Keep at most three inline frames, each exactly one cap-height tall, and make sure none lands at the start of a line. Vertical padding drops to 96px.

Adjusted for this page:
- It sets a sentence large over several lines: jurors read that as a long headline. Either cut it to a two-line headline at display size, or keep it a statement at 36px or less on three lines (24px, four lines on a phone).

### 2. work — Quiet ledger, side preview (list)
A plain ledger in one column 560px wide whose left edge sits at 22% of the viewport width. A heading of two or three words at 24–28px in medium weight, then 32px of space, then one row per project, 44px tall: the name at 16px in full ink at the left, the kind of work at 14px in the soft ink starting at 55% of the column, and the year right-aligned with tabular figures. Nothing is drawn between rows; the three fixed starts make the columns. Hovering or focusing a row keeps it in full ink, drops the others to the soft ink, and shows that project's image in the empty area right of the column: a 4:3 frame 300px wide, `position: absolute`, its top aligned to the hovered row, 64px from the column's right edge, appearing with a 150ms fade. Holds 6–20 rows; group by year with 28px of extra space.
On a phone: The column fills the width with 20px margins. Rows are 52px tall: name at the left, year at the right, kind of work on a second line at 13px. There is no hover preview; instead the first three projects each show a 16:10 image at full column width above their row.

### 3. stats — Figure set by a slider (statement)
The number is the visitor's own. A flat field at least 88svh tall. Top-left, on the page margin: a question of two to five words at 36–44px that serves as the heading ('How many invoices a month?'). 28px under it, one control from the product at true size, 400px wide, root marked data-ui: a slider with its real label and current value, or a stepper, or a short numeric field, preset to a typical answer. Then a long gap. Bottom-left, on the same edge, the result at clamp(140px, 19vw, 320px), heavy, leading 0.8, tabular numerals, with its unit at 0.22em on the same baseline. It recalculates on every input event from a stated, honest formula. Bottom-right, on the figure's baseline, one sentence at 17px, 30ch wide, says how the figure is worked out. No other figure appears in the section.
On a phone: Question at 26px, then the control at full width with a 44px thumb. The figure sits at the bottom at 30vw with its unit beneath it at 20px, both flush left. The working-out sentence follows at 15px. The section is at least 84svh so the gap between control and figure survives.

### 4. showcase — Output index, live sheet (index)
For products that make documents: invoices, reports, emails, statements. Columns 1–5 list the names of three to five outputs, one per line, at clamp(36px, 4vw, 64px), leading 1.1, each a `<button>`; the selected name is in full ink, the rest in the soft ink. There is no other heading. Columns 7–12 are a stage in a flat tone one step from the ground, 88svh tall, and on it lies the selected output as an object: a paper-white sheet 560px wide, built in HTML at true size, root marked data-ui, with a real sender, real line items, sums that add up and a date. The sheet starts 72px below the stage's top and runs past its bottom edge, which crops it. Selecting a name swaps the sheet with a 200ms crossfade; an email is narrower (440px), a receipt narrower still (300px), all left-aligned 64px into the stage.
On a phone: Names become one sideways-scrolling row at 22px with scroll-snap, the selected one in full ink. Beneath, the stage fills the width and is 70svh tall; the sheet keeps its true size, anchored top-left with a 16px inset and cropped by the stage's right and bottom edges. Tapping a name swaps it.

Adjusted for this page:
- It names colours (white): map them onto this palette's roles (ground `--bg`, panels `--surface` and the fields, text `--ink`/`--muted`, the accent `--accent` on the primary action only).

### 5. about — Founder's letter (stack)
A letter in a centred column 520px wide with at least 160px of empty ground above and below. At the column's top-left, a round portrait 44px across. 28px below it, four or five short paragraphs at 16–17px with leading 1.55, each two to four lines and separated by one blank line, written in the first person. No heading. The sign-off is the writer's name at 22px in the text face's italic, with the role at 14px in the soft ink beneath it. One phrase in the letter may carry a flat marker-colour background. Variant for a page built as an object: the letter sits on a filled paper-colour sheet 640px wide with 60px padding on a contrasting ground, square corners, its bottom edge cut into small scallops with `clip-path`, and below a row of 5px round dots a tear-off strip 72px tall that carries the email address.
On a phone: The column takes the full width with 24px margins; text at 16px, portrait 40px. In the sheet variant the sheet spans edge to edge with 24px padding and keeps its shaped bottom edge; the tear-off strip stacks the email above a full-width button.

Adjusted for this page:
- It names a kind of typeface: use the dealt faces (display Work Sans, text Work Sans) and their settings; capitals only where the type system sets them.

### 6. list — Day tabs, banded agenda (list)
A two-line heading at clamp(36px, 4vw, 60px), first line in the soft ink and second in full ink, with the booking button on the same row at the far right. 56px below, left: a 150px column of stacked day tabs, each a filled block 150 by 64px with 6px gaps showing the weekday at 16px and the date at 13px; the active tab takes the darkest fill with inverted text. Right, from 190px to the right margin: the day's sessions as alternating tone bands, each 76px tall with an 8px radius and 4px gaps, the two tones one step apart. Inside a band, on fixed column starts: the time at 20px with tabular figures, 20px from the left; the title at 18px in medium weight at 22%, with a one-line note at 13px in the soft ink under it; speaker names right-aligned at 14px beside up to three overlapping 32px round portraits.
On a phone: The heading stacks above a full-width button. Day tabs become a horizontal row of equal filled blocks, sticky under the top edge. Each band stacks time over title over speakers with 16px padding and grows to fit; portraits sit at the band's top-right corner.

Adjusted for this page:
- It names its own corners: use the one radius of this page (square unless the direction says otherwise) and one button shape.

### 7. cta — First step, already open (split)
The page ends with the product's first screen already open. A full-bleed field in the accent colour, 90svh tall. Right, in columns 7–11, vertically centred: the product's own create dialog at true size, 440px wide, root marked data-ui, with its real title ('Name your workspace'), one text field holding an editable, believable example, one two-option segmented choice and its primary button. Inside the fragment the product's conventions apply, including the shadow its dialog normally casts. The field accepts typing, the choice switches, and the primary button is the page's only action: it carries the typed value into sign-up. Left, in columns 1–5, anchored to the bottom with 40px margins: the closing line, two to four words at clamp(56px, 7vw, 112px), leading 0.92, on two lines, and under it one sentence at 17px saying what happens after the button. The upper-left stays empty.
On a phone: The closing line leads at 13vw on two lines, top-left, with its sentence beneath at 16px. The dialog follows at full width inside 16px margins, at true size, its primary button 52px tall and full width. The field is at least 84svh; the dialog sits in its lower half.

Adjusted for this page:
- It mentions a shadow: only on something open above the page (a menu, a dialog), never on a resting element.

### 8. footer — A footer of tiles (grid)
The footer is the last two rows of a board: filled tiles with 8px gaps and 8–12px outer margins, every cell used. Upper row, each tile 220px tall: a statement tile in columns 1–4 with one sentence at 24px top-left and the legal text at 12px bottom-left; a navigation tile in columns 5–7 with a plain list at 16px; a tile in columns 8–9 giving the city and local time as text; an email tile in columns 10–12 with the address at 20px as a link, bottom-left. Lower row, 320px tall: a tile in the ink colour spanning columns 1–8 with the wordmark set large (about 9vw) and centred, and a tile in the accent colour in columns 9–12 with a two-to-four-word invitation top-left at 28px and an arrow, drawn as inline SVG with 4px strokes, in its bottom-right corner; the whole accent tile is a link.
On a phone: Two columns of tiles with 6px gaps: the wordmark tile full width first at 160px tall, the accent tile full width next, then navigation and email side by side, then time and statement. Every tile keeps 20px padding and top-left text.

Adjusted for this page:
- It sets text at 12px: use your small step (13px or more) instead.

## 8. Motion — Masked headline
On load, each line of the headline rises from behind its own baseline, about 900ms with a strong ease-out, the second line 110ms after the first; the supporting sentence and the action then fade up together. Nothing else on the page performs. Because headlines are two lines at most, you author the line breaks and wrap each line yourself. Recipe: 'Masked line reveal' in references/motion.md.
This is the signature move, the one loud moment. Around it the page still needs the rest of the motion layer (SKILL.md step 9): an entrance for the parts of every section, feedback on every control, and one moving picture. The inspector fails a page that barely moves. Recipes: references/motion.md.

## 10. Where the richness comes from
Fidelity. A fragment with believable content and a real hover state is more interesting than any illustration. Spend the effort there.

## 11. How this direction goes wrong
A fake interface made of grey bars and circles is the worst outcome; if you cannot build a believable fragment, change direction. Mark each fragment's root element with the attribute data-ui: inside it the product's own conventions apply (1px control borders, small radii, a shadow on its open menu) and the inspector leaves them alone; outside it the house rules hold as usual. data-ui is for cropped fragments only, never for a page wrapper. Do not put the fragment in a browser or laptop frame.

## 12. Real references
Studied sites that work this way. Take the principle, never the specifics: no names, copy, images or signature assets of theirs.
- **Rows** (One Page Love, studied score 7.5) `opl-rows`
  Idea: Radical understatement: a product about simple lists is sold with a 27px headline, 12px copy and a column of small live demos, each proving one interaction.
  Hero: Warm off-white page, almost empty. Everything hangs on one vertical line 27% in from the left. A small striped-letter logo mark sits there near the top; 220px lower comes a two-line headline at only 27px (under 2vw), then three lines of 13px body about 370px wide, then a near-black rectangular button with 4px corners beside a text link with an arrow. A single grey sign-in link sits in the top-right corner; there is no navigation bar. More than 80% of the viewport is blank. At the bottom edge the first feature module peeks in: grey demo panel left, short text column right.
  Worth taking: Set the headline at near-body size (under 2vw) and let empty space, not scale, give it importance.
- **Typewave** (One Page Love, studied score 7.2) `opl-typewave`
  Idea: A plugin about type scales sold on a near-black page where flat saturated fields (orange, lime, blue) act as stages for the plugin's own dark interface.
  Hero: Near-black field. Nav row at the top: pixel-block logo and wordmark left, four plain links centred (one carrying a tiny lime tag), a cream rectangular button with a Figma glyph right. The headline sits centred around mid-height: three lines at about 6.7vw, medium weight, off-white, with the third line recoloured pale lime. Below it a three-line 18px paragraph held to about 45% of the width, then a small cream button repeating the nav action. About 150px lower, the top edge of a flat orange slab, 70% of the viewport wide with barely rounded corners, peeks over the fold; it holds the plugin window.
  Worth taking: Stage dark UI screenshots on flat saturated rectangles instead of on the page background, so the colour does the branding and the UI stays legible.
- **Made With Gsap** (Awwwards Site of the Day, studied score 7.9) `aww-made-with-gsap-1`
  Idea: The hero is the product: a draggable vertical reel of effect demos runs through the middle of the headline, which is split and pushed to both edges to make room.
  Hero: White viewport. A one-line headline at 3.3vw is broken in two: the first half flush to the left margin, the second flush to the right, both on the vertical centre line. Between them a black rounded video card, 35% of the width and 16:9, sits dead centre with a mono catalogue number under its left corner; the cards before and after it peek in from the top and bottom edges. Wordmark top-left; top-right a grey capsule of mono uppercase links with a count chip, a round account button and a lime capsule button. Bottom-left a two-line 18px statement; bottom-right a grey mono drag hint.
  Worth taking: Split a single-line headline in two and pin the halves to the left and right edges so the media sits inside the sentence.
Full notes on any of them: `node draw.mjs --site <id>`.

## Always
Headlines two lines or fewer at every width. No eyebrow labels, no monospace, no 01/02/03, no thin rules, no gradients, no shadows at rest,
no cream-and-terracotta, no violet. The frame: one h1 on the first screen that says what this is, equal outer margins, no
section huddled on one side with the other half empty, no blank stretch over half a screen, a heading on every section.
`inspect.mjs` measures all of it.

Saved to website/v1/.taste-skill/direction.json · reprint with `node draw.mjs --show`
