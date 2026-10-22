# DIRECTION SHEET — The dark room
seed 1403016300 · drawn 2026-10-22 · sector personal-portfolio · tone quiet
Brief: 3-DOF robot arm learns reach, hold, track, avoid and push with reinforcement learning in MuJoCo; honest ablations

| | |
| --- | --- |
| Direction | The dark room |
| Hero | Text band, then sheet |
| Type | Sentient |
| Colour | Black and crimson |
| Sections | intro (Pinned sentence, photographs pass) → work (Big names, fixed preview) → stats (Quiet figure lines) → showcase (Close-up with column card) → about (Pinned title, portraits pass) → list (Sticky photograph, active list) → cta (A plain sentence, one action) → footer (Name across the last photograph) |
| Motion | Index with preview |
| Photographs | the "Architecture" set, as atmosphere |

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
One dark field from top to bottom, and in it, lit things. Photographs and objects appear like exhibits in a dim gallery: each with space around it, each the brightest thing on its screen. Type is light, small and sparse. The darkness is not a theme switch; it is a room, and the page is a slow walk through it. It suits subjects sold on atmosphere and material: an evening restaurant, a metal card, a record, a house at dusk.

Before anything else, write one sentence in your notes: "On this page, <the subject's own material> becomes <the device>."
Fill the first blank from your inventory of what only this client has (its name and what it means, its street, its numbers, its
constraint), then put the nearest competitor in the sentence: if it still works, the idea is the category's, not the client's.
Then name the one live thing the visitor can do on the page (SKILL.md step 4).

## 2. Composition
A single near-black or deep-toned ground runs the full page with no section bands. First screen: either the name at full width in a light weight along the bottom edge with one dim photograph above it, or one object or photograph centred at 40–50% of the viewport with a line of text beneath at 16–18px. Each later screen holds one or two lit items at a generous size (the main one 40–60% of the screen width) with its text set beside it at full reading size; the darkness frames the items, it does not replace content. Images vary in size but all share the same darkness at their edges so they sit in the ground without frames. Practical details come last in a simple two-column text block. The only solid light element on the page is the single primary action.

## 3. Type — Sentient
A soft, low-contrast serif in a light weight, more book than fashion magazine, large and calm in sentence case. Suits outdoor and craft products, restaurants, cameras and slow consumer goods.
```html
<link rel="preconnect" href="https://api.fontshare.com">
<link rel="preconnect" href="https://cdn.fontshare.com" crossorigin>
<link href="https://api.fontshare.com/v2/css?f[]=sentient@300,400,500&display=swap" rel="stylesheet">
```
- Display: **Sentient** — weight 300 · tracking -0.015em · leading 1.0 · sentence case; about 8vw for the headline, 40px for statements
- Text: **Sentient** — 400 for body, 500 for emphasis · leading 1.55 · tracking 0
```css
:root {
  --step-small: clamp(0.8125rem, 0.79rem + 0.1vw, 0.875rem);
  --step-body: clamp(1.125rem, 1.05rem + 0.3vw, 1.3125rem);
  --step-lead: clamp(1.25rem, 1.1rem + 0.7vw, 1.625rem);
  --step-head: clamp(2rem, 1.3rem + 3.2vw, 4rem);
}
```
Light or regular weights; they glow on dark. Avoid heavy bold, which looks cheap on black. Headings 4–7vw at most; text 16–18px with leading 1.6 and a slightly lifted ink (not pure white).
The direction decides sizes and how many of them; the type system decides how this face is set. Where the two give different tracking or leading, use the values of the type system: they are tuned to this face.
The display size itself comes from the hero blueprint below.

## 4. Colour — Black and crimson
```css
:root {
  --bg: #0a0a0a;
  --surface: #151515;
  --ink: #efefef;
  --muted: #8c8c8c;
  --accent: #b11226;
  --accent-ink: #ffffff;
  --field-1: #efefef;  --field-1-ink: #0a0a0a;
}
```
Black with off-white text and mid-grey texture takes nearly the whole page, and the last sections may invert to off-white with black type. Deep crimson is scattered sparingly: a few characters in a heading, a slider fill, the one button.
A dark ground with a tint (blue-black, green-black, brown-black), a soft light ink, a dimmer ink for secondary text, and one warm or bright accent used once per screen at most.

## 5. Imagery
**The project has no photographs of its own yet.** First choice: **real pictures of the subject's world from the open collections**: its city and street in old postcards and photographs, an old map of the neighbourhood, its materials up close, its trade in engravings, museum objects of its kind (references/imagery.md §3). Start with `node ~/.claude/skills/design-taste-frontend/scripts/assets.mjs "mujoco postcard"`, then "mujoco old map" and "learns reach", then queries from your inventory (its street, its materials, its trade's tools and history). Download what you use into the project with `--get` and credit every one on the page (`--credits` prints the footer line). They show the world around the client, never the client's own room, dish, product, person or team. Second, for what no honest picture exists of: draw the subject's own things in code, large, in one flat graphic language and the palette's real colours (its tiles, its windows, its objects; references/imagery.md §8). Third: the dealt photographs, only where an honestly generic picture fits (the sea, the weather, a material), never one a visitor could take for the thing itself and never an unrelated scene. One treatment for all of them. If this direction is built on photographs, plan the set before building (four to six pictures of one family) and reuse them at different crops and scales instead of finding or drawing a new picture for every slot; keep drawings as `<symbol>` + `<use>` or generate them with a small script so the page stays light. List every found picture (with its source and licence), every dealt photograph and every drawing that stands in for photography in your report.

Low-key photographs: dark backgrounds, one light source, subjects emerging from shadow. Their dark edges should blend into the page ground.

**Floor (the inspector checks it):** at least four real pictures on the page, usually six to twelve; SVG drawings do not count. One moving picture where an honest clip exists: `node <skill>/scripts/assets.mjs --video "<query>"`, used as a muted loop with a poster frame. Whatever this direction says about pictures, it is carried out with real ones.

Photographs dealt for this build, from the "Architecture" set (structure, lines, bridges, facades and building interiors)
Order of preference for imagery: (1) the project's own photographs and product shots; (2) images you generate, if an image tool is
available; (3) real pictures of the subject's world found in the open collections (assets.mjs, downloaded and credited on the page);
(4) drawings of the subject in code; (5) these. They are real photographs (Unsplash licence, served by picsum.photos) of places, light and
texture. They are atmosphere, never a stand-in for a specific product, dish, person or interface. List the ones you use in your report so
they can be replaced.
Treatment: If a photograph's background is lighter than the page, darken the image slightly (`filter: brightness(0.85) contrast(1.05)`) rather than adding a frame.
- `https://picsum.photos/id/693/1600/1000` — Red suspension bridge tower rising from fog above rugged cliffs · mid tone · rust red, grey, dark teal · calm area: left · native landscape · photo: Rob Bye
- `https://picsum.photos/id/942/1600/1000` — White angular concrete building against a black sky · dark tone · black, white, pale grey · black-and-white · calm area: top · native landscape · photo: Carl Nenzen Loven
- `https://picsum.photos/id/639/1600/1000` — White Ferris wheel curving against a clear blue sky · mid tone · bright blue, white · calm area: left · native landscape · photo: Ilya
- `https://picsum.photos/id/392/1600/1000` — Red suspension bridge spanning a bay beneath a pink sunset sky · mid tone · dusty pink, rust red, warm brown · calm area: top · native landscape · photo: Chris Brignola
- `https://picsum.photos/id/781/1600/1000` — Dark station platform with a row of bright windows, black and white · dark tone · black, white, charcoal · black-and-white · calm area: top · native landscape · photo: Matthew Wiebe
- `https://picsum.photos/id/692/1600/1000` — Grand station concourse with arched windows and a blurred crowd · mid tone · warm gold, cream, grey · native landscape · photo: Nicolai Berntsen
- `https://picsum.photos/id/1031/1600/1000` — Twin glass towers converging upward against a grey sky · mid tone · steel grey, dark blue, pale grey · native landscape · photo: Mike Wilson
- `https://picsum.photos/id/870/1600/1000` — White lighthouse with its lamp lit against a deep blue starry sky · mid tone · deep blue, white, warm orange · calm area: right · native portrait · photo: Joshua Hibbert
The last two path numbers are width and height: request the exact crop you render (`/1200/1500` for a 4:5 portrait, `/2400/1350` for a hero) and
add `?grayscale` to desaturate. Use three to six of them, not all eight. Always set `width`, `height`, `alt` and `loading="lazy"` (not on the hero image).
Other sets that fit this brief: hands-and-making, people, fog-and-quiet, object-still — `node draw.mjs --photos <set>` prints one in full; `--photos list` shows all sets. Taking photographs from another set is your call and is not a redraw; keep them to one light and one treatment.

## 6. Hero — Text band, then sheet
A black or white field on a seven-column grid with 14px margins and 14px gaps. There is no display type: everything is set at one size, 16px, in one weight. The top band is plain text hanging from a line 20px below the top edge: the name in column one; one sentence on availability in column two; a list of five or six services from column three; a longer list of clients or awards from column five; and two links, profile and the single contact action, at the right end of column seven. Lists run one item per line. Below the lists about 300px of ground is left empty. At mid-height the sheet begins: one thumbnail per column, each the full column width (about 190px) at its natural ratio, top-aligned, with the project title in the text size above the first frame of each run of two to four frames. Rows continue below the fold with 40px between them. Frames have square corners and sit directly on the ground with no border and no caption on the image.

On a phone: The grid becomes two columns with 12px gaps. The name and the availability sentence take the top; services and clients sit side by side under them as two short lists at 15px. An 80px gap follows, then the sheet: two frames per row at natural ratios, each project title above its first frame. The contact action is a text link fixed bottom-right.
Headline: None at display size: the name is set at 16px with the rest of the text, and one sentence of up to 60 characters says what the person does.

Adjusted for this page:
- It names colours (black, white): map them onto this palette's roles (ground `--bg`, panels `--surface` and the fields, text `--ink`/`--muted`, the accent `--accent` on the primary action only).

## 7. Sections, in order
Each is a composition to rebuild with this project's content. Keep the geometry; change the subject. Re-order or drop a
section when the argument of the page needs it (see references/copy.md), never to fall back to a card grid.

### 1. intro — Pinned sentence, photographs pass (pinned)
A wrapper 300svh tall. Inside it a `position: sticky; top: 0` layer of 100svh holds one sentence of 8–14 words, centred, three lines at clamp(36px, 4.6vw, 68px), leading 1.1, in a block 62% of the viewport wide. The sentence is a paragraph, not a heading; it is white with `mix-blend-mode: difference` (or a solid ink that reads over every photograph) and has a higher `z-index` than the images. Behind it, three photographs sit in normal flow at different sizes and sides and scroll past the fixed sentence: a 4:5 portrait 24vw wide with its left edge at 6vw; 70svh lower a 16:10 landscape 38vw wide touching the right margin; 70svh lower a 1:1 frame 18vw wide starting at 30vw. Each crosses the sentence once. When the wrapper ends, the sentence releases and scrolls away with the last photograph. No scroll hijacking.
On a phone: Wrapper 260svh. The sentence pins at 40% of the screen height at 8.4vw, four or five lines, with 20px margins. Photographs alternate: 64% wide on the left margin, 78% wide on the right margin, then 52% on the left, with 50svh between them.

Adjusted for this page:
- It names colours (white): map them onto this palette's roles (ground `--bg`, panels `--surface` and the fields, text `--ink`/`--muted`, the accent `--accent` on the primary action only).
- It sets a sentence large over several lines: jurors read that as a long headline. Either cut it to a two-line headline at display size, or keep it a statement at 36px or less on three lines (24px, four lines on a phone).
- It pins something for three screens or more: hold a pin for two or three screens at most, give every screen under it its own content, and let nothing pass over the pinned words (the inspector checks all three).
- It blends text with `difference`: over anything but black, white or grey that gives random hues. Use a solid ink chosen for each ground instead.

### 2. work — Big names, fixed preview (index)
Project names are the index. In columns 1–8, five to nine names stack one per line at clamp(44px, 5.4vw, 80px), leading 1.0, each row a link with 10px of vertical padding. All names rest in the ink at 35% opacity except the active one, which is full ink; the first is active by default, and hover or keyboard focus moves the state. To the right of each name, on its baseline at 14px: the discipline and the year. In columns 9–12 a `position: sticky; top: 15vh` frame, 4:5 and 26vw wide with square corners, shows the active project's photograph; all images are stacked in the frame and swapped by opacity over 200ms. A two-word heading at 28px sits top-left, 48px above the first name. Nothing is drawn between rows: the size of the type and the dimming do the separating.
On a phone: Names at 9.5vw, all in full ink, one per line; discipline and year sit beneath each name at 13px. There is no sticky frame: each name is preceded by its own photograph at 16:10, full width, so the list becomes image then name, repeated, with 40px between projects.

Adjusted for this page:
- It fades text: anything a visitor reads, inactive items of a list included, is `--ink` or `--muted` at full strength (contrast 4.5:1 or more). The active item is told apart by `--ink` against `--muted`, weight or the accent, not by fading the rest.

### 3. stats — Quiet figure lines (list)
Numbers stated without raising the voice. In columns 2–7, a definition-list layout: three to five rows, each a two-track grid. The first track is 132px wide and holds the figure, right-aligned, at 28–32px in medium weight and full ink with tabular numerals. The second track, after a 24px gap, holds one sentence at 16–17px in the soft ink, at most 40ch, whose first baseline matches the figure's baseline. Sentences share one left edge and figures one right edge, so a clean channel of space runs down between them. 28px between rows. The heading, two to four words at 24–28px, sits 56px above on the sentences' left edge, not the figures'. Optionally, a single small square photograph occupies columns 9–11, its top 80px lower than the first row, with no caption. Leave at least 18vh of empty ground above and below the block.
On a phone: The grid stays two-track, with the figure track narrowed to 88px and figures at 24px; sentences at 16px wrap beside them. If the photograph is used it comes after the rows at 56% width, flush right. The heading is 22px on the sentences' left edge.

### 4. showcase — Close-up with column card (full-bleed)
A full-bleed close-up photograph, 100vw by 100svh, with nothing over most of it. A four-column grid lies on top. One solid card in the ink colour, exactly one grid column wide (about 22vw) and 52svh tall, sits in the second column, vertically centred. Inside it, centred: the item's name at 30–36px, a variant line at 17px, the price or key fact at 17px, a small 3:2 photograph of the whole object (the card shows the whole, the background shows the detail), and at the foot a square-cornered action filling the card's width. The card is opaque, radius 0–6px, with no blur. When the section presents two or three items, each gets its own full-bleed frame and its card moves one column to the right each time, so position carries the sequence. The first card's name is the section heading.
On a phone: The photograph fills 100svh, cropped to its most detailed area. The card becomes a full-width block docked to the bottom, 46svh tall, left-aligned inside with 20px padding; the small whole-object photograph sits to the right of the name at 96px wide.

### 5. about — Pinned title, portraits pass (pinned)
A wrapper tall enough for its portraits: about 90svh per person, up to six people. A `position: sticky; top: 0` layer of 100svh holds the heading, two words at clamp(88px, 12vw, 180px), leading 0.85, centred at mid-height, in white with `mix-blend-mode: difference` so it reads over ground and photographs alike. Beneath it in the stacking order, portraits scroll past in normal flow: each a 4:5 photograph 26–30vw wide, alternately starting at column 2 and ending at column 11, 60svh apart, so each one passes behind a different half of the heading. Under each portrait's left corner: the name at 18px, the role at 14px in the soft ink, and one sentence of up to 30 words in a 280px measure. The last portrait is followed by 40svh of empty ground before the heading releases. Square corners. Nothing else is on screen.
On a phone: The heading shrinks to 17vw and pins 16px from the top instead of at mid-height. Portraits become 72% wide, alternating between the left and right margins, 48px apart, each with name, role and sentence beneath at the portrait's width.

Adjusted for this page:
- It names colours (white): map them onto this palette's roles (ground `--bg`, panels `--surface` and the fields, text `--ink`/`--muted`, the accent `--accent` on the primary action only).
- It blends text with `difference`: over anything but black, white or grey that gives random hues. Use a solid ink chosen for each ground instead.

### 6. list — Sticky photograph, active list (split)
Two columns. The left, columns 1–5, is `position: sticky; top: 12vh` and holds one photograph or object at 4:5, 34vw wide, unframed on the ground. The right, columns 7–11, holds a vertical list of five to nine names (treatments, rooms, courses, services) at clamp(20px, 1.6vw, 24px), each row 58px tall with the name at the left and a price or duration right-aligned at 15px in the soft ink. The active row is full ink; the others rest at 40% opacity. Hovering, focusing or scrolling a row to the middle of the viewport makes it active, crossfades the left photograph to that item's image over 300ms, and opens a three-line description at 15px, 34ch wide, directly under the name, pushing later rows down. A heading of two or three words at clamp(32px, 3.4vw, 48px) sits above the list, aligned to its left edge, 48px above the first row.
On a phone: The photograph becomes a 4:3 band pinned to the top of the screen at 38svh while the list scrolls beneath it; the heading sits above the first row. Rows are 56px tall and the description opens on tap. The image changes with the row nearest the band.

Adjusted for this page:
- It fades text: anything a visitor reads, inactive items of a list included, is `--ink` or `--muted` at full strength (contrast 4.5:1 or more). The active item is told apart by `--ink` against `--muted`, weight or the accent, not by fading the rest.

### 7. cta — A plain sentence, one action (stack)
The page ends on an almost empty screen. On the page ground, with no band and no panel, one sentence of eight to fourteen words is set at 18–22px in the regular weight, in the full ink, no wider than 34ch, on the same left edge as the page's text column (or centred only if the whole page is centred). 24px beneath it sits one small filled button, about 40px tall with 18px side padding, and optionally one text link beside it on the same baseline. That is all. Above the sentence leave at least 30svh of empty ground and below it at least 20svh before the footer line, so the emptiness reads as decided. The sentence is written as something a person would say, and it names the action plainly. No headline-sized type appears anywhere in this section.
On a phone: Identical in spirit: the sentence at 18px with 20px margins, the button beneath at its natural width (not full width), 44px tall. Keep 20svh of empty ground above the sentence and 14svh below the button, before the footer line.

### 8. footer — Name across the last photograph (full-bleed)
The footer is the page's last photograph, full-bleed at 100vw by 100svh, darkened evenly with `filter: brightness(0.62)` so paper-coloured type reads everywhere on it. Three layers of text sit on the image. Top row, 32px from the edge: a short link list at 16px at the left margin and contact details at the right margin, each no wider than three columns. Mid-height is left to the photograph. Along the bottom, the wordmark runs the full width in the paper colour, letters about 12–20vw tall depending on the name's length, baseline 56px above the edge; under it one line of legal text at 13px. Choose an image whose interest is in its middle band, since the top and bottom carry text. No panel, no box and no colour block separates the footer from the image: the photograph is the footer.
On a phone: The photograph is 100vw by 88svh, cropped to its strongest vertical slice. Links stack at the top-left at 16px, contact beneath them. The wordmark is fitted to the width along the bottom, with the legal line under it at 12px.

Adjusted for this page:
- It sets text at 12px: use your small step (13px or more) instead.

## 8. Motion — Index with preview
Rows of text are the navigation through the work; pointing at a row brings up its image in one fixed frame and dims the other rows. On touch screens each row shows its image inline instead. Fast and exact (about 240ms crossfade), no lag behind the pointer. Recipe: 'Index with preview, anchored to the row' in references/motion.md.
This is the signature move, the one loud moment. Around it the page still needs the rest of the motion layer (SKILL.md step 9): an entrance for the parts of every section, feedback on every control, and one moving picture. The inspector fails a page that barely moves. Recipes: references/motion.md.

## 10. Where the richness comes from
Light against dark, and the slow pacing.

## 11. How this direction goes wrong
Grey text on black fails contrast quickly: check it. No glow effects, no neon accent, no coloured light blobs. A dark page with a bright busy photograph is not this direction; choose low-key images. Dark does not mean empty: every screen still carries its content at a readable size, and running text is a light ink at 7:1 or more, not a dim grey.

## 12. Real references
Studied sites that work this way. Take the principle, never the specifics: no names, copy, images or signature assets of theirs.
- **IZANAMI** (Awwwards Site of the Day, studied score 7) `aww-izanami`
  Idea: Silence as design: one small sentence in fog, then a slow descent from mist into a dark glass pavilion.
  Hero: After a warm-black loader showing only one 18px serif sentence and a small counter, the hero is a full-bleed, desaturated grey-green image of pine forest dissolving into fog. The same sentence stays at 18px in white, left-aligned on a line about 17% from the left edge and exactly at vertical centre; it is the only text in the picture area. A small white seal-style logo sits top-left; a language switch and menu label in tiny tracked capitals sit top-right, each with a dot above. Along the bottom edge: copyright, two live city clocks, and a scroll cue at far right, all near 10px.
  Worth taking: Place a single 18px sentence at vertical centre on an offset left line over full-bleed fog, and let the absence of a headline be the statement.
- **Cipher** (Awwwards Site of the Day, studied score 7) `aww-cipher`
  Idea: The showreel as a slowly turning wreath: a dozen film stills overlap in an elliptical ring around a tiny emblem on a black screen, with almost no text.
  Hero: The first capture is the preloader: a black screen textured with a fine grid of grey dots that thins out towards the right, the wordmark top-left and a tiny copyright glyph dead centre. Once loaded, the black screen holds a ring of about fourteen landscape film stills, each roughly 10% of the viewport wide, overlapping one another around an ellipse about 35% of the width; the centre is empty except for a 20px white emblem. The wordmark, about 11% wide, stays top-left. Four 9px uppercase links are split: one at the horizontal centre, three at the far right. A partner mark sits bottom-right. There is no headline.
  Worth taking: Arrange work thumbnails as an overlapping elliptical ring with an empty centre, and bind scroll to rotation instead of vertical travel.
- **Léo Parpeix - Portfolio 2026** (Awwwards Site of the Day, studied score 7.6) `aww-leo-parpeix-portfolio-2026`
  Idea: A soft clay-rendered 3D world, a smiling flower in a sunlit gallery and a bee you can feed, carries the designer's personality while the type stays blunt and heavy.
  Hero: Opens on an empty off-white field with one object: a nearly square card about 28% of the viewport wide, centred, with rounded corners and tilted in 3D perspective, showing a grainy beige render of a smiling clay flower in front of a gridded window. There is no text at all. The card then scales up and flattens into a full-bleed 3D gallery room (flower on a plinth, pale wood floor, sky through tall windows) with a one-line 14px nav laid over it: name far left, role about one-eighth across, three links and a dots menu far right.
  Worth taking: Start with one small tilted card on an empty page and scale it into the full-bleed hero scene, so loader and hero are the same object.
Full notes on any of them: `node draw.mjs --site <id>`.

## Always
Headlines two lines or fewer at every width. No eyebrow labels, no monospace, no 01/02/03, no thin rules, no gradients, no shadows at rest,
no cream-and-terracotta, no violet. The frame: one h1 on the first screen that says what this is, equal outer margins, no
section huddled on one side with the other half empty, no blank stretch over half a screen, a heading on every section.
`inspect.mjs` measures all of it.

Saved to website/v4/.taste-skill/direction.json · reprint with `node draw.mjs --show`
