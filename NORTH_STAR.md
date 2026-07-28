# North star

Produce **copyright-safe colouring books** from a keyword or key phrase: varied, high-quality colour plates of one subject across aspects and scenes, each converted into a correctly numbered outline a person can colour to reconstruct the original image—plus full-colour front and back covers—assembled as a printable book.

Until now, work has mainly probed pipelines (search, stylize, Pollinations, palette, region size, outlines). From here, every change should move a measurable step toward that book outcome—not another isolated experiment.

## Product outcome

Given a **keyword or key phrase** (e.g. `pug`, `golden retriever in the park`, `spitfire`):

1. **Generate a set** of distinct colour illustrations of that subject in **different aspects and scenes** (poses, angles, settings)—enough variety for a book section, not a single one-off plate.
2. **Convert each plate** into an **interesting, correctly labelled colour-by-numbers outline** (and colour key) that a human can colour to **reconstruct the original image**.
3. **Produce full-colour front and back cover art** in the same visual family as the interior plates.
4. **Compile** interiors + covers into a **print-ready book format** (ordered pages, consistent size/margins, print resolution).
5. Do all of the above in a way that **does not infringe copyright**—prefer original generated / properly licensed art; never ship scraped proprietary photos as book content.

## Success criteria (definition of done for the product)

| Criterion | What “good” looks like |
|-----------|-------------------------|
| **Prompt → set** | One phrase yields a coherent multi-plate set (target: enough pages for a short book chapter), not a single image. |
| **Variety** | Plates differ in aspect and scene while remaining clearly the same subject. |
| **Fidelity of colouring** | Numbered outline + key let a careful colourist recover the intended flat colours of the colour plate. |
| **Label correctness** | Every colour block is numbered; numbers match the key; no orphan/mislabelled regions. |
| **Print quality** | A4 (or chosen trim) at usable DPI; colouring regions large enough to fill by hand; outlines readable, not muddy. |
| **Covers** | Front and back are full-colour, on-brand with the set, and print-ready. |
| **Book assembly** | Ordered PDF (or equivalent) with covers + interior plates + keys as designed. |
| **Copyright safety** | Book assets are generated or openly licensed with attribution where required; web photo search is reference-only / optional tooling, not the default publish path. |

## Design pillars

1. **Illustration-first, rights-safe** — Default publish path is original generation (or clearly licensed sources). Photo search may inform style or local stylize experiments; it is not the north-star supply for books.
2. **Set over single** — The unit of work is a **bookable set** (subject × aspects × scenes), not one successful plate.
3. **Colour plate ↔ outline contract** — The outline is a faithful, colourable encoding of the plate: shared palette, correct labels, reconstructible result.
4. **Human colourability** — Regions, line weight, and numbering must work on paper for a person with crayons/markers—not only look good on screen.
5. **Keyword as brief** — A short phrase drives subject, variety plan, covers, and book metadata end-to-end.

## Non-goals (for now)

- Photoreal colouring pages derived from arbitrary web photos as the primary book pipeline.
- Infinite style playgrounds without a path into a compiled book.
- Perfect ML “understanding” of every niche subject before sets and covers ship.
- Replacing human colouring with automated fill (the product is for people to colour).

## Near-term build sequence (toward the star)

Ordered so each step unlocks the next; probing continues only when it answers one of these:

1. **Rights-safe plate generator** — Reliable illustration backend + palette/region/outline rules for book interiors.
2. **Set planner** — From one phrase, propose N aspect/scene variants and generate the set consistently.
3. **Outline fidelity** — Guaranteed correct numbering and reconstructibility vs the colour plate.
4. **Cover generator** — Front/back full-colour art matching the set.
5. **Book compiler** — Layout, order, PDF export, print checklist.
6. **Publish guardrails** — Licence/provenance recorded per asset; block unsafe sources from book export.

## Workable progression framework

Use short gated phases. Exit a phase only when its gate passes; do not open parallel “option probes” unless they unblock the current gate.

### Phase A — Subject control (gate: no wrong-entity plates)

- Disambiguate ambiguous names (`spitfire` → WWII fighter aircraft, not a person/character).
- Category negatives (no people on vehicle plates).
- Manual spot-check: 10 generations across aircraft/animals with zero category errors.

### Phase B — Plate quality bar (gate: colourable reconstructible single plate)

- Lock backend(s) that meet quality (see models note below).
- Palette, region size, outline numbering stable for one subject.
- Pass/fail checklist: readable outline, every block numbered, hand-colourable regions, reconstructible vs colour plate.

### Phase C — Format brief from references (gate: written page/cover spec)

- Study **rights-safe** colour-by-numbers examples (own purchases, public-domain / explicitly licensed pages)—**not** scraped commercial books as training or copy targets.
- Extract a short **format brief**: typical page layout, subject fill, legend placement, cover formula, difficulty banding.
- Encode that brief as generation + layout constraints (this repo’s print rules), not as cloned artwork.

### Phase D — Set generation (gate: N varied plates, one phrase)

- Planner turns a keyword/phrase into aspect/scene slots (e.g. side view / takeoff / hangar).
- Generate the set with shared palette language and subject identity.
- Reject duplicates and off-brief plates.

### Phase E — Covers + book compile (gate: print-ready PDF)

- Front/back full-colour covers in the same family.
- Ordered interiors + keys + covers → PDF with trim/DPI checklist.
- Provenance log: every asset marked generated or licensed.

### Phase F — Harden & scale

- More categories, batch runs, weaker prompts still stay on-entity.
- Publish guardrails automated.

## Models, Cursor Pro+, and image quality

**Cursor Pro+ improves the coding agent in Cursor; it does not automatically upgrade Pollinations or other image APIs used by this app.**

Image quality depends on the **illustration backend** and its own plan/key:

| Backend | Notes |
|---------|--------|
| `pollinations` | Free/public; model choice (`flux`, `turbo`, …) is Pollinations-side, not Cursor-side. |
| `openai` | Needs `OPENAI_API_KEY`; quality follows your OpenAI image model access. |
| `local_stylize` | No diffusion model; stylizes a reference photo. |

Progression implication: pick **one primary rights-safe generator** in Phase B and optimize prompts/disambiguation against it; treat other backends as fallbacks, not a perpetual bake-off.

## Using existing colour-by-numbers pictures

**Yes, as a format teacher—carefully.** Search/browse (or buy) colour-by-numbers products to learn **conventions**: how much subject fills the page, how busy scenes are, legend style, cover layout, age/difficulty banding.

**No, as a content source to copy.** Do not scrape commercial pages into the generator or train on them. That fights the copyright north star. Capture findings as a written format brief (Phase C), then implement constraints in our pipeline.

## How to use this document

- New features and PRs should state which **success criterion**, **build-sequence step**, or **framework phase** they advance.
- If a change only “probes an option,” park it unless it reduces risk on the **current** phase gate.
- When trade-offs arise (e.g. more colours vs hand-colourability), prefer the north star: **a human-colourable, reconstructible, rights-safe book**.
