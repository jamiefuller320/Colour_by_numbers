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

## Visual end goal (interior plate quality)

The long-term look we are aiming for is an **adult “vibrant paint-by-numbers” portrait kit**: dense interlocking flat colour wedges (mosaic fur / form), a **full crayon budget (~24–32 solids)**, cool teal/blue accents in shadows mixed with warm mid-tones, bold black outlines, and optional abstract colour-block backgrounds — *not* a handful of large cel blobs on empty white.

Study commercial vibrant kits as **format teachers** (composition, value mosaic, region density). **Do not** scrape or clone any one product’s art.

| Band | Role today |
|------|------------|
| `simple` / `standard` | Current shippable Phase B path (kids / hand-colourable, 8–16 colours, ≥8mm) |
| `vibrant` | End-goal band — denser regions (≥4mm), up to 32 colours, cool shadows, mosaic prompts |

`--style vibrant` selects the end-goal knobs. Closing the gap to reference-quality plates is an ongoing Phase B+/C refinement, not a one-shot switch.

## Success criteria (definition of done for the product)

| Criterion | What “good” looks like |
|-----------|-------------------------|
| **Prompt → set** | One phrase yields a coherent multi-plate set (target: enough pages for a short book chapter), not a single image. |
| **Variety** | Plates differ in aspect and scene while remaining clearly the same subject. |
| **Fidelity of colouring** | Numbered outline + key let a careful colourist recover the intended flat colours of the colour plate. |
| **Label correctness** | Every colour block is numbered; numbers match the key; no orphan/mislabelled regions. |
| **Print quality** | A4 (or chosen trim) at usable DPI; colouring regions large enough to fill by hand; outlines readable, not muddy. |
| **Vibrant plate bar** | Adult-band plates read as dense value mosaics with rich palettes (not 3–8 flat blobs); subject remains unmistakable from the outline. |
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

**Global fix (preferred over per-name patches):**

1. **Category hypernym** — always name the kind (`pug` → `pug dog`, `rose` → `rose flower`).
2. **Kind frame** — prompt lead-in: “subject kind: aircraft/dog/…; depict only this kind.”
3. **Category negatives** — block the usual failure mode (people/characters on non-person plates).
4. **Hard overrides** — only for stubborn collisions (`spitfire` → Supermarine Spitfire WWII fighter aircraft).

Do not grow an endless name list; add an override only when hypernym + negatives still fail.

- Manual spot-check: 10 generations across aircraft/animals/flowers with zero wrong-entity plates.

### Next step (recommended now)

Close the gap from Phase B `standard` plates toward the **vibrant** end-goal bar
(`--style vibrant`: denser mosaic, fuller palette, cool shadows), using critique
exports and rights-safe reference study. In parallel, Phase D sets remain available;
Phase E (covers + book compile) follows once interiors hit the quality bar you want
to publish.

### Phase B — Plate quality bar (gate: colourable reconstructible single plate)

**Locked choices**

| Item | Value |
|------|--------|
| Primary backend | `fal` (`fal-ai/flux/schnell`) — rights-safe generation via fal.ai; needs `FAL_KEY` |
| Fallback | `local_stylize` / `openai` / legacy `pollinations` (optional) |
| Min colourable block | **8mm × 8mm** on A4 |
| Palette budget | **8–16** colours |

**Checklist** (implemented in `colour_by_numbers.quality` / `scripts/phase_b_plate_check.py`):

- Palette within 8–16 colours
- Every connected colour block listed and numbered; numbers ⊆ colour key
- Outline labels + palette reconstruct the colour plate
- Illustration agrees with the simplified plate after palette mapping
- No colourable blocks below the 8mm A4 floor
- Outline has readable ink; legend present

```bash
# Offline gate (synthetic plate, no network)
python scripts/phase_b_plate_check.py --offline --require

# Live primary backend
python scripts/phase_b_plate_check.py --query dogs --type pug --require
```

CLI: `--illustrate` defaults to fal (`FAL_KEY`); add `--require-quality` to fail on checklist miss.

**Subject-recognition feedback loop** (Phase B support for weak entities like Spitfire):

After each generation the pipeline can ask:

1. Is this recognisable as the requested subject?
2. How should the generation prompt improve?

Critics: `rules` (offline feature cues), `openai` (vision, needs `OPENAI_API_KEY`), `human` (interactive). Failed → revise prompt → retry (default 3 attempts). Accepted revisions append to `data/subject_lessons.jsonl` and seed later runs of the same subject.

```bash
# Dry-run: show seeded prompt + rules critique (no network)
python scripts/subject_feedback_loop.py --query aircraft --type spitfire --dry-run

# Live loop on fal (needs FAL_KEY)
python scripts/subject_feedback_loop.py --query aircraft --type spitfire

# Or via CLI
colour-by-numbers --query aircraft --type spitfire --illustrate \
  --subject-feedback --critique-mode rules --output output
```

This does **not** retrain Flux weights. It improves *our* prompts and stores lessons so agent/human feedback compounds. Vision (`openai`) or human critique is what actually judges pixels; `rules` strengthens known hard cases (elliptical wings, breed features, no-people cues) before and between retries.

### Phase C — Format brief from references (gate: written page/cover spec)

**Status:** brief landed in [`FORMAT_BRIEF.md`](FORMAT_BRIEF.md); composition checks encoded in `colour_by_numbers.quality`.

- Study **rights-safe** colour-by-numbers examples (own purchases, public-domain / explicitly licensed pages)—**not** scraped commercial books as training or copy targets.
- Extract a short **format brief**: typical page layout, subject fill, legend placement, cover formula, difficulty banding.
- Encode that brief as generation + layout constraints (this repo’s print rules), not as cloned artwork.

**Encoded composition metrics** (defaults; tune only with evidence):

| Check | Threshold |
|-------|-----------|
| Colourable fill (not ink/line) | ≥ **90%** of page |
| Max single colour share | ≤ **50%** of page |
| Subject fill (non-background colours) | ≥ **50%** of page |

```bash
python scripts/phase_b_plate_check.py --offline --require
```

### Phase D — Set generation (gate: N varied plates, one phrase)

**Status:** planner + set runner landed (`set_plan` / `set_generate`).

- Planner turns a keyword/phrase into aspect/scene slots (e.g. side view / takeoff / hangar).
- Generate the set with shared palette language and subject identity.
- Reject duplicates and off-brief plates (near-duplicate dHash + per-plate B/C checklist).

```bash
# Plan only (no network)
python scripts/phase_d_set_generate.py --query aircraft --type spitfire \
  --set-size 6 --plan-only

# Live set via CLI
colour-by-numbers --query dogs --type pug --illustrate \
  --set-size 4 --seed 100 --output output/pug-set
```

Set gate (`evaluate_set_quality`): enough accepted plates, unique aspect/scene
plan, each accepted plate passes Phase B/C, no near-duplicates, shared subject
label. Manifest: `plan.json` + `manifest.json` under the output directory.

### Phase E — Covers + book compile (gate: print-ready PDF)

- Front/back full-colour covers in the same family.
- Ordered interiors + keys + covers → PDF with trim/DPI checklist.
- Provenance log: every asset marked generated or licensed.

### Phase F — Harden & scale

- More categories, batch runs, weaker prompts still stay on-entity.
- Publish guardrails automated.

## Models, Cursor Pro+, and image quality

**Cursor Pro+ improves the coding agent in Cursor; it does not automatically upgrade fal.ai or other image APIs used by this app.**

Image quality depends on the **illustration backend** and its own plan/key:

| Backend | Notes |
|---------|--------|
| `fal` | **Primary.** Flux via fal.ai (`FAL_KEY`). Pay-as-you-go; production path for sets/books. |
| `openai` | Optional; needs `OPENAI_API_KEY`. |
| `pollinations` | Legacy fallback; anonymous tier is unreliable (Pollen/auth). |
| `local_stylize` | No diffusion model; stylizes a reference photo (not the publish default). |

**GitHub Pages** is a plate/outline **viewer** only (upload → palette/outline). Generation stays in Streamlit/CLI so API keys never sit in the browser.

Progression implication: optimize prompts/disambiguation against **fal**; treat other backends as fallbacks, not a perpetual bake-off.

## Using existing colour-by-numbers pictures

**Yes, as a format teacher—carefully.** Search/browse (or buy) colour-by-numbers products to learn **conventions**: how much subject fills the page, how busy scenes are, legend style, cover layout, age/difficulty banding.

**No, as a content source to copy.** Do not scrape commercial pages into the generator or train on them. That fights the copyright north star. Capture findings as a written format brief (Phase C), then implement constraints in our pipeline.

## How to use this document

- New features and PRs should state which **success criterion**, **build-sequence step**, or **framework phase** they advance.
- If a change only “probes an option,” park it unless it reduces risk on the **current** phase gate.
- When trade-offs arise (e.g. more colours vs hand-colourability), prefer the north star: **a human-colourable, reconstructible, rights-safe book**.
