# Format brief (Phase C)

Rights-safe **layout and colourability conventions** for interior plates and covers.
This is a product spec for our generator—not scraped art, and not a licence to copy
commercial colour-by-numbers pages.

**How to use references:** browse or buy examples, or use public-domain / explicitly
licensed colouring pages, and jot conventions here. Do **not** ingest commercial
pages into the pipeline or treat them as training/copy targets.

The “by numbers” layer only adds a colour key to a flat colouring pattern. Ideal
output is judged first as a **good colouring plate**, then as a correctly labelled
reconstruction target.

## Interior plate (A4 portrait / square export)

| Spec | Target | Why |
|------|--------|-----|
| Subject fill | Subject (non-background colours) covers **≥ ~50%** of the page | Avoids tiny floating subjects on empty fields |
| Colourable fill | **≥ 90%** of pixels are fillable (numbered) regions | Ink/line detail stays a guide, not most of the page |
| Dominant colour | No single palette colour exceeds **~50%** of the page | Blocks empty sky/background floods and one-blob plates |
| Colour budget | **8–16** flat colours (target **16**) | Enough interest, still hand-colourable. Extract adaptively from the generated plate, then snap to distinct crayons — do **not** pre-thin the fixed 32-set (that collapses many fal plates to 3–5 fills). |
| Region size | Every colourable block **≥ 8mm × 8mm** on A4 | Fits a marker tip; finer marks become black line |
| Composition | One clear subject, centred or slight bias; simple or no scene clutter | Readable at colouring-book distance |
| Background | Flat light field (off-white / pale), not textured photo ground | High contrast; easy first fill |
| Outline | Firm black ink; thin enough not to eat small regions | Reconstructible silhouette |
| Legend | Colour key on the same export (or facing page later) | Numbers ⊆ key; swatches match fills |
| Difficulty | “Fine” default = more regions within the 8mm floor; “simple” = fewer larger fills | Band later for age ranges once sets exist |

### What “good” looks like (checklist language)

1. A child can tell what the subject is from the outline alone.
2. Most of the sheet is something to colour, not black lace.
3. Background does not dominate; the subject owns the page.
4. Colouring the numbers rebuilds the flat colour plate.

## Covers (deferred detail — Phase E)

| Spec | Target |
|------|--------|
| Front | Full-colour hero of the subject; title area reserved; same family as interiors |
| Back | Related full-colour motif or secondary pose; blurb/barcode zone reserved |
| Palette | Shared language with the interior set (not a one-off style) |

Exact cover layout grids land with the book compiler; this brief only locks the
visual family rules.

## Difficulty banding

| Band | Intent | Knobs (`--style`) | Notes |
|------|--------|-------------------|-------|
| Simple | Fewer, larger fills | ~10 colours, ≥10mm | Younger colourists |
| Standard (default) | Hand-colourable A4 | 8–16 colours (target 16), ≥8mm, book palette | Current Phase B gate |
| Vibrant (end goal) | Adult paint-by-numbers mosaic | ~24–32 colours, ≥4mm, adaptive palette, cool shadow accents | Dense interlocking wedges; optional abstract colour-block background; study vibrant portrait kits as format teachers — do not copy commercial art |

**Vibrant checklist language** (aspirational):

1. The subject reads instantly from the outline alone.
2. Fur / form is a **value mosaic** of many interlocking fills, not 3–6 large blobs.
3. Warm mid-tones and **cool shadow accents** both appear.
4. Background may be abstract colour blocks; it must not flatten into empty white that starves the palette.
5. Colouring the numbers still rebuilds the flat plate.

## Set variety (Phase D)

One keyword/phrase → N plates that keep **the same subject identity** but change
**aspect** (pose/angle) and **scene** (simple setting). Scene cues stay light so
subject-fill ≥50% remains achievable. Reject near-duplicate plates and any plate
that fails the interior metrics above.

## Encoded metrics (this repo)

Implemented in `colour_by_numbers.quality` alongside the Phase B reconstructibility gate:

| Check | Default threshold |
|-------|-------------------|
| `colourable_fill_fraction` | ≥ **0.90** |
| `max_colour_share` | ≤ **0.50** |
| `subject_fill_fraction` | ≥ **0.50** (share of pixels not the border-dominant background colour) |

Tune only with evidence from rights-safe references or failed production plates;
do not tighten to vanity numbers (e.g. “max 10% one colour”) that break real subjects.

## Out of scope for this brief

- Copying characters, logos, or distinctive commercial compositions
- Photoreal source photos as the publish path
- Per-brand mimicry of a competitor’s line weight or legend artwork
