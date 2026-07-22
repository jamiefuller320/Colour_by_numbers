# Colour by Numbers

Search the web for images by description (for example `aircraft` or `dogs`), map them onto a **standardised 32-colour** colouring palette, and export a numbered outline page suitable for colouring books.

## What it produces

For each source image the tool writes:

| File | Purpose |
|------|---------|
| `*_source.png` | Original (or downloaded) image |
| `*_prepared.png` | Subject plate after isolation + crop (when enabled) |
| `*_quantized.png` | Image reduced to the chosen palette |
| `*_outline.png` | White page with black region outlines and numbers |
| `*_legend.png` | Colour key mapping each number to its RGB swatch |
| `*_page.png` | Outline and legend stacked for printing |

## How it works

1. **Type discovery** — Broad queries such as `dogs` or `aircraft` are narrowed to a concrete type (e.g. pug, golden retriever, biplane) before photo search, so pages stay recognisable rather than generic.
2. **Illustration (optional)** — With `--illustrate`, gather type-specific reference photos and build a flat colouring-book illustration (local stylize today; optional OpenAI backend when `OPENAI_API_KEY` is set), then convert.
3. **Search** — Finds candidate photos for that specific type using Openverse (open licences) first, then Wikimedia Commons, then DuckDuckGo as a fallback. No API keys are required. You can also supply a local file.
4. **Subject engine (default: dual)** — rembg / U²-Net finds the subject, **colour-refines** the silhouette using subject vs background Lab contrast, maps a **firm binary mask** onto the full-resolution photo, and crops so the subject fills **80% of the frame**.
5. **Contrast-aware search** — web queries are biased toward clear subject/background photos; downloads are scored by centre-vs-border colour contrast.
6. **A4 print filter (CLI/UI default: 150 DPI)** — reject plates that would print softer than the DPI floor on A4.
7. **Standard 32-colour palette** — pixels map onto a fixed colouring-book set (optional free median-cut). Touching sections closer than a Lab ΔE floor are merged so adjacent paints stay distinct.
8. **Dual simplify** — Fine cleanup on the subject, light on the background; firm borders; no seam softening.
9. **Outline + legend** — Numbered regions and a colour key (colour plates are the current focus; outline demos are paused).

### Type discovery

```bash
# List ranked specific types for a broad category
colour-by-numbers --query dogs --list-types

# Pick a type explicitly, then search/convert
colour-by-numbers --query dogs --type "golden retriever" --output output

# Or take the top-ranked discovered type (default)
colour-by-numbers --query dogs --output output
```

In the Streamlit UI, searching `dogs` shows a breed shortlist first; choosing one searches photos for that breed only.

### Illustration-first generation

```bash
# Discover type → reference photo → flat illustration → colour-by-numbers
colour-by-numbers --query dogs --type "pug" --illustrate --output output

# Optional OpenAI Images backend (requires OPENAI_API_KEY)
colour-by-numbers --query dogs --type "pug" --illustrate \
  --illustration-backend openai --output output
```

Local stylize isolates the subject, maps fills onto the standard palette, flattens the background, and draws a firm ink outline. True text-to-image generation is available via the OpenAI backend when a key is present.

### Subject engine

| Mode | Behaviour |
|------|-----------|
| `dual` (default) | Mask + 80% fill crop + fine subject / light background |
| `isolate` | Flat background cut-out + 80% fill crop |
| `off` | Full frame, uniform complexity |

```bash
colour-by-numbers --input plane.jpg --subject dual --subject-fill 0.80
colour-by-numbers --input plane.jpg --subject isolate --complexity fine
colour-by-numbers --input plane.jpg --subject off --complexity fine
```

### Complexity presets

Default is **`fine`** (best results so far once the subject is isolated):

| Preset | Intent |
|--------|--------|
| `raw` | 16-colour quantize only |
| `fine` | Default — light cleanup |
| `light` | A little more merging |
| `medium` | More merging |
| `simple` | Strongest merge |

### Demo spreads

Colour-only comparison spreads (outline demos paused while colour plates are refined):

```bash
python scripts/make_demo_spread.py \
  --aircraft output/aircraft_source.png \
  --dogs output/dogs_source.png \
  --output output/demo
```

### A4 resolution & firm borders

```bash
# Require ~150 DPI when the subject plate is printed to fill A4 (default on CLI)
colour-by-numbers --input photo.jpg --min-a4-dpi 150

# Disable the filter
colour-by-numbers --input photo.jpg --no-a4-filter

# Soft alpha edges instead of a hard silhouette mask
colour-by-numbers --input photo.jpg --no-firm-border
```

Search results with known width/height below the DPI floor are filtered out; larger images are preferred.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Or with requirements only:

```bash
pip install -r requirements.txt
pip install -e .
```

## CLI

Search the web and generate a page:

```bash
colour-by-numbers --query "aircraft" --output output --colours 16
```

Use a local image:

```bash
colour-by-numbers --input photo.jpg --output output --colours 12
```

Useful options:

- `--subject dual|isolate|off` — subject engine (default: dual)
- `--subject-fill 0.80` — subject bbox fill of the frame after crop
- `--subject-complexity fine` / `--background-complexity light` — dual-pass presets
- `--palette-mode standard|free` — fixed 32-colour set (default) or adaptive median-cut
- `--colours 32` — how many paints to use from the palette
- `--min-adjacent-delta-e 18` — merge small muddy touches closer than this Lab ΔE
- `--colour-refine` / `--no-colour-refine` — snap silhouette by subject/bg colour
- `--firm-border` / `--no-firm-border` — hard binary subject silhouette (default: on)
- `--min-a4-dpi 150` / `--no-a4-filter` — A4 print-resolution gate
- `--complexity fine|light|medium` — uniform complexity when not using dual
- `--max-regions 36` — hard cap on connected regions
- `--pick N` — choose the Nth search result (0-based)
- `--max-size 900` — longest edge for colour processing (after native A4 check)
- `--line-width 2` — thicker outlines
- `--stem my_page` — output filename prefix

The Streamlit UI exposes these plus per-zone blur / region-simplify overrides.

## Web UI

```bash
streamlit run app.py
```

Search or upload an image, then adjust palette, subject engine, fill, firm borders, A4 DPI floor, dual complexities, and optional blur/region overrides in the sidebar. Colour plates are shown first; outline download remains available in an expander.

## Python API

```python
from colour_by_numbers import create_colour_by_numbers
from colour_by_numbers.pipeline import create_from_query

result = create_from_query("dogs", n_colours=16, complexity="fine", subject_mode="isolate")
result.save("output", stem="dogs")

# Or from an existing Pillow image
from PIL import Image
image = Image.open("photo.jpg")
result = create_colour_by_numbers(image, n_colours=16, subject_mode="isolate")
```

## Tips for colouring-book pages

- Clear single subjects work best; the subject engine helps when the background is sky or clutter.
- Default is `--subject dual` (80% fill, fine subject / light background, ≤16 colours).
- Fewer colours (8–12) are easier for younger colourists; 16 suits more detail.
- Always respect copyright and licensing of source photos before publishing a book.
- Prefer images you own, public-domain sources, or material with a clear commercial licence.

## Tests

```bash
pytest
```

Offline tests cover quantization, outlining, and export. Subject isolation demos need `rembg` (downloaded on first use).
