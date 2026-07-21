# Colour by Numbers

Search the web for images by description (for example `aircraft` or `dogs`), reduce them to a limited palette (default **16 colours**), and export a numbered outline page suitable for colouring books.

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

1. **Search** — Finds candidate photos from a text query using Openverse (open licences) first, then Wikimedia Commons, then DuckDuckGo as a fallback. No API keys are required. You can also supply a local file.
2. **Subject engine (default: dual)** — rembg / U²-Net finds the subject, crops so its bounding box fills **80% of the frame**, then applies **fine** cleanup to the subject and **light** cleanup to the background under a shared palette of at most **16 colours**.
3. **Cartoon prefilter + quantize** — Differential blur (gentler on the subject) then median-cut to ≤16 colours.
4. **Dual simplify** — Fine region absorption on subject pixels; light (stronger) absorption on background pixels; seam softened at the mask edge.
5. **Outline + legend** — Numbered regions and a colour key.

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

Compare original vs full-frame fine vs dual (80% fill, fine/light):

```bash
python scripts/make_demo_spread.py \
  --aircraft output/aircraft_source.png \
  --dogs output/dogs_source.png \
  --output output/demo
```

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
- `--complexity fine|light|medium` — uniform complexity when not using dual
- `--max-regions 36` — hard cap on connected regions
- `--pick N` — choose the Nth search result (0-based)
- `--max-size 900` — longest edge before processing
- `--line-width 2` — thicker outlines
- `--stem my_page` — output filename prefix

## Web UI

```bash
streamlit run app.py
```

Search or upload an image, adjust the palette size, then download the outline, legend, or full printable page.

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
