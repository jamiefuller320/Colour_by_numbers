# Colour by Numbers

Search the web for images by description (for example `aircraft` or `dogs`), reduce them to a limited palette (default **16 colours**), and export a numbered outline page suitable for colouring books.

## What it produces

For each source image the tool writes:

| File | Purpose |
|------|---------|
| `*_source.png` | Original (or downloaded) image |
| `*_quantized.png` | Image reduced to the chosen palette |
| `*_outline.png` | White page with black region outlines and numbers |
| `*_legend.png` | Colour key mapping each number to its RGB swatch |
| `*_page.png` | Outline and legend stacked for printing |

## How it works

1. **Search** — Finds candidate photos from a text query using Openverse (open licences) first, then Wikimedia Commons, then DuckDuckGo as a fallback. No API keys are required. You can also supply a local file.
2. **Cartoon prefilter + quantize** — The image is downscaled to a small “structure” canvas, soft-blurred to kill photo grain, then median-cut reduced to N colours (default 16).
3. **Simplify regions** — On that structure canvas the pipeline:
   - majority-filters labels
   - morphologically opens/closes each colour (removes slivers, fills pinholes)
   - absorbs regions below a minimum area into the neighbour with the longest shared border
   - absorbs regions that are too thin to colour (inscribed diameter test)
   - enforces a hard maximum region count
   - Gaussian-smooths boundaries, then re-absorbs anything that fractured
4. **Upsample + outline** — Simplified labels are nearest-neighbour scaled to print size, lightly re-smoothed, and stroked as black outlines on white. Every surviving region gets its colour number at an interior point.
5. **Legend** — A colour key lists each number with a swatch and hex value.

### Complexity presets

Centred on **`light`** (preferred for photos):

| Preset | Intent |
|--------|--------|
| `raw` | 16-colour quantize only (no region merging) |
| `fine` | Slightly less cleanup than light (−) |
| `light` | Default — gentle cleanup |
| `medium` | Slightly more cleanup than light (+) |
| `simple` | Stronger merge — fewer, larger regions |

Aliases: `detailed` → `light`, `balanced` → `medium`.

### Demo spreads

Compare the original with 16-colour samples just below / at / above light:

```bash
python scripts/make_demo_spread.py \
  --aircraft output/aircraft_source.png \
  --dogs output/dogs_source.png \
  --output output/demo
```

This writes `*_original.png`, `*_spread_colours.png`, and `*_spread_outlines.png`
(panels: original · fine · light · medium).

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

- `--complexity fine|light|medium` — region simplification (default: light)
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

result = create_from_query("dogs", n_colours=16)
result.save("output", stem="dogs")

# Or from an existing Pillow image
from PIL import Image
image = Image.open("photo.jpg")
result = create_colour_by_numbers(image, n_colours=16)
```

## Tips for colouring-book pages

- Simple subjects with clear shapes (animals, vehicles, landmarks) work best.
- Default `--complexity light` suits most photos; try `fine` / `medium` for small adjustments; use `raw` to inspect unmerged 16-colour output.
- Fewer colours (8–12) are easier for younger colourists; 16 suits more detail.
- Always respect copyright and licensing of source photos before publishing a book.
- Prefer images you own, public-domain sources, or material with a clear commercial licence.

## Tests

```bash
pytest
```

Tests cover quantization, outlining, and file export offline (no network).
