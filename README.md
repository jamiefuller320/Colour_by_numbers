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
2. **Quantize** — k-means clustering reduces the image to N colours (default 16).
3. **Outline** — Boundaries between colour regions become black lines on white. Connected regions large enough to colour receive the palette number for that colour.
4. **Legend** — A colour key lists each number with a swatch and hex value.

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
- Fewer colours (8–12) are easier for younger colourists; 16 suits more detail.
- Always respect copyright and licensing of source photos before publishing a book.
- Prefer images you own, public-domain sources, or material with a clear commercial licence.

## Tests

```bash
pytest
```

Tests cover quantization, outlining, and file export offline (no network).
