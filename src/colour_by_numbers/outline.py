"""Build numbered outline drawings from quantized images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


@dataclass(frozen=True)
class Region:
    """A connected colour region large enough to receive a number."""

    colour_index: int  # 0-based palette index
    number: int  # 1-based colour number shown to the user
    centroid: tuple[float, float]  # (x, y)
    area: int


@dataclass(frozen=True)
class OutlinePage:
    """Finished colour-by-numbers artwork and supporting assets."""

    outline: Image.Image
    legend: Image.Image
    regions: list[Region]
    palette: np.ndarray
    colour_numbers: list[int]


def _edges_from_labels(labels: np.ndarray) -> np.ndarray:
    """Return a boolean mask of boundaries between differently labelled pixels."""
    up = labels != np.roll(labels, 1, axis=0)
    left = labels != np.roll(labels, 1, axis=1)
    edges = up | left
    edges[0, :] = False
    edges[:, 0] = False
    return edges


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in (
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "Arial.ttf",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _find_regions(
    labels: np.ndarray,
    *,
    min_area: int,
) -> list[tuple[int, int, tuple[float, float], int]]:
    """Find connected components per colour label.

    Returns tuples of (colour_index, component_label_local, centroid_xy, area).
    """
    found: list[tuple[int, int, tuple[float, float], int]] = []
    structure = np.ones((3, 3), dtype=bool)

    for colour_index in np.unique(labels):
        mask = labels == colour_index
        labeled, count = ndimage.label(mask, structure=structure)
        if count == 0:
            continue
        for comp_id in range(1, count + 1):
            component = labeled == comp_id
            area = int(component.sum())
            if area < min_area:
                continue
            ys, xs = np.nonzero(component)
            centroid = (float(xs.mean()), float(ys.mean()))
            found.append((int(colour_index), comp_id, centroid, area))
    return found


def simplify_labels(labels: np.ndarray, *, smooth_radius: int = 2) -> np.ndarray:
    """Merge speckles so outlines stay simple enough for colouring.

    Uses majority-filter style smoothing: each pixel becomes the most common
    label in its neighbourhood, which removes tiny islands without inventing
    new palette indices.
    """
    if smooth_radius <= 0:
        return labels
    size = smooth_radius * 2 + 1
    best_votes = np.zeros(labels.shape, dtype=np.float32)
    best_labels = labels.astype(np.int32, copy=True)
    for value in np.unique(labels):
        mask = (labels == value).astype(np.float32)
        votes = ndimage.uniform_filter(mask, size=size, mode="nearest")
        better = votes > best_votes
        best_labels = np.where(better, value, best_labels)
        best_votes = np.where(better, votes, best_votes)
    return best_labels.astype(np.int32)


def build_outline_page(
    labels: np.ndarray,
    palette: np.ndarray,
    *,
    min_region_area: int | None = None,
    line_width: int = 1,
    number_font_scale: float = 1.0,
    smooth_radius: int = 2,
    max_numbers_per_colour: int = 8,
) -> OutlinePage:
    """Convert a palette-indexed image into a numbered outline page + legend.

    Each palette colour is assigned a stable number (1..N). Numbers are drawn
    inside sufficiently large connected regions of that colour.
    """
    labels = simplify_labels(labels, smooth_radius=smooth_radius)
    height, width = labels.shape
    if min_region_area is None:
        # Ignore speckles; scale with image size.
        min_region_area = max(80, int(width * height * 0.001))

    edges = _edges_from_labels(labels)
    if line_width > 1:
        edges = ndimage.binary_dilation(edges, iterations=line_width - 1)

    outline = Image.new("RGB", (width, height), "white")
    outline_arr = np.asarray(outline).copy()
    outline_arr[edges] = (0, 0, 0)
    outline = Image.fromarray(outline_arr, mode="RGB")

    draw = ImageDraw.Draw(outline)
    font_size = max(10, int(min(width, height) * 0.028 * number_font_scale))
    font = _load_font(font_size)

    raw_regions = _find_regions(labels, min_area=min_region_area)
    # Colour numbers are 1-based palette indices.
    colour_numbers = list(range(1, len(palette) + 1))
    regions: list[Region] = []

    # Keep only the largest regions per colour to avoid an unreadable page.
    per_colour: dict[int, list[tuple[int, int, tuple[float, float], int]]] = {}
    for item in raw_regions:
        per_colour.setdefault(item[0], []).append(item)
    selected: list[tuple[int, int, tuple[float, float], int]] = []
    for colour_index, items in per_colour.items():
        items_sorted = sorted(items, key=lambda row: row[3], reverse=True)
        selected.extend(items_sorted[:max_numbers_per_colour])

    for colour_index, _comp, centroid, area in sorted(
        selected, key=lambda item: item[3], reverse=True
    ):
        number = colour_index + 1
        x, y = centroid
        text = str(number)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = int(np.clip(x - tw / 2, 1, width - tw - 1))
        ty = int(np.clip(y - th / 2, 1, height - th - 1))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((tx + dx, ty + dy), text, fill="white", font=font)
        draw.text((tx, ty), text, fill="black", font=font)
        regions.append(
            Region(
                colour_index=colour_index,
                number=number,
                centroid=(x, y),
                area=area,
            )
        )

    legend = build_legend(palette, colour_numbers, swatch_size=36)
    return OutlinePage(
        outline=outline,
        legend=legend,
        regions=regions,
        palette=palette,
        colour_numbers=colour_numbers,
    )


def build_legend(
    palette: np.ndarray,
    colour_numbers: list[int],
    *,
    swatch_size: int = 36,
    columns: int = 4,
) -> Image.Image:
    """Create a printable colour key mapping numbers to RGB swatches."""
    n = len(palette)
    columns = max(1, min(columns, n))
    rows = int(np.ceil(n / columns))
    pad = 12
    cell_w = swatch_size + 110
    cell_h = swatch_size + 16
    width = columns * cell_w + pad * 2
    height = rows * cell_h + pad * 2 + 28

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(18)
    body_font = _load_font(14)
    draw.text((pad, 8), "Colour key", fill="black", font=title_font)

    for i, (number, colour) in enumerate(zip(colour_numbers, palette)):
        row, col = divmod(i, columns)
        x = pad + col * cell_w
        y = pad + 28 + row * cell_h
        rgb = tuple(int(c) for c in colour)
        draw.rectangle(
            [x, y, x + swatch_size, y + swatch_size],
            fill=rgb,
            outline=(0, 0, 0),
        )
        hex_colour = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        draw.text(
            (x + swatch_size + 8, y + swatch_size / 2 - 8),
            f"{number}: {hex_colour}",
            fill="black",
            font=body_font,
        )
    return image


def composite_page(
    outline: Image.Image,
    legend: Image.Image,
    *,
    gap: int = 24,
) -> Image.Image:
    """Stack the outline above the colour key into one printable page."""
    width = max(outline.width, legend.width)
    height = outline.height + gap + legend.height
    page = Image.new("RGB", (width, height), "white")
    page.paste(outline, ((width - outline.width) // 2, 0))
    page.paste(legend, ((width - legend.width) // 2, outline.height + gap))
    return page
