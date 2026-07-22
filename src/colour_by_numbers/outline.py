"""Build numbered outline drawings from quantized images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .quantize import upsample_labels
from .simplify import (
    SimplificationStats,
    absorb_small_regions,
    absorb_thin_regions,
    count_regions,
    enforce_colourable_blocks,
    enforce_min_brush_stroke,
    merge_adjacent_same_colour,
    normalize_specular_highlights,
    simplify_labels,
    smooth_boundaries,
)


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
    labels: np.ndarray
    simplification: SimplificationStats | None = None
    detail_ink: np.ndarray | None = None


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
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _region_centroid(component: np.ndarray) -> tuple[float, float]:
    """Pick a label anchor inside the region (distance-transform peak).

    Centroids of crescent/U shapes can fall outside the fill; the darkest
    point of the distance transform is guaranteed to lie inside.
    """
    # Distance to background — peak is a safe interior point.
    distance = ndimage.distance_transform_edt(component)
    peak = np.unravel_index(int(np.argmax(distance)), distance.shape)
    y, x = float(peak[0]), float(peak[1])
    return x, y


def _find_regions(labels: np.ndarray) -> list[Region]:
    """One numbered region per remaining connected component."""
    structure = np.ones((3, 3), dtype=bool)
    found: list[Region] = []

    for colour_index in np.unique(labels):
        mask = labels == colour_index
        labeled, count = ndimage.label(mask, structure=structure)
        if count == 0:
            continue
        areas = np.bincount(labeled.ravel())
        for comp_id in range(1, count + 1):
            component = labeled == comp_id
            area = int(areas[comp_id])
            centroid = _region_centroid(component)
            found.append(
                Region(
                    colour_index=int(colour_index),
                    number=int(colour_index) + 1,
                    centroid=centroid,
                    area=area,
                )
            )
    return found


def _font_size_for_region(
    area: int,
    *,
    base_size: int,
    min_size: int = 7,
) -> int:
    """Scale number glyphs down for small colour blocks so every block fits one."""
    side = max(1.0, float(area) ** 0.5)
    # Keep the glyph roughly inside ~45% of the region's inscribed-ish width.
    sized = int(side * 0.45)
    return max(min_size, min(base_size, sized))


def _draw_region_number(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    xy: tuple[float, float],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    width: int,
    height: int,
) -> None:
    """Paint a haloed number centred on ``xy``, clipped to the plate."""
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = int(np.clip(x - tw / 2, 1, max(1, width - tw - 1)))
    ty = int(np.clip(y - th / 2, 1, max(1, height - th - 1)))
    for dx, dy in (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
    ):
        draw.text((tx + dx, ty + dy), text, fill="white", font=font)
    draw.text((tx, ty), text, fill="black", font=font)


def build_outline_page(
    labels: np.ndarray,
    palette: np.ndarray,
    *,
    min_region_area: int | None = None,
    max_regions: int | None = None,
    line_width: int = 1,
    number_font_scale: float = 1.0,
    smooth_radius: int = 3,
    morph_radius: int = 2,
    boundary_sigma: float = 1.25,
    output_size: tuple[int, int] | None = None,
    simplify: bool = True,
    min_adjacent_delta_e: float = 18.0,
    min_thickness: float | None = None,
    min_width_px: int | None = None,
    min_height_px: int | None = None,
    min_region_mm: float | None = None,
    number_all_regions: bool = True,
    detail_ink: np.ndarray | None = None,
) -> OutlinePage:
    """Convert a palette-indexed image into a numbered outline page + legend.

    Colourable blocks must be at least ``min_region_mm`` wide **and** high when
    printed on A4 (or the explicit ``min_width_px`` / ``min_height_px``). Each
    block must fit a circular colouring tip of that diameter. Undersized
    features become black line detail instead of numbered fills.
    """
    stats: SimplificationStats | None = None
    working_labels = labels
    working_palette = palette
    detail = (
        np.zeros(labels.shape, dtype=bool)
        if detail_ink is None
        else detail_ink.astype(bool, copy=True)
    )

    if simplify:
        working_labels, working_palette, stats = simplify_labels(
            labels,
            palette,
            min_region_area=min_region_area,
            max_regions=max_regions,
            smooth_radius=smooth_radius,
            morph_radius=morph_radius,
            boundary_sigma=boundary_sigma,
            min_adjacent_delta_e=min_adjacent_delta_e,
            min_thickness=min_thickness,
        )
    else:
        before = count_regions(labels)
        stats = SimplificationStats(
            regions_before=before,
            regions_after=before,
            min_region_area=min_region_area or 0,
            smooth_radius=0,
            passes=0,
        )

    if output_size is not None:
        source_h, source_w = working_labels.shape
        working_labels = upsample_labels(working_labels, output_size)
        if detail.shape != working_labels.shape:
            detail_img = Image.fromarray(detail.astype(np.uint8) * 255, mode="L")
            detail_img = detail_img.resize(output_size, Image.Resampling.NEAREST)
            detail = np.asarray(detail_img) > 0
        if simplify:
            height_up, width_up = working_labels.shape
            scale = (width_up * height_up) / max(1, source_h * source_w)
            side_scale = scale**0.5
            up_min = max(
                int(min_region_area or 0),
                int(round((min_region_area or 0) * scale)),
                30,
                int(width_up * height_up * 0.0015),
            )
            if boundary_sigma > 0:
                working_labels = smooth_boundaries(
                    working_labels, sigma=max(0.8, boundary_sigma)
                )
            working_labels = absorb_small_regions(working_labels, min_area=up_min)
            if min_thickness is not None and min_thickness > 0:
                up_thickness = max(2.0, float(min_thickness) * side_scale)
                working_labels = absorb_thin_regions(
                    working_labels, min_thickness=up_thickness
                )
                working_labels = absorb_small_regions(working_labels, min_area=up_min)

    # Colourable blocks: ≥min wide AND high on the final canvas, fitting a
    # min-diameter tip circle. Anything smaller becomes black line detail.
    height, width = working_labels.shape
    width_req = min_width_px
    height_req = min_height_px
    if min_region_mm is not None and min_region_mm > 0:
        from .print_resolution import min_region_size_for_a4_mm

        region = min_region_size_for_a4_mm(width, height, min_mm=min_region_mm)
        width_req = region.min_width_px
        height_req = region.min_height_px
    elif width_req is None and min_thickness is not None and min_thickness > 0:
        width_req = max(1, int(round(min_thickness)))
        height_req = width_req
    if width_req and height_req:
        tip = float(min(width_req, height_req))
        working_labels = enforce_min_brush_stroke(
            working_labels, min_stroke_px=tip
        )
        working_labels = merge_adjacent_same_colour(
            working_labels, bridge_px=max(2.0, tip * 0.6)
        )
        working_labels, _hl = normalize_specular_highlights(
            working_labels,
            working_palette,
            min_width_px=int(width_req),
            min_height_px=int(height_req),
            min_inscribed_px=tip,
        )
        working_labels, more_detail = enforce_colourable_blocks(
            working_labels,
            min_width_px=int(width_req),
            min_height_px=int(height_req),
            min_inscribed_px=tip,
        )
        if detail.shape != working_labels.shape:
            detail = np.zeros(working_labels.shape, dtype=bool)
        detail = detail | more_detail

    height, width = working_labels.shape
    if detail.shape != working_labels.shape:
        detail = np.zeros(working_labels.shape, dtype=bool)
    edges = _edges_from_labels(working_labels) | detail
    stroke = max(1, int(line_width))
    if stroke > 1:
        edges = ndimage.binary_dilation(edges, iterations=stroke - 1)

    outline = Image.new("RGB", (width, height), "white")
    outline_arr = np.asarray(outline).copy()
    outline_arr[edges] = (0, 0, 0)
    outline = Image.fromarray(outline_arr, mode="RGB")

    draw = ImageDraw.Draw(outline)
    base_font_size = max(12, int(min(width, height) * 0.035 * number_font_scale))
    # Cache fonts by pixel size so adaptive numbering stays cheap.
    font_cache: dict[int, ImageFont.ImageFont | ImageFont.FreeTypeFont] = {}

    regions = _find_regions(working_labels)
    colour_numbers = list(range(1, len(working_palette) + 1))

    if number_all_regions:
        numberable = regions
    else:
        # Legacy escape hatch for extremely busy raw plates.
        numberable = regions
        if len(regions) > 100:
            numberable = sorted(regions, key=lambda item: item.area, reverse=True)[:80]

    for region in sorted(numberable, key=lambda item: item.area, reverse=True):
        size = _font_size_for_region(region.area, base_size=base_font_size)
        if size not in font_cache:
            font_cache[size] = _load_font(size)
        _draw_region_number(
            draw,
            text=str(region.number),
            xy=region.centroid,
            font=font_cache[size],
            width=width,
            height=height,
        )

    legend = build_legend(working_palette, colour_numbers, swatch_size=36)
    return OutlinePage(
        outline=outline,
        legend=legend,
        regions=regions,
        palette=working_palette,
        colour_numbers=colour_numbers,
        labels=working_labels,
        simplification=stats,
        detail_ink=detail,
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


# Re-export for callers/tests that previously imported this helper.
__all__ = [
    "Region",
    "OutlinePage",
    "build_outline_page",
    "build_legend",
    "composite_page",
    "count_regions",
]
