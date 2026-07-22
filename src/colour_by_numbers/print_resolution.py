"""Print-resolution helpers for A4 colouring pages."""

from __future__ import annotations

import math
from dataclasses import dataclass

# ISO A4 in mm and inches.
A4_MM: tuple[float, float] = (210.0, 297.0)
A4_INCHES: tuple[float, float] = (A4_MM[0] / 25.4, A4_MM[1] / 25.4)
DEFAULT_MIN_REGION_MM = 5.0


@dataclass(frozen=True)
class RegionPrintSize:
    """Minimum colourable-block footprint when a plate fills A4.

    A block must be at least ``min_mm`` wide **and** ``min_mm`` high so a
    circular colouring tip of diameter ``min_mm`` fits inside it.
    """

    min_width_px: int
    min_height_px: int
    min_area_px: int
    min_mm: float
    mm_per_px: float

    @property
    def min_side_px(self) -> int:
        return min(self.min_width_px, self.min_height_px)

    @property
    def min_inscribed_diameter_px(self) -> int:
        """Pixels needed for a ``min_mm``-diameter circle inside the block."""
        return self.min_side_px


@dataclass(frozen=True)
class PrintResolution:
    """Native pixel size of a plate vs A4 print requirements."""

    width: int
    height: int
    min_dpi: float
    effective_dpi: float
    required_width: int
    required_height: int

    @property
    def adequate(self) -> bool:
        return self.effective_dpi + 0.5 >= self.min_dpi


def a4_pixel_size(dpi: float) -> tuple[int, int]:
    """Return (short_edge_px, long_edge_px) for A4 at ``dpi``."""
    short = int(math.ceil(min(A4_INCHES) * dpi))
    long = int(math.ceil(max(A4_INCHES) * dpi))
    return short, long


def mm_per_pixel_on_a4(width: int, height: int) -> float:
    """Millimetres per pixel when ``width``×``height`` is fit onto A4.

    Chooses the page orientation that prints the plate larger (higher
    mm/px), matching how colouring pages are typically scaled to the page.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    short_mm, long_mm = A4_MM
    # Portrait page: fit into 210×297 mm.
    scale_portrait = min(short_mm / width, long_mm / height)
    # Landscape page: fit into 297×210 mm.
    scale_landscape = min(long_mm / width, short_mm / height)
    return float(max(scale_portrait, scale_landscape))


def min_region_size_for_a4_mm(
    width: int,
    height: int,
    *,
    min_mm: float = DEFAULT_MIN_REGION_MM,
) -> RegionPrintSize:
    """Pixel size of a block that is ``min_mm`` wide and ``min_mm`` high on A4.

    Colourable fills smaller than this in either dimension cannot fit a
    ``min_mm``-diameter colouring tip; those features should become black
    line detail instead of numbered colour blocks.
    """
    if min_mm <= 0:
        return RegionPrintSize(
            min_width_px=1,
            min_height_px=1,
            min_area_px=1,
            min_mm=float(min_mm),
            mm_per_px=mm_per_pixel_on_a4(width, height),
        )
    mm_per_px = mm_per_pixel_on_a4(width, height)
    side = max(1, int(math.ceil(min_mm / mm_per_px)))
    return RegionPrintSize(
        min_width_px=side,
        min_height_px=side,
        min_area_px=side * side,
        min_mm=float(min_mm),
        mm_per_px=mm_per_px,
    )


def min_region_area_for_a4_mm(
    width: int,
    height: int,
    *,
    min_mm: float = DEFAULT_MIN_REGION_MM,
) -> int:
    """Convenience: minimum region area in pixels for ``min_mm`` on A4."""
    return min_region_size_for_a4_mm(width, height, min_mm=min_mm).min_area_px


def min_pixels_for_a4(dpi: float) -> tuple[int, int]:
    """Minimum ``(width, height)`` that can cover A4 at ``dpi`` (portrait)."""
    return a4_pixel_size(dpi)


def image_adequate_for_a4(
    width: int,
    height: int,
    *,
    min_dpi: float = 150.0,
) -> bool:
    """True when ``width``×``height`` can print to fill A4 at ``min_dpi``."""
    return evaluate_print_resolution(width, height, min_dpi=min_dpi).adequate


def evaluate_print_resolution(
    width: int,
    height: int,
    *,
    min_dpi: float = 150.0,
) -> PrintResolution:
    """Estimate effective DPI if ``width``×``height`` is printed to fill A4.

    Uses the orientation (portrait/landscape) that yields the higher minimum
    edge DPI, i.e. how sharply the plate can cover an A4 page.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    req_short, req_long = a4_pixel_size(min_dpi)
    # Portrait fit: width→short, height→long
    dpi_portrait = min(width / A4_INCHES[0], height / A4_INCHES[1])
    # Landscape fit: width→long, height→short
    dpi_landscape = min(width / A4_INCHES[1], height / A4_INCHES[0])
    effective = max(dpi_portrait, dpi_landscape)
    return PrintResolution(
        width=width,
        height=height,
        min_dpi=float(min_dpi),
        effective_dpi=float(effective),
        required_width=req_short,
        required_height=req_long,
    )


def estimate_subject_crop_size(
    image_size: tuple[int, int],
    subject_bbox: tuple[int, int, int, int],
    *,
    subject_fill: float = 0.80,
) -> tuple[int, int]:
    """Estimate native crop size when the subject bbox fills ``subject_fill``.

    ``subject_bbox`` is ``(x0, y0, x1, y1)`` in image coordinates.
    """
    width, height = image_size
    x0, y0, x1, y1 = subject_bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    aspect = width / max(height, 1)
    min_w = bw / subject_fill
    min_h = bh / subject_fill
    crop_h = max(min_h, min_w / aspect)
    crop_w = crop_h * aspect
    if crop_w < min_w:
        crop_w = min_w
        crop_h = crop_w / aspect
    # Clamp to available image when subject already large.
    crop_w = min(crop_w, width)
    crop_h = min(crop_h, height)
    return int(round(crop_w)), int(round(crop_h))


def subject_crop_adequate_for_a4(
    image_size: tuple[int, int],
    subject_bbox: tuple[int, int, int, int],
    *,
    subject_fill: float = 0.80,
    min_dpi: float = 150.0,
) -> PrintResolution:
    """Check whether an 80%-fill subject crop has enough native pixels for A4."""
    crop_w, crop_h = estimate_subject_crop_size(
        image_size, subject_bbox, subject_fill=subject_fill
    )
    return evaluate_print_resolution(crop_w, crop_h, min_dpi=min_dpi)
