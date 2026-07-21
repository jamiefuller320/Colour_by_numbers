"""Print-resolution helpers for A4 colouring pages."""

from __future__ import annotations

import math
from dataclasses import dataclass

# ISO A4 in inches.
A4_INCHES: tuple[float, float] = (210 / 25.4, 297 / 25.4)


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
