"""Plate-quality gate: pass/fail checks for a single colouring plate.

Phase B: rights-safe plate a human can colour from the numbered outline to
reconstruct the flat colour image.

Phase C: format-brief composition metrics (colourable fill, dominant-colour
cap, subject fill) from ``FORMAT_BRIEF.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from .outline import OutlinePage, count_regions
from .palette import MAX_N_COLOURS, MIN_N_COLOURS
from .pipeline import ColourByNumbersResult
from .print_resolution import (
    DEFAULT_MIN_REGION_MM,
    min_region_size_for_a4_mm,
)
from .quantize import QuantizedImage

# Locked Phase B product bar (stricter than legacy 5mm experiments).
PHASE_B_MIN_REGION_MM = 8.0
PHASE_B_PRIMARY_BACKEND = "pollinations"
PHASE_B_PRIMARY_MODEL = "flux"

# Reconstructibility: outline labels → palette must rebuild the colour plate.
RECONSTRUCT_MATCH_MIN = 0.995
# Mean |RGB| difference between illustration and flat colour plate on fill pixels.
ILLUSTRATION_MEAN_DIFF_MAX = 45.0

# Phase C format-brief composition bar (see FORMAT_BRIEF.md).
COLOURABLE_FILL_MIN = 0.90
MAX_COLOUR_SHARE = 0.50
SUBJECT_FILL_MIN = 0.50


@dataclass(frozen=True)
class QualityCheck:
    """One pass/fail row on the plate checklist."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PlateQualityReport:
    """Full Phase B checklist for one plate."""

    checks: tuple[QualityCheck, ...]
    min_region_mm: float = PHASE_B_MIN_REGION_MM

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed(self) -> list[QualityCheck]:
        return [check for check in self.checks if not check.passed]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"Plate quality: {status}"]
        for check in self.checks:
            mark = "✓" if check.passed else "✗"
            lines.append(f"  {mark} {check.name}: {check.detail}")
        return "\n".join(lines)


class PlateQualityError(RuntimeError):
    """Raised when ``require_quality`` is set and the plate fails the gate."""

    def __init__(self, report: PlateQualityReport) -> None:
        super().__init__(report.summary())
        self.report = report


def _region_bbox_sides(component: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(component)
    if ys.size == 0:
        return 0, 0
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _undersized_region_count(
    labels: np.ndarray,
    *,
    min_width_px: int,
    min_height_px: int,
    min_area_px: int,
) -> int:
    """Count connected components that fail the A4 colourable-block floor."""
    structure = np.ones((3, 3), dtype=bool)
    bad = 0
    for colour in np.unique(labels):
        labeled, n = ndimage.label(labels == colour, structure=structure)
        if n == 0:
            continue
        areas = np.bincount(labeled.ravel())
        for comp_id in range(1, n + 1):
            component = labeled == comp_id
            area = int(areas[comp_id])
            width, height = _region_bbox_sides(component)
            if area < min_area_px or width < min_width_px or height < min_height_px:
                bad += 1
    return bad


def _border_dominant_label(labels: np.ndarray) -> int:
    """Palette index that dominates the image border (proxy for background)."""
    border = np.concatenate(
        [
            labels[0, :].ravel(),
            labels[-1, :].ravel(),
            labels[:, 0].ravel(),
            labels[:, -1].ravel(),
        ]
    )
    return int(np.bincount(border.astype(np.int64)).argmax())


def colourable_fill_fraction(ink_mask: np.ndarray) -> float:
    """Fraction of pixels that are fillable (not outline/line ink)."""
    if ink_mask.size == 0:
        return 0.0
    return float((~ink_mask).mean())


def max_colour_share(labels: np.ndarray) -> float:
    """Largest single palette-index share of the page."""
    if labels.size == 0:
        return 1.0
    counts = np.bincount(labels.astype(np.int64).ravel())
    return float(counts.max() / labels.size)


def subject_fill_fraction(labels: np.ndarray) -> float:
    """Fraction of pixels that are not the border-dominant background colour."""
    if labels.size == 0:
        return 0.0
    background = _border_dominant_label(labels)
    return float(np.mean(labels != background))


def evaluate_plate_quality(
    result: ColourByNumbersResult,
    *,
    colour_plate: Image.Image | None = None,
    min_region_mm: float = PHASE_B_MIN_REGION_MM,
    min_colours: int = MIN_N_COLOURS,
    max_colours: int = MAX_N_COLOURS,
    min_colourable_fill: float = COLOURABLE_FILL_MIN,
    max_single_colour_share: float = MAX_COLOUR_SHARE,
    min_subject_fill: float = SUBJECT_FILL_MIN,
) -> PlateQualityReport:
    """Run the Phase B pass/fail checklist on a colour-by-numbers result.

    ``colour_plate`` is the flat illustration the outline should reconstruct
    (defaults to ``result.source`` when omitted).
    """
    page: OutlinePage = result.page
    quantized: QuantizedImage = result.quantized
    labels = page.labels
    palette = page.palette
    height, width = labels.shape
    checks: list[QualityCheck] = []

    n = int(palette.shape[0])
    # Budget is a ceiling for book plates; simple subjects may use fewer paints.
    checks.append(
        QualityCheck(
            name="palette_budget",
            passed=2 <= n <= max_colours,
            detail=(
                f"{n} colours used (max {max_colours}; "
                f"generation target {min_colours}–{max_colours})"
            ),
        )
    )

    region_count = count_regions(labels)
    listed = len(page.regions)
    checks.append(
        QualityCheck(
            name="every_block_listed",
            passed=listed == region_count and listed > 0,
            detail=f"{listed} listed / {region_count} connected blocks",
        )
    )

    numbers = {region.number for region in page.regions}
    key_numbers = set(page.colour_numbers)
    checks.append(
        QualityCheck(
            name="numbers_match_key",
            passed=numbers <= key_numbers and len(key_numbers) == n,
            detail=(
                f"region numbers {sorted(numbers)[:8]}… "
                f"key {sorted(key_numbers)}"
                if numbers
                else "no region numbers"
            ),
        )
    )

    # Reconstruct colour plate from labels + palette (exact contract).
    reconstructed = palette[labels]
    preview = np.asarray(quantized.preview.convert("RGB"), dtype=np.uint8)
    if preview.shape[:2] != labels.shape:
        preview_img = quantized.preview.convert("RGB").resize(
            (width, height), Image.Resampling.NEAREST
        )
        preview = np.asarray(preview_img, dtype=np.uint8)
    reconstruct_match = float(np.mean(np.all(preview == reconstructed, axis=2)))
    checks.append(
        QualityCheck(
            name="outline_reconstructs_colour_plate",
            passed=reconstruct_match >= RECONSTRUCT_MATCH_MIN,
            detail=f"{reconstruct_match:.1%} pixels match labels→palette rebuild",
        )
    )

    plate = (colour_plate or result.source).convert("RGB")
    if plate.size != (width, height):
        plate = plate.resize((width, height), Image.Resampling.NEAREST)
    plate_arr = np.asarray(plate, dtype=np.uint8)
    outline_arr = np.asarray(page.outline.convert("RGB"), dtype=np.uint8)
    ink = np.all(outline_arr < 40, axis=2)
    fill = ~ink
    # Faithfulness of the flat colour plate to the illustration (after palette
    # snap + simplify). Exact index agreement is too brittle; use mean |RGB|.
    if fill.any():
        mean_diff = float(
            np.abs(
                plate_arr[fill].astype(np.int16) - reconstructed[fill].astype(np.int16)
            ).mean()
        )
    else:
        mean_diff = 255.0
    checks.append(
        QualityCheck(
            name="illustration_agreement",
            passed=mean_diff <= ILLUSTRATION_MEAN_DIFF_MAX,
            detail=(
                f"mean |RGB| diff vs colour plate {mean_diff:.1f} "
                f"(max {ILLUSTRATION_MEAN_DIFF_MAX:g})"
            ),
        )
    )

    region = min_region_size_for_a4_mm(width, height, min_mm=min_region_mm)
    undersized = _undersized_region_count(
        labels,
        min_width_px=region.min_width_px,
        min_height_px=region.min_height_px,
        min_area_px=region.min_area_px,
    )
    checks.append(
        QualityCheck(
            name="colourable_block_size",
            passed=undersized == 0,
            detail=(
                f"{undersized} blocks below {min_region_mm:g}mm×{min_region_mm:g}mm "
                f"on A4 (~{region.min_side_px}px side)"
            ),
        )
    )

    ink_fraction = float(ink.mean())
    checks.append(
        QualityCheck(
            name="outline_has_ink",
            passed=0.001 <= ink_fraction <= 0.35,
            detail=f"{ink_fraction:.2%} ink pixels on outline",
        )
    )

    legend_ok = page.legend.size[0] > 0 and page.legend.size[1] > 0
    checks.append(
        QualityCheck(
            name="legend_present",
            passed=legend_ok and len(page.colour_numbers) == n,
            detail=f"legend {page.legend.size[0]}×{page.legend.size[1]}, {n} swatches",
        )
    )

    # Phase C format-brief composition metrics.
    fill_fraction = colourable_fill_fraction(ink)
    checks.append(
        QualityCheck(
            name="colourable_fill_fraction",
            passed=fill_fraction >= min_colourable_fill,
            detail=(
                f"{fill_fraction:.1%} of page is colourable fill "
                f"(min {min_colourable_fill:.0%})"
            ),
        )
    )

    dominant_share = max_colour_share(labels)
    checks.append(
        QualityCheck(
            name="max_colour_share",
            passed=dominant_share <= max_single_colour_share,
            detail=(
                f"largest single colour covers {dominant_share:.1%} of page "
                f"(max {max_single_colour_share:.0%})"
            ),
        )
    )

    subject_fraction = subject_fill_fraction(labels)
    checks.append(
        QualityCheck(
            name="subject_fill_fraction",
            passed=subject_fraction >= min_subject_fill,
            detail=(
                f"non-background colours cover {subject_fraction:.1%} of page "
                f"(min {min_subject_fill:.0%})"
            ),
        )
    )

    return PlateQualityReport(checks=tuple(checks), min_region_mm=min_region_mm)


def assert_plate_quality(
    result: ColourByNumbersResult,
    *,
    colour_plate: Image.Image | None = None,
    min_region_mm: float = PHASE_B_MIN_REGION_MM,
) -> PlateQualityReport:
    """Evaluate quality and raise ``PlateQualityError`` on failure."""
    report = evaluate_plate_quality(
        result, colour_plate=colour_plate, min_region_mm=min_region_mm
    )
    if not report.passed:
        raise PlateQualityError(report)
    return report


# Re-export default mm for callers that want the legacy constant nearby.
__all__ = [
    "PHASE_B_MIN_REGION_MM",
    "PHASE_B_PRIMARY_BACKEND",
    "PHASE_B_PRIMARY_MODEL",
    "COLOURABLE_FILL_MIN",
    "MAX_COLOUR_SHARE",
    "SUBJECT_FILL_MIN",
    "QualityCheck",
    "PlateQualityReport",
    "PlateQualityError",
    "colourable_fill_fraction",
    "max_colour_share",
    "subject_fill_fraction",
    "evaluate_plate_quality",
    "assert_plate_quality",
    "DEFAULT_MIN_REGION_MM",
]
