"""Tests for Phase B plate-quality gate."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from colour_by_numbers.pipeline import create_colour_by_numbers
from colour_by_numbers.quality import (
    PHASE_B_MIN_REGION_MM,
    PHASE_B_PRIMARY_BACKEND,
    PlateQualityError,
    assert_plate_quality,
    evaluate_plate_quality,
)


def _large_block_plate(size: int = 420) -> Image.Image:
    image = Image.new("RGB", (size, size), (240, 238, 232))
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 40, 200, 200), fill=(230, 170, 60))
    draw.rectangle((220, 40, 380, 200), fill=(50, 110, 210))
    draw.rectangle((40, 220, 200, 380), fill=(90, 50, 30))
    draw.ellipse((220, 220, 380, 380), fill=(50, 150, 60))
    return image


def test_phase_b_primary_backend_is_pollinations() -> None:
    assert PHASE_B_PRIMARY_BACKEND == "pollinations"
    assert PHASE_B_MIN_REGION_MM == 8.0


def test_quality_gate_passes_on_simple_large_blocks() -> None:
    plate = _large_block_plate()
    result = create_colour_by_numbers(
        plate,
        n_colours=12,
        max_size=420,
        complexity="simple",
        subject_mode="off",
        palette_mode="standard",
        min_region_mm=PHASE_B_MIN_REGION_MM,
        min_a4_dpi=None,
    )
    report = evaluate_plate_quality(
        result, colour_plate=plate, min_region_mm=PHASE_B_MIN_REGION_MM
    )
    assert report.passed, report.summary()
    assert_plate_quality(
        result, colour_plate=plate, min_region_mm=PHASE_B_MIN_REGION_MM
    )


def test_quality_gate_fails_when_too_many_colours() -> None:
    plate = _large_block_plate()
    result = create_colour_by_numbers(
        plate,
        n_colours=12,
        max_size=420,
        complexity="simple",
        subject_mode="off",
        palette_mode="standard",
        min_region_mm=PHASE_B_MIN_REGION_MM,
        min_a4_dpi=None,
    )
    # Pretend the product max is below colours actually used.
    report = evaluate_plate_quality(
        result,
        colour_plate=plate,
        min_region_mm=PHASE_B_MIN_REGION_MM,
        max_colours=3,
    )
    names = {check.name: check for check in report.checks}
    assert result.page.palette.shape[0] > 3
    assert not names["palette_budget"].passed
    assert not report.passed
    with pytest.raises(PlateQualityError):
        raise PlateQualityError(report)


def test_quality_gate_flags_tiny_speckles() -> None:
    """A plate that still has sub-8mm islands should fail colourable_block_size."""
    # Build labels directly via a noisy image that quantization keeps speckles
    # when simplification is effectively off (raw).
    rng = np.random.default_rng(0)
    arr = np.zeros((120, 120, 3), dtype=np.uint8)
    arr[:, :] = (240, 240, 240)
    arr[10:70, 10:70] = (220, 40, 40)
    # Salt speckles of other colours.
    for _ in range(40):
        y, x = rng.integers(0, 120, size=2)
        arr[y, x] = (50, 110, 210)
    image = Image.fromarray(arr, mode="RGB")
    result = create_colour_by_numbers(
        image,
        n_colours=12,
        max_size=120,
        complexity="raw",
        subject_mode="off",
        palette_mode="standard",
        min_region_mm=None,
        min_a4_dpi=None,
    )
    report = evaluate_plate_quality(
        result, colour_plate=image, min_region_mm=PHASE_B_MIN_REGION_MM
    )
    names = {check.name: check for check in report.checks}
    assert not names["colourable_block_size"].passed
