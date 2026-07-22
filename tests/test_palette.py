"""Tests for standardised palette and adjacent colour contrast."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from colour_by_numbers.palette import (
    STANDARD_PALETTE_32,
    clamp_n_colours,
    colour_distance_matrix,
    contrast_delta_e,
    is_earthy_shadow_colour,
    nearest_palette_indices,
    select_active_palette,
)
from colour_by_numbers.pipeline import create_colour_by_numbers
from colour_by_numbers.quantize import quantize_colours
from colour_by_numbers.simplify import merge_low_contrast_neighbours


def test_standard_palette_has_thirty_two_distinct_colours() -> None:
    assert STANDARD_PALETTE_32.shape == (32, 3)
    dist = colour_distance_matrix(STANDARD_PALETTE_32)
    # Off-diagonal minimum should be clearly visible (not near-duplicates).
    off = dist[np.triu_indices(32, k=1)]
    assert float(off.min()) > 8.0


def test_quantize_standard_uses_fixed_palette() -> None:
    image = Image.new("RGB", (80, 60), (40, 160, 50))
    ImageDraw.Draw(image).ellipse((10, 10, 50, 50), fill=(240, 180, 40))
    q = quantize_colours(
        image, n_colours=32, max_size=80, structure_size=80, blur_radius=0, palette_mode="standard"
    )
    # Every used colour must be one of the standard set.
    for colour in q.palette:
        assert any(np.array_equal(colour, row) for row in STANDARD_PALETTE_32)


def test_merge_low_contrast_neighbours_collapses_close_paints() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:, 20:] = 1
    # Tiny muddy patch of a near-duplicate green.
    labels[0:3, 18:22] = 1
    labels[0:3, 0:4] = 0
    labels2 = np.zeros((40, 40), dtype=np.int32)
    labels2[:, :] = 0
    labels2[5:10, 5:10] = 1  # small region
    palette = np.array([[40, 160, 50], [45, 165, 55]], dtype=np.uint8)
    merged_labels, merged_palette = merge_low_contrast_neighbours(
        labels2, palette, min_delta_e=18.0, max_merge_fraction=0.5
    )
    assert merged_palette.shape[0] == 1
    assert len(np.unique(merged_labels)) == 1


def test_merge_keeps_large_adjacent_blocks() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:, 20:] = 1
    palette = np.array([[40, 160, 50], [45, 165, 55]], dtype=np.uint8)
    merged_labels, merged_palette = merge_low_contrast_neighbours(
        labels, palette, min_delta_e=18.0, max_merge_fraction=0.06
    )
    # Both halves are large — keep both paints.
    assert merged_palette.shape[0] == 2
    assert len(np.unique(merged_labels)) == 2


def test_pipeline_standard_palette_default() -> None:
    image = Image.new("RGB", (120, 90), (30, 120, 40))
    ImageDraw.Draw(image).rectangle((20, 15, 90, 75), fill=(230, 160, 40))
    result = create_colour_by_numbers(
        image,
        n_colours=32,
        max_size=120,
        subject_mode="off",
        palette_mode="standard",
        min_adjacent_delta_e=18.0,
    )
    assert result.palette_mode == "standard"
    assert result.quantized.n_colours <= 32
    assert result.quantized.n_colours >= 2


def test_select_active_palette_respects_n() -> None:
    active = select_active_palette(STANDARD_PALETTE_32, n_colours=12)
    assert active.shape[0] == 12


def test_clamp_n_colours_illustration_range() -> None:
    assert clamp_n_colours(3) == 8
    assert clamp_n_colours(12) == 12
    assert clamp_n_colours(40) == 16


def test_dog_dark_fur_maps_to_earthy_not_purple() -> None:
    """Low-light warm greys should not snap onto purple/teal for dogs."""
    # Dark warm charcoal fur sample.
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:, :] = (48, 38, 30)
    image[10:30, 10:30] = (70, 48, 32)
    active = select_active_palette(
        STANDARD_PALETTE_32, n_colours=12, image_rgb=image, category="dogs"
    )
    labels = nearest_palette_indices(image, active, category="dogs")
    used = {tuple(int(c) for c in active[i]) for i in np.unique(labels)}
    for colour in used:
        assert is_earthy_shadow_colour(np.array(colour, dtype=np.uint8)), colour
    # Purple-ish crayons should not appear on this dark fur plate.
    for colour in used:
        r, g, b = colour
        assert not (b > r + 25 and b > g + 15), colour


def test_gold_vs_green_contrast_is_high() -> None:
    gold = np.array([230, 170, 50], dtype=np.uint8)
    green = np.array([40, 140, 55], dtype=np.uint8)
    assert contrast_delta_e(gold, green) > 30
