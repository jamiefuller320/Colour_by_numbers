"""Tests for region simplification."""

from __future__ import annotations

import numpy as np

from colour_by_numbers.simplify import (
    absorb_small_regions,
    absorb_thin_regions,
    count_regions,
    enforce_colourable_blocks,
    is_colourable_block,
    limit_region_count,
    simplify_labels,
)


def _speckled_labels(size: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Build a noisy two-tone field with many tiny islands."""
    rng = np.random.default_rng(0)
    labels = np.zeros((size, size), dtype=np.int32)
    labels[:, size // 2 :] = 1
    # Sprinkle opposite-colour speckles.
    speckles = rng.random((size, size)) < 0.08
    labels[speckles] = 1 - labels[speckles]
    palette = np.array([[20, 20, 20], [220, 220, 220]], dtype=np.uint8)
    return labels, palette


def test_absorb_thin_regions_removes_ribbons() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:, 20:] = 1
    # One-pixel-wide vertical ribbon of colour 2.
    labels[:, 10] = 2
    palette = np.array([[0, 0, 0], [200, 200, 200], [255, 0, 0]], dtype=np.uint8)
    cleaned = absorb_thin_regions(labels, min_thickness=3.0)
    assert not np.any(cleaned == 2)
    assert set(np.unique(cleaned)).issubset({0, 1})
    assert palette.shape[0] == 3  # palette unused here; keep for clarity


def test_absorb_small_regions_reduces_count() -> None:
    labels, _palette = _speckled_labels()
    before = count_regions(labels)
    merged = absorb_small_regions(labels, min_area=40)
    after = count_regions(merged)
    assert after < before
    assert after <= 4


def test_enforce_colourable_blocks_keeps_detail_as_ink() -> None:
    from scipy import ndimage

    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:, 20:] = 1
    # Tiny island that cannot fit a 5px tip circle.
    labels[5:8, 5:8] = 2
    # Thin whisker-like ribbon.
    labels[30, 5:18] = 2
    cleaned, detail = enforce_colourable_blocks(
        labels, min_width_px=5, min_height_px=5, min_inscribed_px=5.0
    )
    assert not np.any(cleaned == 2)
    assert detail.any()
    structure = np.ones((3, 3), dtype=bool)
    for colour in np.unique(cleaned):
        labeled, n = ndimage.label(cleaned == colour, structure=structure)
        for comp_id in range(1, n + 1):
            component = labeled == comp_id
            assert is_colourable_block(
                component, min_width_px=5, min_height_px=5, min_inscribed_px=5.0
            )


def test_limit_region_count_respects_cap() -> None:
    labels, _palette = _speckled_labels()
    limited = limit_region_count(labels, max_regions=3)
    assert count_regions(limited) <= 3


def test_simplify_labels_compacts_palette() -> None:
    labels, palette = _speckled_labels()
    # Add an unused third colour to the palette.
    fat_palette = np.vstack([palette, [[255, 0, 0]]])
    simplified, new_palette, stats = simplify_labels(
        labels,
        fat_palette,
        min_region_area=50,
        max_regions=4,
        smooth_radius=2,
        smooth_iterations=1,
        morph_radius=1,
    )
    assert stats.regions_after <= stats.regions_before
    assert stats.regions_after <= 4
    assert new_palette.shape[0] == len(np.unique(simplified))
    assert simplified.min() == 0
    assert simplified.max() == new_palette.shape[0] - 1
