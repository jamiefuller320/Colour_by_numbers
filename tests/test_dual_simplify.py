"""Tests for dual subject/background simplification."""

from __future__ import annotations

import numpy as np

from colour_by_numbers.simplify import count_regions, simplify_dual


def test_simplify_dual_merges_background_more_than_subject() -> None:
    rng = np.random.default_rng(0)
    h, w = 80, 100
    labels = rng.integers(0, 8, size=(h, w), dtype=np.int32)
    # Strong subject block on the left.
    labels[:, :40] = 1
    labels[20:60, 10:35] = 2
    palette = np.array(
        [[i * 30, i * 20, 255 - i * 20] for i in range(8)], dtype=np.uint8
    )
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :40] = True

    subject_params = dict(
        min_region_area=30,
        max_regions=20,
        smooth_radius=1,
        morph_radius=1,
        boundary_sigma=0.4,
        smooth_iterations=1,
        min_thickness=2.0,
    )
    background_params = dict(
        min_region_area=120,
        max_regions=8,
        smooth_radius=2,
        morph_radius=1,
        boundary_sigma=0.8,
        smooth_iterations=1,
        min_thickness=3.0,
    )
    combined, new_palette, _, _ = simplify_dual(
        labels,
        palette,
        mask,
        subject_params=subject_params,
        background_params=background_params,
    )
    assert combined.shape == labels.shape
    assert new_palette.shape[0] <= 8
    assert count_regions(combined) < count_regions(labels)
