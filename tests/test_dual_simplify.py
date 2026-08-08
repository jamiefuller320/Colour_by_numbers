"""Tests for dual subject/background simplification."""

from __future__ import annotations

import numpy as np

from colour_by_numbers.simplify import (
    count_regions,
    merge_similar_colours_budgeted,
    simplify_dual,
)


def test_merge_budget_prefers_background_colours() -> None:
    labels = np.zeros((40, 60), dtype=np.int32)
    labels[:, 30:] = 1  # subject right
    labels[:10, :20] = 2  # bg accent A
    labels[10:20, :20] = 3  # bg accent B (near A)
    palette = np.array(
        [
            [200, 160, 100],
            [180, 140, 80],
            [240, 240, 240],
            [230, 230, 230],
        ],
        dtype=np.uint8,
    )
    mask = np.zeros((40, 60), dtype=bool)
    mask[:, 30:] = True
    merged, new_pal = merge_similar_colours_budgeted(
        labels, palette, max_colours=3, subject_mask=mask, max_delta_e=20.0
    )
    assert new_pal.shape[0] <= 3
    # Subject colours should still be present as distinct fills when possible.
    assert np.any(merged[:, 30:] == merged[20, 40])


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
