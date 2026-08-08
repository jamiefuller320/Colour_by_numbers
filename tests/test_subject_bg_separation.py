"""Tests for subject/background colour separation."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from colour_by_numbers.contrast import (
    enforce_subject_background_separation,
    recover_subject_mask,
)
from colour_by_numbers.palette import colour_distance_matrix
from colour_by_numbers.simplify import merge_similar_colours_budgeted
from colour_by_numbers.subject import SubjectMask


def test_shared_subject_bg_paint_is_split_in_separation_band() -> None:
    labels = np.zeros((40, 60), dtype=np.int32)
    labels[:, :] = 1  # muddy shared tan everywhere
    labels[10:30, 15:40] = 0  # darker subject core
    # Subject also uses the muddy tan on its rim.
    labels[10:30, 15:18] = 1
    palette = np.array(
        [
            [90, 60, 30],  # subject dark
            [210, 190, 150],  # shared light tan / bg
        ],
        dtype=np.uint8,
    )
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 15:40] = True

    new_labels, new_pal = enforce_subject_background_separation(
        labels,
        palette,
        mask,
        min_delta_e=18.0,
        separation_mm=5.0,
    )
    # Background should no longer use the same index as subject rim paint.
    subj_cols = set(int(c) for c in np.unique(new_labels[mask]))
    bg_cols = set(int(c) for c in np.unique(new_labels[~mask]))
    assert not (subj_cols & bg_cols)
    dist = colour_distance_matrix(new_pal)
    for c in bg_cols:
        min_de = min(float(dist[c, s]) for s in subj_cols)
        assert min_de >= 17.5


def test_distant_shared_paint_islands_are_split() -> None:
    """Subject-coloured blobs far from the silhouette must not keep the paint."""
    labels = np.zeros((50, 80), dtype=np.int32)
    labels[:, :] = 2  # cool bg
    labels[15:35, 25:50] = 0  # subject core
    labels[15:35, 25:28] = 1  # subject rim cream
    labels[2:10, 2:12] = 1  # distant cream island (same index as rim)
    palette = np.array(
        [
            [90, 60, 30],
            [230, 210, 170],
            [40, 70, 110],
        ],
        dtype=np.uint8,
    )
    mask = np.zeros((50, 80), dtype=bool)
    mask[15:35, 25:50] = True

    new_labels, _new_pal = enforce_subject_background_separation(
        labels,
        palette,
        mask,
        min_delta_e=18.0,
        separation_mm=5.0,
    )
    subj_cols = {int(c) for c in np.unique(new_labels[mask])}
    bg_cols = {int(c) for c in np.unique(new_labels[~mask])}
    assert not (subj_cols & bg_cols)
    # Distant island must no longer use the subject rim paint.
    assert int(new_labels[5, 5]) not in subj_cols


def test_separation_keeps_multi_block_background_family() -> None:
    """Offending bg paints remap onto more than one cool swatch when possible."""
    labels = np.zeros((40, 60), dtype=np.int32)
    labels[:, :] = 1
    labels[8:32, 12:45] = 0
    labels[8:32, 12:16] = 2  # subject cream rim
    labels[:6, :] = 2  # distant shared cream
    labels[34:, :20] = 3  # second bg offender (near-subject tan)
    labels[34:, 20:] = 1
    # Put tan also in the separation band.
    labels[8:32, 45:50] = 3
    palette = np.array(
        [
            [90, 60, 30],
            [40, 70, 110],
            [230, 210, 170],
            [200, 175, 130],
        ],
        dtype=np.uint8,
    )
    mask = np.zeros((40, 60), dtype=bool)
    mask[8:32, 12:45] = True
    new_labels, new_pal = enforce_subject_background_separation(
        labels, palette, mask, min_delta_e=18.0, separation_mm=5.0
    )
    bg_cols = {int(c) for c in np.unique(new_labels[~mask])}
    subj_cols = {int(c) for c in np.unique(new_labels[mask])}
    assert not (subj_cols & bg_cols)
    # Background should not collapse to a single paint when multiple offenders exist.
    assert len(bg_cols) >= 2
    dist = colour_distance_matrix(new_pal)
    for c in bg_cols:
        assert min(float(dist[c, s]) for s in subj_cols) >= 17.5


def test_recover_subject_mask_reclaims_truncated_chest() -> None:
    """Head-only rembg mats should reclaim the under-chin torso column."""
    image = Image.new("RGB", (100, 120), (180, 185, 190))
    draw = ImageDraw.Draw(image)
    # Warm head filling the mid frame (portrait head-heavy).
    draw.ellipse((15, 5, 85, 70), fill=(200, 140, 70))
    # Pale chest wash under the chin (same family as background grey).
    draw.rectangle((28, 65, 72, 115), fill=(170, 155, 120))
    # Warm chest accents rembg often drops.
    draw.rectangle((35, 75, 55, 105), fill=(195, 145, 85))
    alpha = np.zeros((120, 100), dtype=np.uint8)
    alpha[5:70, 15:85] = 255  # head only
    mask = SubjectMask(
        alpha=alpha, model="test", foreground_fraction=float((alpha > 0).mean())
    )
    recovered = recover_subject_mask(image, mask)
    # Chest accents near the bottom should now be subject.
    assert bool(recovered.binary[90, 45])
    assert recovered.foreground_fraction > mask.foreground_fraction


def test_merge_refuses_subject_background_cross_merge() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:, 20:] = 1
    palette = np.array(
        [
            [200, 170, 120],  # subject
            [205, 175, 125],  # bg near-twin
        ],
        dtype=np.uint8,
    )
    mask = np.zeros((40, 40), dtype=bool)
    mask[:, 20:] = True
    merged, new_pal = merge_similar_colours_budgeted(
        labels,
        palette,
        max_colours=1,
        min_delta_e=10.0,
        subject_mask=mask,
        max_delta_e=20.0,
    )
    # Cross-zone merge is forbidden, so both paints remain.
    assert new_pal.shape[0] == 2
    assert np.any(merged[:, :20] != merged[:, 20:])
