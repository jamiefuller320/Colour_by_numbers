"""Regression tests for blue-notch, cream-bg, and nose-blemish fixes."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from colour_by_numbers.contrast import (
    enforce_subject_background_separation,
    heal_mask_notches,
    reclaim_warm_subject_edge,
    refine_mask_by_colour,
)
from colour_by_numbers.eyes import absorb_muzzle_specks
from colour_by_numbers.subject import SubjectMask


def test_refine_mask_keeps_warm_cream_edge() -> None:
    """Pale cream fur against a light sheet must not be carved to background."""
    image = Image.new("RGB", (80, 80), (210, 212, 218))
    draw = ImageDraw.Draw(image)
    draw.ellipse((15, 15, 65, 65), fill=(220, 185, 140))
    alpha = np.zeros((80, 80), dtype=np.uint8)
    # Slightly under-cover the top of the head.
    alpha[22:65, 18:62] = 255
    mask = SubjectMask(
        alpha=alpha, model="test", foreground_fraction=float((alpha > 0).mean())
    )
    refined = refine_mask_by_colour(image, mask, band_px=8, min_advantage=2.0)
    # Warm pixels at the crown should stay / return to subject.
    assert bool(refined.binary[18, 40])


def test_heal_mask_notches_fills_small_crown_bite() -> None:
    alpha = np.zeros((60, 60), dtype=np.uint8)
    alpha[10:50, 10:50] = 255
    # Small rectangular notch in the top edge.
    alpha[10:16, 25:35] = 0
    mask = SubjectMask(
        alpha=alpha, model="test", foreground_fraction=float((alpha > 0).mean())
    )
    healed = heal_mask_notches(mask, close_iterations=8)
    assert bool(healed.binary[12, 30])
    assert healed.foreground_fraction > mask.foreground_fraction


def test_contrasting_family_no_recursion_when_pool_blocked() -> None:
    """Subject paints that defeat every pool swatch must not recurse forever."""
    from colour_by_numbers.contrast import _contrasting_background_family

    # Include every cool-pool neighbour so ΔE gates fail for all candidates.
    subj = np.array(
        [
            [70, 95, 125],
            [55, 75, 105],
            [100, 120, 140],
            [120, 145, 170],
            [45, 65, 90],
            [230, 236, 245],
            [210, 220, 235],
            [200, 210, 220],
            [180, 195, 215],
            [160, 180, 205],
            [210, 222, 235],
            [180, 200, 220],
            [140, 165, 195],
            [100, 125, 155],
            [160, 185, 210],
        ],
        dtype=np.uint8,
    )
    family = _contrasting_background_family(
        subj.mean(axis=0), subj, min_delta_e=80.0
    )
    assert family.shape[0] >= 1
    assert family.shape[1] == 3


def test_separation_remaps_warm_cream_background() -> None:
    labels = np.zeros((40, 60), dtype=np.int32)
    labels[:, :] = 1  # cool bg
    labels[10:30, 15:40] = 0  # subject
    labels[:8, :20] = 2  # cream block in background (ΔE may be ≥18)
    palette = np.array(
        [
            [160, 110, 50],  # warm subject
            [70, 95, 125],  # cool bg
            [235, 220, 185],  # cream bg survivor
        ],
        dtype=np.uint8,
    )
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 15:40] = True
    new_labels, new_pal = enforce_subject_background_separation(
        labels, palette, mask, min_delta_e=18.0, separation_mm=5.0
    )
    cream_idx = int(new_labels[2, 2])
    rgb = new_pal[cream_idx]
    # Cream must not survive as a warm coat-like background fill.
    assert not (int(rgb[0]) >= 170 and int(rgb[0]) >= int(rgb[2]) + 8)


def test_reclaim_warm_edge_repairs_crown_notch() -> None:
    image = Image.new("RGB", (60, 60), (90, 120, 150))
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 10, 50, 50), fill=(210, 160, 90))
    labels = np.zeros((60, 60), dtype=np.int32)
    labels[:, :] = 1
    labels[15:50, 15:50] = 0
    # Notch already painted cool in labels.
    labels[10:16, 25:35] = 1
    palette = np.array([[210, 160, 90], [90, 120, 150]], dtype=np.uint8)
    mask = np.zeros((60, 60), dtype=bool)
    mask[15:50, 15:50] = True
    new_labels, _pal, new_mask = reclaim_warm_subject_edge(
        image, labels, palette, mask, band_px=10
    )
    assert bool(new_mask[12, 30])
    assert int(new_labels[12, 30]) == 0


def test_absorb_muzzle_specks_removes_nose_fleck() -> None:
    labels = np.zeros((80, 80), dtype=np.int32)
    labels[:, :] = 0  # fur
    # Head band
    labels[10:55, 20:60] = 0
    # Nose leather
    labels[40:50, 35:45] = 1
    # Light blemish on the nose
    labels[42:46, 38:42] = 2
    # Eyes (protected)
    labels[22:28, 28:34] = 1
    labels[22:28, 46:52] = 1
    palette = np.array(
        [
            [200, 150, 80],
            [25, 20, 18],
            [240, 235, 220],
        ],
        dtype=np.uint8,
    )
    protected = np.zeros((80, 80), dtype=bool)
    protected[22:28, 28:34] = True
    protected[22:28, 46:52] = True
    cleaned = absorb_muzzle_specks(
        labels, palette, category="dogs", protected=protected, max_area=40
    )
    assert int(cleaned[44, 40]) != 2
    # Eyes untouched.
    assert int(cleaned[25, 30]) == 1
