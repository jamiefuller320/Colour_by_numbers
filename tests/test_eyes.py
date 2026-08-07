"""Tests for portrait eye preservation."""

from __future__ import annotations

import numpy as np

from colour_by_numbers.eyes import compute_eye_protection_mask, portrait_subject
from colour_by_numbers.simplify import enforce_colourable_blocks, normalize_specular_highlights


def test_portrait_subject_includes_animals_and_people() -> None:
    assert portrait_subject("dogs")
    assert portrait_subject("people")
    assert not portrait_subject("aircraft")


def test_paired_pupils_are_protected() -> None:
    labels = np.zeros((80, 100), dtype=np.int32)
    labels[:, :] = 1
    # Two dark pupils in the upper face band.
    labels[18:24, 28:34] = 0
    labels[18:24, 66:72] = 0
    # Light sclera patches beside pupils.
    labels[16:26, 22:28] = 2
    labels[16:26, 72:78] = 2
    palette = np.array(
        [[20, 20, 20], [200, 170, 120], [245, 245, 245]], dtype=np.uint8
    )
    protected = compute_eye_protection_mask(
        labels, palette, category="dogs", min_region_mm=8.0
    )
    assert protected[20, 31]
    assert protected[20, 69]
    assert protected[20, 25]
    assert protected[20, 75]


def test_protected_highlight_survives_single_eye_rule() -> None:
    palette = np.array(
        [[20, 20, 20], [40, 40, 40], [245, 245, 245]], dtype=np.uint8
    )
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[:, :] = 1
    labels[18:26, 18:26] = 2
    protected = np.zeros_like(labels, dtype=bool)
    protected[18:26, 18:26] = True
    cleaned, _ = normalize_specular_highlights(
        labels,
        palette,
        min_width_px=8,
        min_height_px=8,
        min_inscribed_px=8.0,
        protected=protected,
        protected_relaxed=(4, 4, 4.0),
    )
    assert np.any(cleaned == 2)


def test_protected_pupil_not_absorbed_by_enforce() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:, 20:] = 1
    labels[10:14, 12:16] = 0
    palette = np.array([[20, 20, 20], [200, 200, 200]], dtype=np.uint8)
    protected = np.zeros_like(labels, dtype=bool)
    protected[10:14, 12:16] = True
    cleaned, detail = enforce_colourable_blocks(
        labels,
        min_width_px=8,
        min_height_px=8,
        min_inscribed_px=8.0,
        protected=protected,
        protected_relaxed=(4, 4, 4.0),
    )
    assert np.any(cleaned == 0)
    assert not detail[10:14, 12:16].any()
