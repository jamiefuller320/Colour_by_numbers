"""Tests for portrait eye preservation."""

from __future__ import annotations

import numpy as np

from colour_by_numbers.eyes import (
    compute_eye_protection_mask,
    face_region_mask,
    portrait_subject,
)
from colour_by_numbers.simplify import (
    absorb_small_regions,
    enforce_colourable_blocks,
    normalize_specular_highlights,
)


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


def test_profile_single_eye_is_protected() -> None:
    labels = np.zeros((60, 120), dtype=np.int32)
    labels[:, :] = 1
    # One compact pupil on the left (side view).
    labels[22:28, 24:30] = 0
    palette = np.array([[15, 15, 15], [210, 170, 110]], dtype=np.uint8)
    protected = compute_eye_protection_mask(
        labels, palette, category="dogs", min_region_mm=8.0
    )
    assert protected[25, 27]


def test_full_body_face_mask_includes_side_head() -> None:
    subject = np.zeros((80, 160), dtype=bool)
    subject[30:70, 10:150] = True  # elongated lying pose
    face = face_region_mask((80, 160), subject)
    # Head on the left must be searchable — not only the top strip of the bbox.
    assert face[45, 25]
    assert face[45, 40]


def test_emphasize_protected_pupils_is_disabled() -> None:
    """Forced pupil recolouring is a no-op (it misplaced eyes / crushed plates)."""
    from colour_by_numbers.eyes import emphasize_protected_pupils

    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:, :] = 1
    labels[12:20, 12:20] = 0
    palette = np.array([[120, 80, 40], [210, 170, 120]], dtype=np.uint8)
    protected = np.zeros_like(labels, dtype=bool)
    protected[12:20, 12:20] = True
    new_labels, new_palette = emphasize_protected_pupils(labels, palette, protected)
    assert np.array_equal(new_labels, labels)
    assert np.array_equal(new_palette, palette)


def test_absorb_small_respects_protected_pupils() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:, :] = 1
    labels[10:14, 12:16] = 0
    protected = np.zeros_like(labels, dtype=bool)
    protected[10:14, 12:16] = True
    cleaned = absorb_small_regions(labels, min_area=50, protected=protected)
    assert np.any(cleaned == 0)


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


def test_undersized_protected_eye_becomes_detail_ink() -> None:
    """Eyes too small even for relaxed fills should remain as black ink."""
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:, :] = 1
    labels[12:14, 12:14] = 0  # 2x2 pupil
    protected = np.zeros_like(labels, dtype=bool)
    protected[12:14, 12:14] = True
    cleaned, detail = enforce_colourable_blocks(
        labels,
        min_width_px=10,
        min_height_px=10,
        min_inscribed_px=10.0,
        protected=protected,
        protected_relaxed=(8, 8, 8.0),
    )
    # Fill is absorbed, but ink must mark the pupil location.
    assert detail[12:14, 12:14].all()
    assert not np.any(cleaned == 0)
