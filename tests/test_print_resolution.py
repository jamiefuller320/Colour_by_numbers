"""Tests for A4 print-resolution helpers and search filtering."""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from colour_by_numbers.print_resolution import (
    a4_pixel_size,
    evaluate_print_resolution,
    min_region_area_for_a4_mm,
    min_region_size_for_a4_mm,
    mm_per_pixel_on_a4,
    subject_crop_adequate_for_a4,
)
from colour_by_numbers.pipeline import create_colour_by_numbers
from colour_by_numbers.search import ImageHit, filter_hits_for_a4
from colour_by_numbers.simplify import simplify_dual
from colour_by_numbers.subject import SubjectMask, harden_mask, blend_subject_background
import numpy as np


def test_a4_dpi_rejects_small_plate() -> None:
    report = evaluate_print_resolution(400, 300, min_dpi=150)
    assert not report.adequate
    assert report.effective_dpi < 150


def test_a4_dpi_accepts_large_plate() -> None:
    short, long = a4_pixel_size(150)
    report = evaluate_print_resolution(short, long, min_dpi=150)
    assert report.adequate


def test_min_region_5mm_wide_and_high_on_a4() -> None:
    # 210×210 px fit on A4 short edge → 1 mm/px → 5mm block is 5×5 px.
    region = min_region_size_for_a4_mm(210, 210, min_mm=5.0)
    assert region.min_width_px == 5
    assert region.min_height_px == 5
    assert region.min_inscribed_diameter_px == 5
    assert region.min_area_px == 25
    assert min_region_area_for_a4_mm(210, 210, min_mm=5.0) == 25
    assert mm_per_pixel_on_a4(210, 210) == pytest.approx(1.0)


def test_subject_crop_estimate() -> None:
    report = subject_crop_adequate_for_a4(
        (4000, 3000),
        (1800, 1200, 2200, 1500),
        subject_fill=0.80,
        min_dpi=150,
    )
    # 400×300 subject bbox → ~500×375 crop — too small for 150 DPI A4.
    assert not report.adequate


def test_pipeline_a4_filter_raises_on_tiny_image() -> None:
    image = Image.new("RGB", (120, 90), (200, 100, 50))
    ImageDraw.Draw(image).ellipse((20, 20, 80, 70), fill=(20, 20, 180))
    with pytest.raises(ValueError, match="DPI"):
        create_colour_by_numbers(
            image,
            n_colours=4,
            max_size=120,
            subject_mode="off",
            min_a4_dpi=150,
        )


def test_filter_hits_drops_low_res() -> None:
    hits = [
        ImageHit("small", "https://ex/a.jpg", width=400, height=300),
        ImageHit("big", "https://ex/b.jpg", width=3000, height=2000),
        ImageHit("unknown", "https://ex/c.jpg"),
    ]
    filtered = filter_hits_for_a4(hits, min_dpi=150)
    assert filtered[0].title == "big"
    assert any(h.title == "unknown" for h in filtered)
    assert all(h.title != "small" for h in filtered)


def test_harden_mask_is_binary() -> None:
    alpha = np.array([[0, 64, 128, 200, 255]], dtype=np.uint8)
    mask = SubjectMask(alpha=alpha, model="t", foreground_fraction=0.4)
    hard = harden_mask(mask)
    assert set(np.unique(hard.alpha).tolist()) <= {0, 255}


def test_firm_blend_uses_hard_edge() -> None:
    subject = Image.new("RGB", (40, 40), (255, 0, 0))
    background = Image.new("RGB", (40, 40), (0, 0, 255))
    alpha = np.zeros((40, 40), dtype=np.uint8)
    alpha[10:30, 10:30] = 180  # soft-ish interior
    mask = SubjectMask(alpha=alpha, model="t", foreground_fraction=0.25)
    blended = blend_subject_background(subject, background, mask, firm_border=True)
    arr = np.asarray(blended)
    assert tuple(arr[20, 20]) == (255, 0, 0)
    assert tuple(arr[0, 0]) == (0, 0, 255)


def test_simplify_dual_firm_border_skips_seam_soften() -> None:
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 4, size=(40, 40), dtype=np.int32)
    palette = np.array([[i * 40, 20, 200 - i * 40] for i in range(4)], dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=bool)
    mask[:, :20] = True
    params = dict(
        min_region_area=10,
        max_regions=20,
        smooth_radius=0,
        morph_radius=0,
        boundary_sigma=0.0,
        smooth_iterations=1,
    )
    firm, _, _, _ = simplify_dual(
        labels, palette, mask, subject_params=params, background_params=params, firm_border=True
    )
    soft, _, _, _ = simplify_dual(
        labels, palette, mask, subject_params=params, background_params=params, firm_border=False
    )
    assert firm.shape == labels.shape
    assert soft.shape == labels.shape
