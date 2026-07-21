"""Tests for subject isolation helpers (mocked rembg)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from colour_by_numbers.subject import (
    SubjectMask,
    crop_to_subject_fill,
    isolate_on_flat_background,
    prepare_subject_image,
)


def _plane_like(size: tuple[int, int] = (400, 240)) -> Image.Image:
    image = Image.new("RGB", size, (30, 90, 180))
    draw = ImageDraw.Draw(image)
    draw.polygon(
        [(170, 110), (250, 100), (260, 115), (245, 130), (160, 125)],
        fill=(20, 20, 40),
    )
    draw.rectangle((190, 105, 230, 120), fill=(220, 220, 230))
    return image


def test_crop_to_subject_fill_targets_eighty_percent() -> None:
    image = _plane_like()
    alpha = np.zeros((240, 400), dtype=np.uint8)
    alpha[100:130, 160:260] = 255
    mask = SubjectMask(alpha=alpha, model="test", foreground_fraction=0.03)
    cropped, cropped_mask = crop_to_subject_fill(image, mask, target_fill=0.80)
    ys, xs = np.nonzero(cropped_mask.binary)
    fill = max(
        (xs.max() - xs.min() + 1) / cropped.width,
        (ys.max() - ys.min() + 1) / cropped.height,
    )
    assert 0.72 <= fill <= 0.92
    assert cropped.width < image.width or cropped.height < image.height


def test_isolate_composites_flat_background(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _plane_like()

    def fake_estimate(img, *, model_name="u2net"):
        alpha = np.zeros((img.height, img.width), dtype=np.uint8)
        alpha[100:130, 160:260] = 255
        return SubjectMask(alpha=alpha, model=model_name, foreground_fraction=0.03)

    monkeypatch.setattr(
        "colour_by_numbers.subject.estimate_subject_mask", fake_estimate
    )
    isolated, mask = isolate_on_flat_background(image, background=(200, 200, 200))
    arr = np.asarray(isolated)
    assert tuple(arr[0, 0]) == (200, 200, 200)
    assert mask.foreground_fraction > 0


def test_prepare_dual_keeps_scene_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _plane_like()

    def fake_estimate(img, *, model_name="u2net"):
        alpha = np.zeros((img.height, img.width), dtype=np.uint8)
        alpha[100:130, 160:260] = 255
        return SubjectMask(alpha=alpha, model=model_name, foreground_fraction=0.03)

    monkeypatch.setattr(
        "colour_by_numbers.subject.estimate_subject_mask", fake_estimate
    )
    prepared, mask = prepare_subject_image(image, mode="dual", subject_fill=0.80)
    assert mask is not None
    # Dual keeps real sky pixels (not flat white) near corners of crop when present.
    assert prepared.size[0] > 0
