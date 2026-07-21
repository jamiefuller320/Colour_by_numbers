"""Tests for subject isolation helpers (mocked rembg)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from colour_by_numbers.subject import (
    SubjectMask,
    crop_to_subject,
    isolate_on_flat_background,
    prepare_subject_image,
)


def _plane_like(size: tuple[int, int] = (200, 120)) -> Image.Image:
    image = Image.new("RGB", size, (30, 90, 180))  # sky
    draw = ImageDraw.Draw(image)
    # Small dark "plane" in the centre.
    draw.polygon([(90, 55), (130, 50), (135, 58), (128, 65), (85, 62)], fill=(20, 20, 40))
    draw.rectangle((100, 52, 118, 60), fill=(220, 220, 230))
    return image


def test_crop_to_subject_fills_frame() -> None:
    image = _plane_like()
    alpha = np.zeros((120, 200), dtype=np.uint8)
    alpha[48:68, 84:140] = 255
    mask = SubjectMask(alpha=alpha, model="test", foreground_fraction=0.05)
    cropped, cropped_mask = crop_to_subject(image, mask, padding_fraction=0.1)
    assert cropped.width < image.width
    assert cropped.height < image.height
    assert cropped_mask.foreground_fraction > mask.foreground_fraction


def test_isolate_composites_flat_background(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _plane_like()

    def fake_estimate(img, *, model_name="u2net"):
        alpha = np.zeros((img.height, img.width), dtype=np.uint8)
        alpha[48:68, 84:140] = 255
        return SubjectMask(alpha=alpha, model=model_name, foreground_fraction=0.05)

    monkeypatch.setattr(
        "colour_by_numbers.subject.estimate_subject_mask", fake_estimate
    )
    isolated, mask = isolate_on_flat_background(image, background=(200, 200, 200))
    arr = np.asarray(isolated)
    # Corners should be the flat background.
    assert tuple(arr[0, 0]) == (200, 200, 200)
    assert mask.foreground_fraction > 0


def test_prepare_subject_off_is_noop() -> None:
    image = _plane_like()
    prepared, mask = prepare_subject_image(image, mode="off")
    assert mask is None
    assert prepared.size == image.size
