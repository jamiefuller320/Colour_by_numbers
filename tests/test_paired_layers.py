"""Tests for paired subject-only + subject+background illustration layers."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from colour_by_numbers.illustrate import (
    subject_only_illustration_prompt,
)
from colour_by_numbers.pipeline import create_colour_by_numbers
from colour_by_numbers.style_presets import STYLE_VIBRANT
from colour_by_numbers.subject import (
    SubjectMask,
    mask_from_subject_layer,
    prepare_subject_image,
)


def test_subject_only_prompt_asks_for_flat_studio_ground() -> None:
    prompt = subject_only_illustration_prompt(
        "golden retriever",
        category="dogs",
        style_preset="vibrant",
        framing="portrait",
        scene_prompt="Wide shot of a golden retriever: sitting upright. Cool abstract background.",
    )
    lowered = prompt.lower()
    assert "flat" in lowered
    assert ("grey" in lowered) or ("gray" in lowered)
    assert "#f0f0f0" in lowered
    assert ("no abstract" in lowered) or ("no second background" in lowered)
    assert "golden retriever" in lowered
    # Must not ask for the busy multi-block scene background.
    assert "cool abstract background" not in lowered


def test_vibrant_preset_enables_paired_layers() -> None:
    assert STYLE_VIBRANT.paired_illustration_layers is True


def test_mask_from_subject_layer_prefers_flat_ground_cutout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subject-only on grey → clean mask; scene has busy blue bg."""
    scene = Image.new("RGB", (120, 100), (70, 100, 140))
    draw = ImageDraw.Draw(scene)
    draw.ellipse((30, 15, 90, 75), fill=(200, 140, 60))
    # Busy bg blotches that would confuse rembg on the scene alone.
    draw.rectangle((0, 70, 40, 100), fill=(190, 160, 110))
    draw.rectangle((85, 0, 120, 35), fill=(160, 150, 90))

    subject_only = Image.new("RGB", (120, 100), (240, 240, 240))
    draw_o = ImageDraw.Draw(subject_only)
    draw_o.ellipse((30, 15, 90, 75), fill=(200, 140, 60))

    def fake_estimate(img, *, model_name="u2net"):
        arr = np.asarray(img.convert("RGB"))
        # Orange subject ellipse only (ignore muddy tan bg blotches).
        warm = (
            (arr[:, :, 0] > 180)
            & (arr[:, :, 1] > 100)
            & (arr[:, :, 1] < 170)
            & (arr[:, :, 2] < 100)
        )
        alpha = np.where(warm, 255, 0).astype(np.uint8)
        return SubjectMask(
            alpha=alpha,
            model=model_name,
            foreground_fraction=float((alpha > 0).mean()),
        )

    monkeypatch.setattr(
        "colour_by_numbers.subject.estimate_subject_mask", fake_estimate
    )
    mask = mask_from_subject_layer(scene, subject_only, absdiff_boost=False)
    assert mask.model.startswith("paired:")
    # Centre of the ellipse should be subject.
    assert bool(mask.binary[45, 60])
    # Far corner / muddy blotch should not be subject.
    assert not bool(mask.binary[5, 5])
    assert not bool(mask.binary[90, 10])


def test_prepare_accepts_precomputed_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("RGB", (80, 60), (30, 90, 180))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 10, 60, 50), fill=(200, 140, 60))
    alpha = np.zeros((60, 80), dtype=np.uint8)
    alpha[10:50, 20:60] = 255
    given = SubjectMask(alpha=alpha, model="test", foreground_fraction=0.3)

    def boom(*_a, **_k):
        raise AssertionError("rembg should be skipped when mask is provided")

    monkeypatch.setattr("colour_by_numbers.subject.estimate_subject_mask", boom)
    prepared, mask = prepare_subject_image(
        image, mode="dual", subject_mask=given, subject_fill=0.8, colour_refine=False
    )
    assert mask is not None
    assert mask.model == "test" or mask.foreground_fraction > 0
    assert prepared.size[0] > 0


def test_pipeline_uses_injected_subject_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("RGB", (100, 80), (80, 120, 160))
    draw = ImageDraw.Draw(image)
    draw.ellipse((25, 15, 75, 65), fill=(210, 150, 70))
    alpha = np.zeros((80, 100), dtype=np.uint8)
    alpha[15:65, 25:75] = 255
    given = SubjectMask(alpha=alpha, model="paired:test", foreground_fraction=0.4)

    def boom(*_a, **_k):
        raise AssertionError("pipeline should not rembg when subject_mask is set")

    monkeypatch.setattr("colour_by_numbers.subject.estimate_subject_mask", boom)
    result = create_colour_by_numbers(
        image,
        n_colours=12,
        complexity="vibrant",
        subject_mode="dual",
        subject_complexity="preserve",
        background_complexity="simple",
        firm_border=True,
        colour_refine=False,
        subject_mask=given,
        subject_bg_separation_mm=5.0,
        min_subject_bg_delta_e=18.0,
        silhouette_outline=True,
    )
    assert result.subject_mask is not None
    assert result.page.labels.shape[0] > 0
