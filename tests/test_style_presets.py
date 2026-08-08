"""Tests for simple / standard / vibrant style presets."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from colour_by_numbers.illustrate import illustration_prompt
from colour_by_numbers.pipeline import COMPLEXITY_PRESETS, create_colour_by_numbers
from colour_by_numbers.quantize import quantize_colours
from colour_by_numbers.style_presets import (
    STYLE_STANDARD,
    STYLE_VIBRANT,
    resolve_style_preset,
)


def test_resolve_vibrant_alias() -> None:
    assert resolve_style_preset("adult").name == "vibrant"
    assert resolve_style_preset("VIBRANT").max_colours == 32


def test_unknown_style_raises() -> None:
    with pytest.raises(ValueError, match="Unknown style"):
        resolve_style_preset("neon-disco")


def test_vibrant_prompt_asks_for_mosaic_and_cool_shadows() -> None:
    prompt = illustration_prompt(
        "golden retriever", category="dogs", style_preset="vibrant"
    )
    assert "interlocking" in prompt.lower() or "mosaic" in prompt.lower()
    assert "teal" in prompt.lower() or "blue" in prompt.lower()
    assert "24" in prompt or "32" in prompt
    assert "any subject" in prompt.lower() or "works for any" in prompt.lower()
    assert STYLE_VIBRANT.min_region_mm == 3.0
    assert STYLE_VIBRANT.complexity == "vibrant"
    assert STYLE_VIBRANT.pipeline_palette_mode == "exact"
    assert STYLE_VIBRANT.subject_mode == "dual"
    assert STYLE_VIBRANT.subject_complexity == "preserve"
    assert STYLE_VIBRANT.background_complexity == "simple"
    assert STYLE_VIBRANT.keep_illustration_plate is True
    assert STYLE_VIBRANT.max_plate_colours == 28


def test_vibrant_complexity_preset_is_dense() -> None:
    preset = COMPLEXITY_PRESETS["vibrant"]
    assert float(preset["blur_radius"]) == 0.0
    assert int(preset["max_regions"]) >= 300
    assert float(preset["min_area_fraction"]) < float(
        COMPLEXITY_PRESETS["fine"]["min_area_fraction"]
    )


def test_exact_palette_preserves_flat_solids() -> None:
    image = Image.new("RGB", (120, 90), (20, 40, 180))
    draw = ImageDraw.Draw(image)
    colours = [
        (240, 160, 40),
        (40, 180, 160),
        (220, 60, 50),
        (30, 30, 30),
        (250, 230, 80),
        (90, 60, 40),
        (60, 140, 220),
        (180, 100, 200),
    ]
    for i, colour in enumerate(colours):
        x0 = 8 + (i % 4) * 28
        y0 = 8 + (i // 4) * 40
        draw.rectangle((x0, y0, x0 + 24, y0 + 32), fill=colour)
    q = quantize_colours(
        image,
        n_colours=32,
        max_size=120,
        structure_size=120,
        blur_radius=0,
        palette_mode="exact",
    )
    assert q.n_colours == 1 + len(colours)


def test_exact_palette_keeps_solids_above_target_n() -> None:
    """A 29-colour plate must not median-cut just because n_colours is 28."""
    image = Image.new("RGB", (160, 120), (12, 12, 12))
    draw = ImageDraw.Draw(image)
    for i in range(28):
        colour = ((i * 17) % 230 + 20, (i * 41) % 230 + 20, (i * 73) % 230 + 20)
        x0 = 4 + (i % 7) * 22
        y0 = 4 + (i // 7) * 28
        draw.rectangle((x0, y0, x0 + 18, y0 + 22), fill=colour)
    q = quantize_colours(
        image,
        n_colours=28,
        max_size=160,
        structure_size=160,
        blur_radius=0,
        palette_mode="exact",
    )
    assert q.n_colours == 29


def test_vibrant_pipeline_keeps_rich_palette() -> None:
    """Exact + vibrant complexity should not crush a 24-colour plate to ~10."""
    rng = np.random.default_rng(0)
    image = Image.new("RGB", (200, 160), (245, 245, 250))
    draw = ImageDraw.Draw(image)
    palette = [
        tuple(int(v) for v in rng.integers(20, 250, size=3)) for _ in range(24)
    ]
    # Ensure cool shadow-ish colours are present.
    palette[0] = (40, 120, 140)
    palette[1] = (50, 90, 170)
    for i, colour in enumerate(palette):
        x0 = 10 + (i % 6) * 30
        y0 = 10 + (i // 6) * 36
        draw.rectangle((x0, y0, x0 + 26, y0 + 30), fill=colour)
    result = create_colour_by_numbers(
        image,
        n_colours=28,
        max_size=200,
        subject_mode="off",
        complexity="vibrant",
        palette_mode="exact",
        min_adjacent_delta_e=8.0,
        min_region_mm=3.0,
    )
    assert result.palette_mode == "exact"
    assert result.complexity == "vibrant"
    assert result.quantized.n_colours >= 18
    assert len(np.unique(result.page.labels)) >= 14


def test_vibrant_style_applies_across_categories() -> None:
    """House style should read as mosaic/cool-shadow for non-dog subjects too."""
    for category, subject in (
        ("aircraft", "biplane"),
        ("flowers", "sunflower"),
        ("birds", "robin"),
        ("cars", "vintage car"),
    ):
        prompt = illustration_prompt(
            subject, category=category, style_preset="vibrant"
        ).lower()
        assert "interlocking" in prompt or "mosaic" in prompt
        assert "teal" in prompt or "blue" in prompt or "cool" in prompt
        assert "nostril" not in prompt or category in {"dogs", "cats", "horses"}


def test_standard_prompt_stays_kids_cel() -> None:
    prompt = illustration_prompt(
        "golden retriever", category="dogs", style_preset="standard"
    )
    assert "children's colouring book" in prompt
    assert "white background" in prompt
    assert STYLE_STANDARD.max_colours == 16
    assert STYLE_STANDARD.complexity == "fine"
