"""Tests for demo spread generation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from colour_by_numbers.demo_spread import build_demo_spread
from colour_by_numbers.pipeline import DEMO_SPREAD_SETTINGS, create_colour_by_numbers


def _sample(path: Path) -> Path:
    image = Image.new("RGB", (200, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 20, 90, 100), fill=(200, 40, 40))
    draw.rectangle((110, 25, 180, 115), fill=(40, 100, 200))
    image.save(path)
    return path


def test_raw_keeps_more_regions_than_simple(tmp_path: Path) -> None:
    # Noisy field so raw quantization fragments, while simple merges hard.
    rng_image = Image.new("RGB", (160, 120), "white")
    pixels = rng_image.load()
    assert pixels is not None
    for y in range(120):
        for x in range(160):
            pixels[x, y] = (
                (x * 3 + y * 5) % 255,
                (x * 7 + y * 2) % 255,
                (x * 11 + y * 13) % 255,
            )
    path = tmp_path / "noise.png"
    rng_image.save(path)
    image = Image.open(path)
    raw = create_colour_by_numbers(image, n_colours=12, max_size=160, complexity="raw")
    simple = create_colour_by_numbers(
        image, n_colours=12, max_size=160, complexity="simple"
    )
    assert raw.page.simplification is not None
    assert simple.page.simplification is not None
    assert raw.page.simplification.regions_after > simple.page.simplification.regions_after
    assert simple.page.simplification.regions_after <= 28


def test_demo_spread_includes_original_and_settings(tmp_path: Path) -> None:
    path = _sample(tmp_path / "sample.png")
    image = Image.open(path)
    colours, outlines, stats = build_demo_spread(
        image, n_colours=8, max_size=200, tile_width=120
    )
    assert colours.width > outlines.width * 0.5
    assert set(stats) == set(DEMO_SPREAD_SETTINGS)
    # Spread is wider than a single tile (original + settings).
    assert colours.width > 400
