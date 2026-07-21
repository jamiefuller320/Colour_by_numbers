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


def _noise(path: Path) -> Path:
    image = Image.new("RGB", (160, 120), "white")
    pixels = image.load()
    assert pixels is not None
    for y in range(120):
        for x in range(160):
            pixels[x, y] = (
                (x * 3 + y * 5) % 255,
                (x * 7 + y * 2) % 255,
                (x * 11 + y * 13) % 255,
            )
    image.save(path)
    return path


def test_light_bracketed_by_fine_and_medium(tmp_path: Path) -> None:
    image = Image.open(_noise(tmp_path / "noise.png"))
    fine = create_colour_by_numbers(image, n_colours=12, max_size=160, complexity="fine")
    light = create_colour_by_numbers(
        image, n_colours=12, max_size=160, complexity="light"
    )
    medium = create_colour_by_numbers(
        image, n_colours=12, max_size=160, complexity="medium"
    )
    assert fine.page.simplification is not None
    assert light.page.simplification is not None
    assert medium.page.simplification is not None
    assert (
        fine.page.simplification.regions_after
        >= light.page.simplification.regions_after
        >= medium.page.simplification.regions_after
    )


def test_demo_spread_centres_on_light(tmp_path: Path) -> None:
    path = _sample(tmp_path / "sample.png")
    image = Image.open(path)
    colours, outlines, stats = build_demo_spread(
        image, n_colours=8, max_size=200, tile_width=120
    )
    assert DEMO_SPREAD_SETTINGS == ("fine", "light", "medium")
    assert set(stats) == {"fine", "light", "medium"}
    assert colours.width > 400
    assert outlines.width > 400
