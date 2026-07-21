"""Unit tests for colour-by-numbers conversion (offline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from colour_by_numbers.outline import build_outline_page, composite_page, count_regions
from colour_by_numbers.pipeline import create_colour_by_numbers, create_from_path
from colour_by_numbers.quantize import quantize_colours


def _make_sample_image(path: Path) -> Path:
    image = Image.new("RGB", (240, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 20, 110, 110), fill=(220, 60, 60))
    draw.rectangle((130, 30, 220, 150), fill=(40, 120, 220))
    draw.polygon([(40, 140), (100, 170), (10, 170)], fill=(40, 180, 80))
    image.save(path)
    return path


def _make_noisy_photo_like(path: Path) -> Path:
    """Gradient + noise that would explode into tiny segments without simplification."""
    rng = np.random.default_rng(1)
    base = np.zeros((200, 280, 3), dtype=np.uint8)
    for y in range(200):
        for x in range(280):
            base[y, x] = (
                int(40 + 180 * x / 280),
                int(30 + 100 * y / 200),
                int(80 + 90 * ((x + y) % 60) / 60),
            )
    noise = rng.integers(-25, 26, size=base.shape, dtype=np.int16)
    arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Clear subject shapes on top.
    image = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 30, 150, 150), fill=(210, 50, 50))
    draw.rectangle((170, 40, 250, 160), fill=(40, 90, 200))
    image.save(path)
    return path


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    return _make_sample_image(tmp_path / "sample.png")


@pytest.fixture()
def noisy_image(tmp_path: Path) -> Path:
    return _make_noisy_photo_like(tmp_path / "noisy.png")


def test_quantize_to_requested_colours(sample_image: Path) -> None:
    image = Image.open(sample_image).convert("RGB")
    quantized = quantize_colours(image, n_colours=8, max_size=240, blur_radius=0)
    assert 2 <= quantized.n_colours <= 8
    assert quantized.labels.shape == (image.height, image.width) or (
        max(quantized.labels.shape) <= 240
    )
    assert quantized.palette.shape[1] == 3
    assert quantized.preview.mode == "RGB"


def test_outline_numbers_and_legend(sample_image: Path) -> None:
    image = Image.open(sample_image).convert("RGB")
    quantized = quantize_colours(image, n_colours=6, max_size=240, blur_radius=0)
    page = build_outline_page(
        quantized.labels,
        quantized.palette,
        min_region_area=40,
        max_regions=12,
    )
    assert page.outline.mode == "RGB"
    assert page.legend.mode == "RGB"
    assert page.colour_numbers[0] == 1
    assert len(page.colour_numbers) == page.palette.shape[0]
    arr = np.asarray(page.outline)
    assert (arr == 0).all(axis=2).any()
    assert len(page.regions) >= 1
    assert page.simplification is not None
    assert page.simplification.regions_after <= page.simplification.regions_before


def test_simplification_cuts_noisy_photo_regions(noisy_image: Path) -> None:
    image = Image.open(noisy_image).convert("RGB")
    raw = quantize_colours(image, n_colours=16, max_size=280, blur_radius=0)
    raw_count = count_regions(raw.labels)

    result = create_colour_by_numbers(
        image,
        n_colours=16,
        max_size=280,
        complexity="light",
    )
    simplified_count = len(result.page.regions)
    assert raw_count > 40
    assert simplified_count <= 80
    assert simplified_count < raw_count / 2
    assert result.page.simplification is not None
    assert result.page.simplification.regions_after <= 80


def test_create_from_path_writes_outputs(sample_image: Path, tmp_path: Path) -> None:
    result = create_from_path(
        sample_image, n_colours=8, max_size=240, complexity="simple"
    )
    paths = result.save(tmp_path / "out", stem="demo")
    for key in ("source", "quantized", "outline", "legend", "page"):
        assert paths[key].exists()
        assert paths[key].stat().st_size > 0
    printable = composite_page(result.page.outline, result.page.legend)
    assert printable.height > result.page.outline.height


def test_create_colour_by_numbers_defaults() -> None:
    image = Image.new("RGB", (80, 80), (255, 200, 50))
    ImageDraw.Draw(image).rectangle((10, 10, 50, 50), fill=(20, 20, 180))
    result = create_colour_by_numbers(image, n_colours=4, max_size=80)
    assert result.quantized.n_colours <= 4
    assert result.printable.size[0] >= result.page.outline.size[0]
