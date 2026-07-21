"""Unit tests for colour-by-numbers conversion (offline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from colour_by_numbers.outline import build_outline_page, composite_page
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


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    return _make_sample_image(tmp_path / "sample.png")


def test_quantize_to_requested_colours(sample_image: Path) -> None:
    image = Image.open(sample_image).convert("RGB")
    quantized = quantize_colours(image, n_colours=8, max_size=240)
    assert 2 <= quantized.n_colours <= 8
    assert quantized.labels.shape == (image.height, image.width) or (
        max(quantized.labels.shape) <= 240
    )
    assert quantized.palette.shape[1] == 3
    assert quantized.preview.mode == "RGB"


def test_outline_numbers_and_legend(sample_image: Path) -> None:
    image = Image.open(sample_image).convert("RGB")
    quantized = quantize_colours(image, n_colours=6, max_size=240)
    page = build_outline_page(
        quantized.labels,
        quantized.palette,
        min_region_area=20,
    )
    assert page.outline.mode == "RGB"
    assert page.legend.mode == "RGB"
    assert len(page.colour_numbers) == quantized.n_colours
    assert page.colour_numbers[0] == 1
    # Outlines should introduce some black pixels.
    arr = np.asarray(page.outline)
    assert (arr == 0).all(axis=2).any()
    assert len(page.regions) >= 1


def test_create_from_path_writes_outputs(sample_image: Path, tmp_path: Path) -> None:
    result = create_from_path(sample_image, n_colours=8, max_size=240)
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
