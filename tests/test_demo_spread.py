"""Tests for demo spread generation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from colour_by_numbers.demo_spread import build_demo_spread
from colour_by_numbers.pipeline import DEMO_SUBJECT_COMPARE, create_colour_by_numbers


def _sample(path: Path) -> Path:
    image = Image.new("RGB", (200, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 20, 90, 100), fill=(200, 40, 40))
    draw.rectangle((110, 25, 180, 115), fill=(40, 100, 200))
    image.save(path)
    return path


def test_fine_off_runs_without_subject_engine(tmp_path: Path) -> None:
    image = Image.open(_sample(tmp_path / "sample.png"))
    result = create_colour_by_numbers(
        image, n_colours=8, max_size=200, complexity="fine", subject_mode="off"
    )
    assert result.subject_mode == "off"
    assert result.prepared is None
    assert result.page.outline.size[0] > 0


def test_demo_spread_compares_subject_modes(tmp_path: Path, monkeypatch) -> None:
    path = _sample(tmp_path / "sample.png")
    image = Image.open(path)

    def fake_prepare(img, *, mode="isolate", **kwargs):
        if mode in {"off", "none"}:
            return img.convert("RGB"), None
        # Pretend we cropped to the red circle area.
        cropped = img.crop((10, 10, 100, 110)).convert("RGB")
        return cropped, None

    monkeypatch.setattr(
        "colour_by_numbers.pipeline.prepare_subject_image", fake_prepare
    )
    colours, outlines, stats = build_demo_spread(
        image, n_colours=8, max_size=200, tile_width=120, complexity="fine"
    )
    assert set(stats) == set(DEMO_SUBJECT_COMPARE)
    assert colours.width > 400
    assert outlines.width > 300
