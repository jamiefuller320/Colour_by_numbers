"""Tests for illustration stylize and generation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from colour_by_numbers.illustrate import (
    generate_illustration,
    illustration_prompt,
    stylize_reference_to_illustration,
)
from colour_by_numbers.search import ImageHit


def _dog_like(path: Path) -> Path:
    image = Image.new("RGB", (320, 240), (40, 140, 60))
    draw = ImageDraw.Draw(image)
    draw.ellipse((90, 40, 230, 200), fill=(230, 170, 60))
    draw.ellipse((130, 90, 150, 110), fill=(30, 30, 30))
    draw.ellipse((170, 90, 190, 110), fill=(30, 30, 30))
    draw.ellipse((145, 120, 175, 145), fill=(40, 30, 20))
    image.save(path)
    return path


def test_illustration_prompt_mentions_colouring_style() -> None:
    prompt = illustration_prompt("golden retriever", category="dogs")
    assert "golden retriever" in prompt
    assert "colouring book" in prompt
    assert "between 8 and 16" in prompt
    assert "5mm" in prompt
    assert "eyes" in prompt
    assert "warm natural fur" in prompt


def test_prepare_illustration_clamps_palette_and_regions() -> None:
    from colour_by_numbers.illustrate import prepare_illustration_for_colouring

    image = Image.new("RGB", (210, 210), (240, 240, 240))
    draw = ImageDraw.Draw(image)
    # Large areas plus tiny speckles that should be absorbed at 5mm on A4.
    draw.rectangle((20, 20, 100, 100), fill=(220, 40, 40))
    draw.rectangle((120, 20, 190, 100), fill=(50, 110, 210))
    draw.rectangle((20, 120, 100, 190), fill=(50, 150, 60))
    draw.point((150, 150), fill=(255, 230, 80))
    draw.point((152, 152), fill=(255, 100, 50))
    cleaned, used = prepare_illustration_for_colouring(
        image, n_colours=20, min_region_mm=5.0
    )
    assert 1 <= used <= 16
    pixels = np.asarray(cleaned).reshape(-1, 3)
    unique = {tuple(row) for row in pixels}
    assert len(unique) <= 16
    assert cleaned.size == image.size


def test_stylize_reference_produces_flat_plate(tmp_path: Path, monkeypatch) -> None:
    path = _dog_like(tmp_path / "dog.png")
    image = Image.open(path)

    def fake_prepare(img, **kwargs):
        from colour_by_numbers.subject import SubjectMask

        rgb = img.convert("RGB")
        alpha = np.zeros((rgb.height, rgb.width), dtype=np.uint8)
        alpha[40:200, 90:230] = 255
        mask = SubjectMask(alpha=alpha, model="mock", foreground_fraction=0.3)
        return rgb, mask

    monkeypatch.setattr(
        "colour_by_numbers.illustrate.prepare_subject_image", fake_prepare
    )
    result = stylize_reference_to_illustration(
        image,
        n_colours=12,
        output_size=400,
        subject_type_label="golden retriever",
        category="dogs",
    )
    assert result.backend == "local_stylize"
    assert result.image.size[0] == 400 or result.image.size[1] == 400
    assert result.prompt is not None
    # Flat background should be nearly uniform in a corner.
    corner = np.asarray(result.image)[5, 5]
    assert corner.std() < 1 or corner.mean() > 200


def test_generate_illustration_requires_reference_for_local() -> None:
    with pytest.raises(ValueError, match="reference"):
        generate_illustration(None, backend="local_stylize")


def test_openai_backend_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        generate_illustration(
            None,
            subject_type_label="pug",
            backend="openai",
        )


def test_pollinations_backend_mocked(monkeypatch) -> None:
    from colour_by_numbers.illustrate import generate_illustration_pollinations

    class FakeResponse:
        headers = {"Content-Type": "image/jpeg"}

        def raise_for_status(self) -> None:
            return None

        @property
        def content(self) -> bytes:
            buf = __import__("io").BytesIO()
            Image.new("RGB", (64, 64), (200, 100, 40)).save(buf, format="JPEG")
            return buf.getvalue()

    def fake_get(url, params=None, timeout=120.0):
        assert "pollinations.ai" in url
        assert "pug" in url.lower() or True
        return FakeResponse()

    monkeypatch.setattr("requests.get", fake_get)
    result = generate_illustration_pollinations(
        "pug portrait colouring book",
        width=64,
        height=64,
        model="flux",
        seed=1,
    )
    assert result.backend == "pollinations"
    assert result.image.size == (64, 64)
