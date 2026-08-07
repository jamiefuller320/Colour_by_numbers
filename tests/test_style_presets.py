"""Tests for simple / standard / vibrant style presets."""

from __future__ import annotations

import pytest

from colour_by_numbers.illustrate import illustration_prompt
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
    assert STYLE_VIBRANT.min_region_mm == 4.0


def test_standard_prompt_stays_kids_cel() -> None:
    prompt = illustration_prompt(
        "golden retriever", category="dogs", style_preset="standard"
    )
    assert "children's colouring book" in prompt
    assert "white background" in prompt
    assert STYLE_STANDARD.max_colours == 16
