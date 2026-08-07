"""Difficulty / visual-style presets for colouring plates.

``standard`` is the current Phase B kids/hand-colourable gate.
``vibrant`` is the long-term adult paint-by-numbers end goal: denser
interlocking fills, larger crayon budgets, and cooler shadow accents —
inspired by rights-safe study of commercial “vibrant portrait” kits, not
copied from any one product.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StylePreset:
    """Tunable knobs for one difficulty / visual band."""

    name: str
    n_colours: int
    min_colours: int
    max_colours: int
    min_region_mm: float
    palette_mode: str
    cool_shadows: bool
    prompt_style: str
    description: str


STYLE_SIMPLE = StylePreset(
    name="simple",
    n_colours=10,
    min_colours=6,
    max_colours=12,
    min_region_mm=10.0,
    palette_mode="book",
    cool_shadows=False,
    prompt_style=(
        "very large simple colour regions, minimal fur texture, "
        "few value steps, easy for young colourists"
    ),
    description="Fewer, larger fills for younger colourists.",
)

STYLE_STANDARD = StylePreset(
    name="standard",
    n_colours=16,
    min_colours=8,
    max_colours=16,
    min_region_mm=8.0,
    palette_mode="book",
    cool_shadows=False,
    prompt_style=(
        "clear flat cel fills with about 12 to 16 distinct solid colours "
        "and clear value steps between neighbouring parts"
    ),
    description="Current Phase B default — hand-colourable A4 with 8mm floor.",
)

STYLE_VIBRANT = StylePreset(
    name="vibrant",
    n_colours=28,
    min_colours=20,
    max_colours=32,
    min_region_mm=4.0,
    palette_mode="adaptive",
    cool_shadows=True,
    prompt_style=(
        "adult vibrant paint-by-numbers portrait style: dense interlocking "
        "flat colour wedges and mosaic fur shapes (not a few large blobs), "
        "about 24 to 32 distinct solid colours, strong value mosaic across "
        "the subject, cool teal and blue accents in shadows and highlights "
        "mixed with warm golds and oranges, bold black outlines of varying "
        "weight, optional abstract colour-block background (not empty white), "
        "high energy pop-art colouring-kit look, no gradients, no photorealism"
    ),
    description=(
        "End-goal adult vibrant band — denser regions, fuller palette, "
        "cool shadow accents."
    ),
)

STYLE_PRESETS: dict[str, StylePreset] = {
    STYLE_SIMPLE.name: STYLE_SIMPLE,
    STYLE_STANDARD.name: STYLE_STANDARD,
    STYLE_VIBRANT.name: STYLE_VIBRANT,
}

# Aliases
STYLE_PRESETS["kids"] = STYLE_SIMPLE
STYLE_PRESETS["default"] = STYLE_STANDARD
STYLE_PRESETS["phase_b"] = STYLE_STANDARD
STYLE_PRESETS["adult"] = STYLE_VIBRANT
STYLE_PRESETS["mosaic"] = STYLE_VIBRANT

DEFAULT_STYLE = "standard"


def resolve_style_preset(name: str | None = None) -> StylePreset:
    """Return a style preset by name (case-insensitive)."""
    key = (name or DEFAULT_STYLE).strip().lower().replace("-", "_")
    if key not in STYLE_PRESETS:
        known = ", ".join(sorted({p.name for p in STYLE_PRESETS.values()}))
        raise ValueError(f"Unknown style {name!r}; choose one of: {known}")
    return STYLE_PRESETS[key]


def list_style_names() -> tuple[str, ...]:
    return tuple(sorted({p.name for p in STYLE_PRESETS.values()}))
