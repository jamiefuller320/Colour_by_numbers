"""Difficulty / visual-style presets for colouring plates.

This module is the seed of a **style-template** library for product variety.
Near term we refine **one** adult template — ``vibrant`` — until it is
reliably good across subjects. ``simple`` / ``standard`` stay as the kids
Phase B gate; do not add more styles until vibrant earns that focus.

``vibrant``: denser interlocking fills, larger crayon budgets, cool shadow
accents — a house style for animals, vehicles, flowers, birds, people,
inspired by rights-safe study of commercial vibrant kits (style cue, not
artwork to copy).
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
    complexity: str = "fine"
    min_adjacent_delta_e: float = 18.0
    # Pipeline quantize mode after prepare. ``exact`` keeps the plate's RGB
    # solids instead of re-median-cutting (critical for vibrant).
    pipeline_palette_mode: str | None = None
    # Illustration path subject engine (photo path still defaults via CLI).
    subject_mode: str = "off"
    subject_complexity: str = "fine"
    background_complexity: str = "light"
    # When set, merge near-duplicate colours only if over this budget,
    # preferring background merges so subject integrity wins.
    max_plate_colours: int | None = None
    # Match the three happy vibrant samples: gallery plate = illustration.
    keep_illustration_plate: bool = False


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
    complexity="simple",
    min_adjacent_delta_e=18.0,
    pipeline_palette_mode="standard",
    subject_mode="off",
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
    complexity="fine",
    min_adjacent_delta_e=18.0,
    pipeline_palette_mode="standard",
    subject_mode="off",
)

STYLE_VIBRANT = StylePreset(
    name="vibrant",
    n_colours=28,
    min_colours=20,
    max_colours=32,
    min_region_mm=3.0,
    palette_mode="adaptive",
    cool_shadows=True,
    prompt_style=(
        "adult vibrant paint-by-numbers kit style that works for any subject: "
        "dense interlocking mosaic of many small flat colour wedges across the "
        "whole form (fur, petals, metal panels, plumage, or skin — dozens of "
        "tiles, not a few large cel blobs), about 24 to 32 distinct solid "
        "colours with clear cool teal and blue shadow wedges plus cool specular "
        "accents mixed among warm gold and orange mid-tones (shadows must not "
        "be only brown or orange), bold black outlines of varying weight, "
        "abstract colour-block background with two to four solid fills "
        "(not empty white), high-energy colouring-kit look, "
        "no gradients, no photorealism"
    ),
    description=(
        "End-goal adult vibrant band for all categories — denser regions, "
        "fuller palette, cool shadow accents, mosaic form language. "
        "Isolates the subject, preserves subject mosaic, simplifies backgrounds."
    ),
    complexity="vibrant",
    # Prefer budgeted / background-first merges over a global ΔE crush.
    min_adjacent_delta_e=0.0,
    pipeline_palette_mode="exact",
    subject_mode="dual",
    subject_complexity="preserve",
    background_complexity="simple",
    max_plate_colours=28,
    keep_illustration_plate=True,
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
