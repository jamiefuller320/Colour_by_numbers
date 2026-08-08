"""Alternate colourways for editable full-colour plates.

Outlines stay fixed (same numbers / label map). Colour plates and legends are
re-rendered from ``labels + palette`` under a named colourway transform so a
pair can ship as natural / vivid / pop-art without regenerating geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Colourway:
    """Named remap applied to a base RGB palette."""

    id: str
    label: str
    description: str


COLOURWAY_NATURAL = Colourway(
    id="natural",
    label="Natural",
    description="Base plate colours as generated / prepared.",
)
COLOURWAY_VIVID = Colourway(
    id="vivid",
    label="Vivid",
    description="Boosted saturation while keeping value structure.",
)
COLOURWAY_POP_ART = Colourway(
    id="pop_art",
    label="Pop art",
    description="Snap mid/high chroma fills toward bold primaries.",
)
COLOURWAY_PASTEL = Colourway(
    id="pastel",
    label="Pastel",
    description="Lifted, softened fills for a gentler guide plate.",
)

COLOURWAYS: dict[str, Colourway] = {
    COLOURWAY_NATURAL.id: COLOURWAY_NATURAL,
    COLOURWAY_VIVID.id: COLOURWAY_VIVID,
    COLOURWAY_POP_ART.id: COLOURWAY_POP_ART,
    COLOURWAY_PASTEL.id: COLOURWAY_PASTEL,
}

# Bold anchors for pop-art remapping (RGB).
_POP_ANCHORS = np.array(
    [
        [20, 20, 20],
        [245, 245, 245],
        [230, 50, 50],
        [50, 90, 220],
        [250, 200, 40],
        [40, 180, 90],
        [240, 100, 30],
        [160, 50, 200],
        [30, 180, 200],
    ],
    dtype=np.float32,
)


def list_colourways() -> tuple[Colourway, ...]:
    return tuple(COLOURWAYS[key] for key in sorted(COLOURWAYS))


def resolve_colourway(name: str | None) -> Colourway:
    key = (name or "natural").strip().lower().replace("-", "_")
    if key not in COLOURWAYS:
        known = ", ".join(sorted(COLOURWAYS))
        raise ValueError(f"Unknown colourway {name!r}; choose one of: {known}")
    return COLOURWAYS[key]


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    chroma = maxc - minc
    s = np.where(maxc > 1e-6, chroma / maxc, 0.0)
    h = np.zeros_like(maxc)
    mask = chroma > 1e-6
    safe = np.where(mask, chroma, 1.0)
    rc = np.where(mask, (maxc - r) / safe, 0.0)
    gc = np.where(mask, (maxc - g) / safe, 0.0)
    bc = np.where(mask, (maxc - b) / safe, 0.0)
    h = np.where(mask & (maxc == r), (bc - gc) % 6.0, h)
    h = np.where(mask & (maxc == g), 2.0 + rc - bc, h)
    h = np.where(mask & (maxc == b), 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i % 6
    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=1)


def remap_palette(palette: np.ndarray, colourway: str | Colourway) -> np.ndarray:
    """Return an Nx3 uint8 palette under the named colourway."""
    way = colourway if isinstance(colourway, Colourway) else resolve_colourway(colourway)
    base = np.asarray(palette, dtype=np.uint8)
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("palette must be Nx3")
    if way.id == "natural":
        return base.copy()

    hsv = _rgb_to_hsv(base)
    if way.id == "vivid":
        hsv[:, 1] = np.clip(hsv[:, 1] * 1.35 + 0.05, 0.0, 1.0)
        hsv[:, 2] = np.clip(hsv[:, 2] * 1.05, 0.0, 1.0)
        out = _hsv_to_rgb(hsv)
        return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)

    if way.id == "pastel":
        hsv[:, 1] = np.clip(hsv[:, 1] * 0.55, 0.0, 1.0)
        hsv[:, 2] = np.clip(hsv[:, 2] * 0.35 + 0.65, 0.0, 1.0)
        out = _hsv_to_rgb(hsv)
        return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)

    if way.id == "pop_art":
        # Keep near-black / near-white; snap other colours to bold anchors.
        values = base.astype(np.float32)
        luminance = values.mean(axis=1)
        chroma = values.max(axis=1) - values.min(axis=1)
        out = values.copy()
        for i, (lum, ch) in enumerate(zip(luminance, chroma)):
            if lum < 35:
                out[i] = _POP_ANCHORS[0]
            elif lum > 230 and ch < 40:
                out[i] = _POP_ANCHORS[1]
            else:
                dists = ((values[i] - _POP_ANCHORS) ** 2).sum(axis=1)
                out[i] = _POP_ANCHORS[int(dists.argmin())]
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    raise ValueError(f"Unhandled colourway {way.id!r}")


def render_plate(labels: np.ndarray, palette: np.ndarray) -> Image.Image:
    """Rebuild a flat RGB plate from a label map + palette."""
    pal = np.asarray(palette, dtype=np.uint8)
    lab = np.asarray(labels, dtype=np.int32)
    if lab.min() < 0 or lab.max() >= len(pal):
        raise ValueError("labels contain indices outside the palette")
    return Image.fromarray(pal[lab], mode="RGB")


def render_colourway_plate(
    labels: np.ndarray,
    base_palette: np.ndarray,
    colourway: str | Colourway,
) -> tuple[Image.Image, np.ndarray]:
    """Return (plate image, remapped palette) for a colourway."""
    remapped = remap_palette(base_palette, colourway)
    return render_plate(labels, remapped), remapped
