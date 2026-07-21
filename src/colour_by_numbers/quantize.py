"""Reduce images to a limited colour palette."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QuantizedImage:
    """An image reduced to a fixed palette."""

    labels: np.ndarray  # HxW int labels in [0, n_colours)
    palette: np.ndarray  # Nx3 uint8 RGB colours
    preview: Image.Image  # RGB preview using the palette

    @property
    def n_colours(self) -> int:
        return int(self.palette.shape[0])


def resize_for_processing(
    image: Image.Image,
    *,
    max_size: int = 900,
) -> Image.Image:
    """Downscale large images while preserving aspect ratio."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_size:
        return image.copy()
    scale = max_size / longest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def quantize_colours(
    image: Image.Image,
    *,
    n_colours: int = 16,
    max_size: int = 900,
    seed: int = 42,
) -> QuantizedImage:
    """Quantize an RGB image to ``n_colours`` using median-cut.

    Returns palette-indexed labels plus an RGB preview. Colours are ordered
    from darkest to lightest so legend numbers stay stable for similar images.
    ``seed`` is accepted for API compatibility and reserved for future use.
    """
    del seed  # median-cut is deterministic for a given image
    if n_colours < 2 or n_colours > 64:
        raise ValueError("n_colours must be between 2 and 64.")

    working = resize_for_processing(image.convert("RGB"), max_size=max_size)
    paletted = working.quantize(
        colors=n_colours,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )

    raw_palette = paletted.getpalette() or []
    # Pillow stores up to 256 RGB triples; keep only colours actually used.
    labels = np.asarray(paletted, dtype=np.int32)
    used = np.unique(labels)
    full = np.array(raw_palette, dtype=np.uint8).reshape(-1, 3)
    # Guard against odd palette lengths.
    if full.shape[0] < int(used.max()) + 1:
        raise RuntimeError("Palette shorter than labelled colour indices.")

    palette_unsorted = full[used]
    # Remap labels to a compact 0..K-1 range, then sort dark → light.
    compact = np.zeros_like(labels)
    for new_idx, old_idx in enumerate(used):
        compact[labels == old_idx] = new_idx

    order = np.argsort(palette_unsorted.mean(axis=1))
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    labels_sorted = remap[compact]
    palette = palette_unsorted[order]

    preview = Image.fromarray(palette[labels_sorted], mode="RGB")
    return QuantizedImage(labels=labels_sorted, palette=palette, preview=preview)
