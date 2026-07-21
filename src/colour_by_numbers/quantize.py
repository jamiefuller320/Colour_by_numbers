"""Reduce images to a limited colour palette."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from .palette import (
    STANDARD_PALETTE_32,
    nearest_palette_indices,
    select_active_palette,
)


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


def prefilter_for_regions(
    image: Image.Image,
    *,
    blur_radius: float = 2.0,
) -> Image.Image:
    """Soft-blur before quantization so flat areas dominate over photo noise."""
    if blur_radius <= 0:
        return image
    softened = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return softened.filter(ImageFilter.SMOOTH_MORE)


def upsample_labels(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a label map to ``(width, height)``."""
    width, height = size
    if labels.shape[0] == height and labels.shape[1] == width:
        return labels.astype(np.int32, copy=True)
    # Labels may exceed 255 when using a 32-colour palette — use I mode.
    label_img = Image.fromarray(labels.astype(np.int32), mode="I")
    label_img = label_img.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(label_img, dtype=np.int32)


def _sort_palette_dark_to_light(
    labels: np.ndarray, palette: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(palette.mean(axis=1))
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[labels], palette[order]


def quantize_colours(
    image: Image.Image,
    *,
    n_colours: int = 32,
    max_size: int = 900,
    structure_size: int | None = 420,
    blur_radius: float = 2.0,
    seed: int = 42,
    palette_mode: str = "standard",
    standard_palette: np.ndarray | None = None,
) -> QuantizedImage:
    """Quantize an RGB image to a limited palette.

    ``palette_mode``:
      - ``standard`` (default): map onto the fixed 32-colour colouring set
        (or a subset of size ``n_colours``)
      - ``free``: classic median-cut adaptive palette
    """
    del seed
    if n_colours < 2 or n_colours > 64:
        raise ValueError("n_colours must be between 2 and 64.")

    output = resize_for_processing(image.convert("RGB"), max_size=max_size)
    struct_limit = structure_size if structure_size is not None else max_size
    working = resize_for_processing(output, max_size=struct_limit)
    working = prefilter_for_regions(working, blur_radius=blur_radius)
    mode = palette_mode.lower().strip()

    if mode in {"standard", "fixed", "book"}:
        base = (
            STANDARD_PALETTE_32
            if standard_palette is None
            else np.asarray(standard_palette, dtype=np.uint8)
        )
        pixels = np.asarray(working, dtype=np.uint8)
        active = select_active_palette(base, n_colours=n_colours, image_rgb=pixels)
        labels = nearest_palette_indices(pixels, active)
        # Compact to colours actually used.
        used = np.unique(labels)
        compact = np.zeros_like(labels)
        for new_idx, old_idx in enumerate(used):
            compact[labels == old_idx] = new_idx
        palette = active[used]
        labels_sorted, palette = _sort_palette_dark_to_light(compact, palette)
        preview = Image.fromarray(palette[labels_sorted], mode="RGB")
        return QuantizedImage(labels=labels_sorted, palette=palette, preview=preview)

    if mode not in {"free", "adaptive", "mediancut"}:
        raise ValueError(f"Unknown palette_mode {palette_mode!r}")

    paletted = working.quantize(
        colors=n_colours,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    raw_palette = paletted.getpalette() or []
    labels = np.asarray(paletted, dtype=np.int32)
    used = np.unique(labels)
    full = np.array(raw_palette, dtype=np.uint8).reshape(-1, 3)
    if full.shape[0] < int(used.max()) + 1:
        raise RuntimeError("Palette shorter than labelled colour indices.")

    palette_unsorted = full[used]
    compact = np.zeros_like(labels)
    for new_idx, old_idx in enumerate(used):
        compact[labels == old_idx] = new_idx
    labels_sorted, palette = _sort_palette_dark_to_light(compact, palette_unsorted)
    preview = Image.fromarray(palette[labels_sorted], mode="RGB")
    return QuantizedImage(labels=labels_sorted, palette=palette, preview=preview)


def preview_from_labels(labels: np.ndarray, palette: np.ndarray) -> Image.Image:
    """Rebuild an RGB preview from a (possibly simplified) label map."""
    return Image.fromarray(palette[labels], mode="RGB")
