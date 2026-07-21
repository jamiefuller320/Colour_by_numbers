"""Reduce images to a limited colour palette."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


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
    # Two passes: box-ish blur via Gaussian approximates a cartoon filter.
    softened = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return softened.filter(ImageFilter.SMOOTH_MORE)


def upsample_labels(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a label map to ``(width, height)``."""
    width, height = size
    if labels.shape[0] == height and labels.shape[1] == width:
        return labels.astype(np.int32, copy=True)
    label_img = Image.fromarray(labels.astype(np.uint8), mode="L")
    label_img = label_img.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(label_img, dtype=np.int32)


def quantize_colours(
    image: Image.Image,
    *,
    n_colours: int = 16,
    max_size: int = 900,
    structure_size: int | None = 420,
    blur_radius: float = 2.0,
    seed: int = 42,
) -> QuantizedImage:
    """Quantize an RGB image to ``n_colours`` using median-cut.

    When ``structure_size`` is set, quantization stays on that smaller canvas
    so later region simplification can run cheaply on large shapes. Upsample
    with :func:`upsample_labels` after simplification.
    """
    del seed  # median-cut is deterministic for a given image
    if n_colours < 2 or n_colours > 64:
        raise ValueError("n_colours must be between 2 and 64.")

    output = resize_for_processing(image.convert("RGB"), max_size=max_size)
    struct_limit = structure_size if structure_size is not None else max_size
    working = resize_for_processing(output, max_size=struct_limit)
    working = prefilter_for_regions(working, blur_radius=blur_radius)
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

    order = np.argsort(palette_unsorted.mean(axis=1))
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    labels_sorted = remap[compact]
    palette = palette_unsorted[order]

    preview = Image.fromarray(palette[labels_sorted], mode="RGB")
    return QuantizedImage(labels=labels_sorted, palette=palette, preview=preview)


def preview_from_labels(labels: np.ndarray, palette: np.ndarray) -> Image.Image:
    """Rebuild an RGB preview from a (possibly simplified) label map."""
    return Image.fromarray(palette[labels], mode="RGB")
