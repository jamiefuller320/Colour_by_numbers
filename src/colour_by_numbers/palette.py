"""Standardised colouring-book palette and colour-distance helpers."""

from __future__ import annotations

import numpy as np

# Fixed 32-colour set spaced for distinct adjacent sections (crayon / paint style).
STANDARD_PALETTE_32: np.ndarray = np.array(
    [
        # Neutrals
        [20, 20, 20],
        [80, 80, 80],
        [160, 160, 160],
        [240, 240, 240],
        # Warm yellows / golds / oranges
        [255, 230, 80],
        [245, 180, 40],
        [230, 120, 30],
        [200, 70, 20],
        # Reds / pinks
        [220, 40, 40],
        [180, 30, 70],
        [240, 120, 160],
        [255, 190, 200],
        # Browns / earth
        [90, 50, 30],
        [140, 90, 45],
        [190, 140, 80],
        [220, 190, 140],
        # Greens
        [30, 90, 40],
        [50, 150, 60],
        [120, 200, 80],
        [180, 220, 120],
        # Cyans / teals
        [20, 120, 130],
        [40, 180, 190],
        [140, 220, 220],
        # Blues
        [30, 60, 150],
        [50, 110, 210],
        [130, 180, 240],
        # Purples
        [80, 40, 140],
        [140, 80, 200],
        [200, 160, 230],
        # Accents
        [255, 250, 180],
        [255, 100, 50],
        [0, 160, 100],
    ],
    dtype=np.uint8,
)

assert STANDARD_PALETTE_32.shape == (32, 3)

# Illustration / colouring-book plates use a tighter crayon budget.
MIN_N_COLOURS = 8
MAX_N_COLOURS = 16
DEFAULT_N_COLOURS = 32
DEFAULT_ILLUSTRATION_COLOURS = 12
DEFAULT_MIN_ADJACENT_DELTA_E = 18.0
DEFAULT_MIN_SUBJECT_BG_CONTRAST = 22.0


def clamp_n_colours(
    n_colours: int,
    *,
    minimum: int = MIN_N_COLOURS,
    maximum: int = MAX_N_COLOURS,
) -> int:
    """Clamp a colour count into the allowed illustration range (default 8–16)."""
    return max(int(minimum), min(int(maximum), int(n_colours)))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 (…,3) to CIE Lab float64."""
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    mask = arr <= 0.04045
    linear = np.empty_like(arr)
    linear[mask] = arr[mask] / 12.92
    linear[~mask] = ((arr[~mask] + 0.055) / 1.055) ** 2.4
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = linear @ m.T
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    x = xyz / white
    eps = 216 / 24389
    kappa = 24389 / 27

    def f(t: np.ndarray) -> np.ndarray:
        out = np.empty_like(t)
        small = t <= eps
        out[small] = (kappa * t[small] + 16.0) / 116.0
        out[~small] = np.cbrt(t[~small])
        return out

    fx, fy, fz = f(x[..., 0]), f(x[..., 1]), f(x[..., 2])
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def delta_e_rgb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """CIE76 ΔE between RGB uint8 colours (broadcastable on trailing axis)."""
    lab_a = rgb_to_lab(np.asarray(a, dtype=np.uint8))
    lab_b = rgb_to_lab(np.asarray(b, dtype=np.uint8))
    return np.sqrt(np.sum((lab_a - lab_b) ** 2, axis=-1))


def colour_distance_matrix(palette: np.ndarray) -> np.ndarray:
    """NxN ΔE matrix for an RGB palette."""
    lab = rgb_to_lab(palette)
    diff = lab[:, None, :] - lab[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def nearest_palette_indices(
    pixels: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    """Map HxWx3 RGB pixels to nearest palette index (Lab ΔE)."""
    h, w, _ = pixels.shape
    flat = pixels.reshape(-1, 3).astype(np.uint8)
    lab_pix = rgb_to_lab(flat)
    lab_pal = rgb_to_lab(palette)
    d2 = np.sum((lab_pix[:, None, :] - lab_pal[None, :, :]) ** 2, axis=-1)
    return np.argmin(d2, axis=1).astype(np.int32).reshape(h, w)


def contrast_delta_e(rgb_a: np.ndarray, rgb_b: np.ndarray) -> float:
    """Scalar ΔE between two RGB colours."""
    a = np.asarray(rgb_a, dtype=np.uint8).reshape(3)
    b = np.asarray(rgb_b, dtype=np.uint8).reshape(3)
    return float(delta_e_rgb(a, b))


def mean_rgb(pixels: np.ndarray) -> np.ndarray:
    """Mean RGB of an (N,3) pixel set as float64."""
    arr = np.asarray(pixels, dtype=np.float64).reshape(-1, 3)
    if arr.size == 0:
        return np.array([128.0, 128.0, 128.0])
    return arr.mean(axis=0)


def select_active_palette(
    full_palette: np.ndarray,
    *,
    n_colours: int,
    image_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Choose up to ``n_colours`` entries from the standard set.

    When ``image_rgb`` is provided, prefer the palette colours that best cover
    the image's colour distribution (greedy farthest-point on used candidates).
    """
    n = min(int(n_colours), len(full_palette))
    if n >= len(full_palette):
        return full_palette.copy()
    if image_rgb is None:
        # Even spacing through the fixed list.
        idx = np.linspace(0, len(full_palette) - 1, n, dtype=int)
        return full_palette[idx]

    # Score each palette colour by how often it is the nearest match.
    labels = nearest_palette_indices(image_rgb, full_palette)
    counts = np.bincount(labels.ravel(), minlength=len(full_palette))
    # Take the top-n by usage, but drop near-duplicates via greedy ΔE spacing.
    order = np.argsort(-counts)
    chosen: list[int] = []
    dist = colour_distance_matrix(full_palette)
    min_sep = 12.0
    for idx in order:
        if counts[idx] <= 0 and chosen:
            continue
        if all(dist[idx, c] >= min_sep for c in chosen):
            chosen.append(int(idx))
        if len(chosen) >= n:
            break
    # Fill if image was nearly monochrome.
    for idx in order:
        if len(chosen) >= n:
            break
        if int(idx) not in chosen:
            chosen.append(int(idx))
    chosen_arr = np.array(sorted(chosen), dtype=np.int32)
    return full_palette[chosen_arr]
