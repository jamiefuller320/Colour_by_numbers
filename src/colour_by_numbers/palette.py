"""Standardised colouring-book palette and colour-distance helpers."""

from __future__ import annotations

import numpy as np

# Fixed 32-colour set spaced for distinct adjacent sections (crayon / paint style).
# Shadow / earth rungs are intentionally denser so fur, eyes, and low-light
# areas do not snap onto purple / teal / green just because those sit nearby
# in Lab when the active subset is thin.
STANDARD_PALETTE_32: np.ndarray = np.array(
    [
        # Neutrals / shadows (warm bias for animal subjects)
        [18, 18, 18],
        [42, 36, 32],
        [78, 72, 68],
        [150, 148, 145],
        [240, 238, 232],
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
        # Browns / earth (dark → light)
        [55, 30, 16],
        [90, 50, 30],
        [130, 85, 45],
        [175, 130, 75],
        [220, 185, 135],
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
        # Purples (kept sparse — easy to steal dark fur if over-selected)
        [110, 70, 160],
        [190, 155, 220],
        # Accents
        [255, 100, 50],
        [0, 150, 110],
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

# Soft categories that should keep warm darks instead of cool chromatic shadows.
# Broad animal supersets — fur, feathers, and hide all need earthy low-light mapping.
ANIMAL_CATEGORIES = frozenset(
    {
        "dogs",
        "cats",
        "horses",
        "birds",
        "wildlife",
        "animals",
        "pets",
        "farm animals",
        "mammals",
    }
)
EARTHY_CATEGORIES = ANIMAL_CATEGORIES


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


def _palette_chroma(palette: np.ndarray) -> np.ndarray:
    """Approximate chroma from Lab (√(a²+b²))."""
    lab = rgb_to_lab(palette)
    return np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)


def _palette_lightness(palette: np.ndarray) -> np.ndarray:
    return rgb_to_lab(palette)[:, 0]


def is_earthy_shadow_colour(rgb: np.ndarray) -> bool:
    """True for neutrals / browns suitable for fur and low-light modelling."""
    colour = np.asarray(rgb, dtype=np.uint8).reshape(3)
    lab = rgb_to_lab(colour).reshape(3)
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    chroma = (a * a + b * b) ** 0.5
    # Warm-ish neutrals and earths (brown sits in +a/+b).
    if chroma <= 18.0:
        return True
    if L <= 55.0 and a >= -5.0 and b >= 2.0 and chroma <= 55.0:
        return True
    return False


def earthy_shadow_mask(palette: np.ndarray) -> np.ndarray:
    """Boolean mask of palette rows safe for dark fur / shadow fills."""
    return np.array(
        [is_earthy_shadow_colour(row) for row in np.asarray(palette)],
        dtype=bool,
    )


def nearest_palette_indices(
    pixels: np.ndarray,
    palette: np.ndarray,
    *,
    category: str | None = None,
    dark_lightness: float = 42.0,
) -> np.ndarray:
    """Map HxWx3 RGB pixels to nearest palette index (Lab ΔE).

    For earthy subject categories (dogs, etc.), dark pixels are restricted to
    neutral / brown palette entries so low-light fur does not snap onto purple,
    teal, or green crayons.
    """
    h, w, _ = pixels.shape
    flat = pixels.reshape(-1, 3).astype(np.uint8)
    lab_pix = rgb_to_lab(flat)
    lab_pal = rgb_to_lab(palette)
    d2 = np.sum((lab_pix[:, None, :] - lab_pal[None, :, :]) ** 2, axis=-1)

    if category in EARTHY_CATEGORIES:
        safe = earthy_shadow_mask(palette)
        if safe.any() and not safe.all():
            dark = lab_pix[:, 0] < dark_lightness
            # Huge penalty for unsafe darks keeps argmin on earthy shadows.
            penalty = np.where(safe, 0.0, 1.0e6)
            d2 = d2.copy()
            d2[dark] = d2[dark] + penalty[None, :]

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


def _reserved_palette_indices(
    full_palette: np.ndarray,
    *,
    category: str | None,
    n_colours: int,
) -> list[int]:
    """Seed the active set with luminance / earth anchors for animal plates."""
    if category not in EARTHY_CATEGORIES or n_colours < 6:
        return []

    lightness = _palette_lightness(full_palette)
    safe = earthy_shadow_mask(full_palette)
    reserved: list[int] = []

    def _take(candidates: np.ndarray, limit: int = 1) -> None:
        for idx in candidates.tolist():
            if len(reserved) >= limit:
                break
            if int(idx) not in reserved:
                reserved.append(int(idx))

    # Near-black, warm mid-shadow, light paper.
    dark_safe = np.where(safe & (lightness < 30))[0]
    mid_safe = np.where(safe & (lightness >= 30) & (lightness < 60))[0]
    light = np.where(lightness > 85)[0]
    _take(dark_safe[np.argsort(lightness[dark_safe])], limit=2)
    _take(mid_safe[np.argsort(lightness[mid_safe])], limit=2)
    _take(light[np.argsort(-lightness[light])], limit=1)
    # One lighter tan/cream if room.
    tan = np.where(safe & (lightness >= 60) & (lightness <= 85))[0]
    _take(tan[np.argsort(lightness[tan])], limit=1)
    return reserved[: max(3, min(6, n_colours // 2))]


def select_active_palette(
    full_palette: np.ndarray,
    *,
    n_colours: int,
    image_rgb: np.ndarray | None = None,
    category: str | None = None,
) -> np.ndarray:
    """Choose up to ``n_colours`` entries from the standard set.

    When ``image_rgb`` is provided, prefer the palette colours that best cover
    the image's colour distribution (greedy farthest-point on used candidates).
    For earthy categories, reserve warm darks and demote cool chromatic paints
    unless they are genuinely dominant in the plate.
    """
    n = min(int(n_colours), len(full_palette))
    if n >= len(full_palette):
        return full_palette.copy()
    if image_rgb is None:
        # Even spacing through the fixed list.
        idx = np.linspace(0, len(full_palette) - 1, n, dtype=int)
        return full_palette[idx]

    # Score each palette colour by how often it is the nearest match.
    labels = nearest_palette_indices(
        image_rgb, full_palette, category=category
    )
    counts = np.bincount(labels.ravel(), minlength=len(full_palette)).astype(np.float64)

    if category in EARTHY_CATEGORIES:
        safe = earthy_shadow_mask(full_palette)
        chroma = _palette_chroma(full_palette)
        # Cool / vivid crayons need clear majority use before they earn a slot.
        demote = (~safe) & (chroma > 22.0)
        counts = counts.copy()
        counts[demote] *= 0.2

    order = np.argsort(-counts)
    chosen: list[int] = _reserved_palette_indices(
        full_palette, category=category, n_colours=n
    )
    dist = colour_distance_matrix(full_palette)
    min_sep = 12.0
    for idx in order:
        if len(chosen) >= n:
            break
        if counts[idx] <= 0 and chosen:
            continue
        if int(idx) in chosen:
            continue
        if all(dist[idx, c] >= min_sep for c in chosen):
            chosen.append(int(idx))
    # Fill if image was nearly monochrome or reservations left gaps.
    for idx in order:
        if len(chosen) >= n:
            break
        if int(idx) not in chosen:
            chosen.append(int(idx))
    # Still short (pathological) — take sequential leftovers.
    for idx in range(len(full_palette)):
        if len(chosen) >= n:
            break
        if idx not in chosen:
            chosen.append(idx)
    chosen_arr = np.array(sorted(chosen[:n]), dtype=np.int32)
    return full_palette[chosen_arr]
