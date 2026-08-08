"""Subject/background colour contrast helpers."""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from .palette import contrast_delta_e, mean_rgb, rgb_to_lab
from .quantize import resize_for_processing
from .subject import SubjectMask, align_mask, harden_mask

logger = logging.getLogger(__name__)


def _border_mask(height: int, width: int, border_frac: float = 0.12) -> np.ndarray:
    by = max(1, int(height * border_frac))
    bx = max(1, int(width * border_frac))
    mask = np.zeros((height, width), dtype=bool)
    mask[:by, :] = True
    mask[-by:, :] = True
    mask[:, :bx] = True
    mask[:, -bx:] = True
    return mask


def estimate_centre_border_contrast(image: Image.Image) -> float:
    """Fast ΔE between centre crop and border ring (no rembg)."""
    rgb = np.asarray(
        resize_for_processing(image.convert("RGB"), max_size=320), dtype=np.uint8
    )
    h, w, _ = rgb.shape
    border = _border_mask(h, w)
    cy0, cy1 = int(h * 0.25), int(h * 0.75)
    cx0, cx1 = int(w * 0.25), int(w * 0.75)
    centre = np.zeros((h, w), dtype=bool)
    centre[cy0:cy1, cx0:cx1] = True
    centre &= ~border
    if not centre.any() or not border.any():
        return 0.0
    return contrast_delta_e(mean_rgb(rgb[centre]), mean_rgb(rgb[border]))


def subject_background_contrast(
    image: Image.Image,
    mask: SubjectMask | None = None,
) -> float:
    """ΔE between mean subject colour and mean background colour."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if mask is None:
        return estimate_centre_border_contrast(image)
    mask = align_mask(harden_mask(mask), image.size, firm=True)
    fg = mask.binary
    bg = ~fg
    if not fg.any() or not bg.any():
        return 0.0
    return contrast_delta_e(mean_rgb(rgb[fg]), mean_rgb(rgb[bg]))


def refine_mask_by_colour(
    image: Image.Image,
    mask: SubjectMask,
    *,
    band_px: int = 6,
    min_advantage: float = 2.0,
) -> SubjectMask:
    """Snap soft silhouette pixels using subject vs background colour.

    In a band around the firm edge, assign each pixel to subject if it is
    closer (Lab) to the subject mean than the background mean. Helps golden
    fur against green foliage where rembg mattes are fuzzy.
    """
    from scipy import ndimage

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = align_mask(harden_mask(mask), image.size, firm=True)
    hard = mask.binary
    if not hard.any() or hard.all():
        return mask

    dil = ndimage.binary_dilation(hard, iterations=band_px)
    ero = ndimage.binary_erosion(hard, iterations=max(1, band_px // 2))
    band = dil & ~ero
    if not band.any():
        return mask

    subj_mean = mean_rgb(rgb[hard])
    bg_mean = mean_rgb(rgb[~hard])
    lab = rgb_to_lab(rgb)
    subj_u8 = np.clip(np.round(subj_mean), 0, 255).astype(np.uint8).reshape(1, 3)
    bg_u8 = np.clip(np.round(bg_mean), 0, 255).astype(np.uint8).reshape(1, 3)
    subj_lab = rgb_to_lab(subj_u8)[0]
    bg_lab = rgb_to_lab(bg_u8)[0]
    d_subj = np.sqrt(np.sum((lab - subj_lab) ** 2, axis=-1))
    d_bg = np.sqrt(np.sum((lab - bg_lab) ** 2, axis=-1))

    refined = hard.copy()
    # Prefer subject when clearly closer; prefer background when clearly closer.
    refined[band & (d_subj + min_advantage < d_bg)] = True
    refined[band & (d_bg + min_advantage < d_subj)] = False

    alpha = np.where(refined, 255, 0).astype(np.uint8)
    logger.info(
        "Colour-refined mask: fg %.1f%% → %.1f%%",
        100 * hard.mean(),
        100 * refined.mean(),
    )
    return SubjectMask(
        alpha=alpha,
        model=mask.model,
        foreground_fraction=float(refined.mean()),
    )


def _contrasting_background_rgb(
    subject_mean: np.ndarray,
    subject_palette: np.ndarray,
    *,
    min_delta_e: float,
) -> np.ndarray:
    """Pick a flat background RGB at least ``min_delta_e`` from subject paints."""
    from .palette import colour_distance_matrix

    mean = np.asarray(subject_mean, dtype=np.float64).reshape(3)
    mean_l = float(0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2])
    # Prefer cool pale behind warm subjects (typical animals); flip if subject is cool/dark.
    if mean_l >= 90:
        candidates = np.array(
            [
                [70, 95, 125],
                [55, 75, 105],
                [100, 120, 140],
                [45, 55, 70],
            ],
            dtype=np.uint8,
        )
    elif mean_l <= 50:
        candidates = np.array(
            [
                [230, 236, 245],
                [210, 220, 235],
                [245, 240, 230],
                [200, 210, 220],
            ],
            dtype=np.uint8,
        )
    else:
        candidates = np.array(
            [
                [210, 222, 235],
                [180, 200, 220],
                [140, 165, 195],
                [230, 236, 245],
                [90, 115, 145],
            ],
            dtype=np.uint8,
        )

    subj = np.asarray(subject_palette, dtype=np.uint8).reshape(-1, 3)
    if subj.size == 0:
        return candidates[0]
    best = candidates[0]
    best_score = -1.0
    for cand in candidates:
        stacked = np.vstack([cand, subj])
        dist = colour_distance_matrix(stacked)[0, 1:]
        score = float(dist.min())
        if score >= min_delta_e and score > best_score:
            best = cand
            best_score = score
    if best_score < 0:
        # Force a cool pale / dark pair until separation is met.
        fallback = np.array([220, 230, 242], dtype=np.uint8)
        stacked = np.vstack([fallback, subj])
        if float(colour_distance_matrix(stacked)[0, 1:].min()) < min_delta_e:
            fallback = np.array([50, 70, 100], dtype=np.uint8)
        return fallback
    return best


def enforce_subject_background_separation(
    labels: np.ndarray,
    palette: np.ndarray,
    subject_mask: np.ndarray,
    *,
    min_delta_e: float = 18.0,
    separation_mm: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep background paints distinct from the subject near the silhouette.

    Two-layer rule:
    1. A paint index may not appear in both subject and background (splits
       distant subject-coloured islands as well as edge bleed).
    2. Within ``separation_mm`` of the subject on A4, a background fill may not
       use a *different* colour within ``min_delta_e`` of any subject paint.

    Offending background pixels are remapped onto a contrasting flat swatch;
    subject pixels of a previously shared paint are left intact.
    """
    from scipy import ndimage

    from .print_resolution import min_region_size_for_a4_mm
    from .simplify import compact_palette

    if min_delta_e <= 0 or separation_mm <= 0:
        return labels.astype(np.int32, copy=True), palette.astype(np.uint8, copy=True)

    work = labels.astype(np.int32, copy=True)
    pal = palette.astype(np.uint8, copy=True)
    mask = subject_mask.astype(bool)
    if mask.shape != work.shape or not mask.any() or bool(mask.all()):
        return work, pal

    h, w = work.shape
    band_px = max(1, min_region_size_for_a4_mm(w, h, min_mm=separation_mm).min_width_px)
    dilated = ndimage.binary_dilation(mask, iterations=int(band_px))
    band = dilated & ~mask

    from .palette import colour_distance_matrix

    subj_colours = {int(c) for c in np.unique(work[mask])}
    if not subj_colours:
        return work, pal
    subj_pal = pal[sorted(subj_colours)]
    subj_mean = mean_rgb(pal[work[mask]])
    dist = colour_distance_matrix(pal)

    offenders: set[int] = set()
    # Global: never share a paint index across subject and background.
    bg_colours = {int(c) for c in np.unique(work[~mask])}
    offenders.update(subj_colours & bg_colours)
    # Local: within the separation band, forbid near-twin (ΔE) bg paints.
    if band.any():
        for colour in np.unique(work[band]):
            c = int(colour)
            if c in subj_colours:
                continue
            min_de = min(float(dist[c, s]) for s in subj_colours)
            if min_de < float(min_delta_e):
                offenders.add(c)

    if not offenders:
        return work, pal

    # One contrasting swatch for all remapped background paints.
    target = _contrasting_background_rgb(
        subj_mean, subj_pal, min_delta_e=float(min_delta_e)
    )
    # Reuse an existing palette entry if it's already far enough from the subject.
    new_idx = None
    for idx, rgb in enumerate(pal):
        if idx in subj_colours or idx in offenders:
            continue
        stacked = np.vstack([rgb, subj_pal])
        if float(colour_distance_matrix(stacked)[0, 1:].min()) >= float(min_delta_e):
            new_idx = int(idx)
            break
    if new_idx is None:
        pal = np.vstack([pal, target.reshape(1, 3)]).astype(np.uint8)
        new_idx = int(pal.shape[0] - 1)

    remapped = 0
    for c in sorted(offenders):
        # Remap all background uses (keep subject pixels of a shared paint).
        hit = (~mask) & (work == c)
        if hit.any():
            work[hit] = new_idx
            remapped += int(hit.sum())

    work, pal = compact_palette(work, pal)
    logger.info(
        "Subject/bg separation: remapped %d bg px (shared paints + ≈%.1fmm band) "
        "using ΔE≥%.1f (%dpx band)",
        remapped,
        separation_mm,
        min_delta_e,
        band_px,
    )
    return work, pal
