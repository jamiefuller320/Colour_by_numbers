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
