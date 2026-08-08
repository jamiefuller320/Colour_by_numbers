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


def _warm_fur_pixels(rgb: np.ndarray) -> np.ndarray:
    """True for warm / cream coat-like pixels that must not be carved to bg."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    warm = (r > g + 6) & (r > b + 8) & (r > 90)
    # Pale cream highlights: high L, still warmer than cool sheet greys.
    cream = (r >= 170) & (g >= 140) & (b >= 100) & (r >= b + 10) & (r >= g - 5)
    return warm | cream


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

    Warm / cream coat pixels are never carved out — pale fur against a light
    sheet is often closer to the background mean and would otherwise notch
    the silhouette (then cool separation paints those notches blue).
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
    warm = _warm_fur_pixels(rgb)

    refined = hard.copy()
    # Prefer subject when clearly closer; prefer background when clearly closer.
    refined[band & (d_subj + min_advantage < d_bg)] = True
    carve = band & (d_bg + min_advantage < d_subj) & ~warm
    refined[carve] = False
    # Pull warm edge pixels that rembg dropped back into the subject.
    refined[band & warm & (d_subj < d_bg + 8.0)] = True

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


def heal_mask_notches(
    mask: SubjectMask,
    *,
    close_iterations: int = 8,
    max_fill_px: int | None = None,
) -> SubjectMask:
    """Fill small concave bites in the subject silhouette.

    Open crown notches are not closed by morphology (they open to the
    exterior). Instead, fill compact background pockets that see subject on
    opposite sides (or on ≥3 sides) within ``close_iterations`` px. Long
    corridors (leg gaps) are rejected by a bbox-extent cap.
    """
    from scipy import ndimage

    hard = harden_mask(mask).binary
    if not hard.any() or hard.all():
        return mask
    r = max(1, int(close_iterations))
    left = np.zeros_like(hard)
    right = np.zeros_like(hard)
    up = np.zeros_like(hard)
    down = np.zeros_like(hard)
    for i in range(1, r + 1):
        left[:, i:] |= hard[:, :-i]
        right[:, :-i] |= hard[:, i:]
        up[i:, :] |= hard[:-i, :]
        down[:-i, :] |= hard[i:, :]
    pinch = (left & right) | (up & down)
    dirs = left.astype(np.uint8) + right + up + down
    cupped = (~hard) & (pinch | (dirs >= 3))
    # Tiny internal pinholes only — large closing bridges leg gaps.
    closed = ndimage.binary_closing(hard, iterations=1)
    candidates = cupped | (closed & ~hard)
    if not candidates.any():
        return mask
    limit = (
        int(max_fill_px)
        if max_fill_px is not None
        else int(r * r * 4)
    )
    max_extent = 2 * r + 2
    labelled, n = ndimage.label(candidates)
    healed = hard.copy()
    filled = 0
    for comp_id in range(1, n + 1):
        comp = labelled == comp_id
        area = int(comp.sum())
        if area <= 0 or area > limit:
            continue
        ys, xs = np.where(comp)
        if (int(ys.max()) - int(ys.min()) + 1) > max_extent:
            continue
        if (int(xs.max()) - int(xs.min()) + 1) > max_extent:
            continue
        healed |= comp
        filled += area
    if filled == 0:
        return mask
    alpha = np.where(healed, 255, 0).astype(np.uint8)
    logger.info(
        "Healed subject mask notches: +%d px (fg %.1f%% → %.1f%%)",
        filled,
        100 * hard.mean(),
        100 * healed.mean(),
    )
    return SubjectMask(
        alpha=alpha,
        model=mask.model,
        foreground_fraction=float(healed.mean()),
    )


def _needs_body_recovery(rgb: np.ndarray, hard: np.ndarray) -> bool:
    """True when many subject-like pixels under the chin sit outside the mask.

    Catches pale-chested portraits where rembg keeps the head (and maybe a
    thin spike) but drops the torso — without swallowing full-body ground
    planes that merely sit under a standing animal.
    """
    h, w = hard.shape
    ys, xs = np.nonzero(hard)
    if len(ys) == 0:
        return False
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    y_med = int(np.median(ys))
    pad_x = int(0.08 * max(1, x1 - x0 + 1))
    x0e, x1e = max(0, x0 - pad_x), min(w, x1 + pad_x + 1)
    under = np.zeros_like(hard)
    under[y_med:, x0e:x1e] = True
    zone = under & ~hard
    if int(zone.sum()) < 80:
        return False

    by, bx = max(2, h // 15), max(2, w // 15)
    ring = np.zeros_like(hard)
    ring[:by, :] = ring[-by:, :] = ring[:, :bx] = ring[:, -bx:] = True
    ring &= ~hard
    lab = rgb_to_lab(rgb)
    subj_mean = mean_rgb(rgb[hard])
    bg_mean = mean_rgb(rgb[ring]) if ring.any() else mean_rgb(rgb[~hard])
    subj_lab = rgb_to_lab(
        np.clip(np.round(subj_mean), 0, 255).astype(np.uint8).reshape(1, 3)
    )[0]
    bg_lab = rgb_to_lab(
        np.clip(np.round(bg_mean), 0, 255).astype(np.uint8).reshape(1, 3)
    )[0]
    d_subj = np.sqrt(np.sum((lab - subj_lab) ** 2, axis=-1))
    d_bg = np.sqrt(np.sum((lab - bg_lab) ** 2, axis=-1))
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    warm = (r > g + 8) & (r > b + 8) & (r > 100)
    # Subject-like: nearer the subject than the border, or warm fur accents.
    like = (d_subj + 1.5 < d_bg) | (warm & (d_subj < 36))
    like_under = like & under
    total = int(like_under.sum())
    if total < 80:
        return False
    outside = int((like_under & ~hard).sum())
    # Portrait head-fills: rembg covers the mid frame densely but leaves the
    # lower third sparse. Full-body side views have lower mid fill and must
    # not trigger (or the ground plane gets swallowed).
    lower_frac = float(hard[int(h * 0.65) :, :].mean())
    mid_frac = float(hard[int(h * 0.20) : int(h * 0.55), :].mean())
    head_heavy = mid_frac >= 0.70 and lower_frac < 0.35
    return head_heavy and outside / total >= 0.28


def recover_subject_mask(
    image: Image.Image,
    mask: SubjectMask,
    *,
    close_iterations: int = 8,
) -> SubjectMask:
    """Fill holes and recover body regions rembg dropped as background.

    Always fills interior holes. When many subject-like pixels under the chin
    were excluded (pale chests on animal portraits), reclaim those pixels so
    the torso is not flattened into the background wash.
    """
    from scipy import ndimage

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = align_mask(harden_mask(mask), image.size, firm=True)
    hard = mask.binary
    if not hard.any() or hard.all():
        return mask

    h, w = hard.shape
    closed = ndimage.binary_closing(hard, iterations=int(close_iterations))
    filled = ndimage.binary_fill_holes(closed)

    # Interior background pockets (not touching the frame) → subject.
    bg = ~filled
    labelled, count = ndimage.label(bg)
    if count > 0:
        border = np.zeros_like(filled)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        border_ids = {int(v) for v in np.unique(labelled[border]) if v}
        for idx in range(1, count + 1):
            if idx not in border_ids:
                filled[labelled == idx] = True

    grown = filled
    if _needs_body_recovery(rgb, filled):
        ys, xs = np.nonzero(filled)
        y_med = int(np.median(ys))
        x0, x1 = int(xs.min()), int(xs.max())
        pad_x = int(0.12 * max(1, x1 - x0 + 1))
        x0e, x1e = max(0, x0 - pad_x), min(w, x1 + pad_x + 1)
        under = np.zeros_like(filled)
        under[y_med:, x0e:x1e] = True

        by, bx = max(2, h // 15), max(2, w // 15)
        ring = np.zeros_like(filled)
        ring[:by, :] = ring[-by:, :] = ring[:, :bx] = ring[:, -bx:] = True
        ring &= ~filled
        lab = rgb_to_lab(rgb)
        bg_mean = mean_rgb(rgb[ring]) if ring.any() else mean_rgb(rgb[~filled])
        subj_mean = mean_rgb(rgb[filled])
        bg_lab = rgb_to_lab(
            np.clip(np.round(bg_mean), 0, 255).astype(np.uint8).reshape(1, 3)
        )[0]
        subj_lab = rgb_to_lab(
            np.clip(np.round(subj_mean), 0, 255).astype(np.uint8).reshape(1, 3)
        )[0]
        d_bg = np.sqrt(np.sum((lab - bg_lab) ** 2, axis=-1))
        d_subj = np.sqrt(np.sum((lab - subj_lab) ** 2, axis=-1))
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        warm = (r > g + 8) & (r > b + 8) & (r > 100)
        # Only reclaim subject-like / warm torso pixels — never the whole ground.
        # Slightly permissive on d_subj vs d_bg so pale chest wash under a
        # portrait head can rejoin before hole-fill bridges the warm accents.
        like = (d_subj < d_bg + 6.0) | (warm & (d_subj < 42))
        under_keep = under & like
        # Geodesic grow from the current silhouette into those candidates.
        grown = filled.copy()
        for _ in range(max(40, min(h, w) // 8)):
            edge = ndimage.binary_dilation(grown, iterations=1) & ~grown
            take = edge & under_keep
            if not take.any():
                break
            grown |= take
        # Portrait torso completion: fill the under-chin column, but drop any
        # component that touches the left/right frame (true side background).
        inset = int(0.05 * max(1, x1e - x0e))
        col = np.zeros_like(filled)
        col[y_med:, max(0, x0e + inset) : max(0, x1e - inset)] = True
        extra = col & ~grown
        if extra.any():
            labelled_extra, n_extra = ndimage.label(extra)
            keep_extra = np.zeros_like(extra)
            for idx in range(1, n_extra + 1):
                comp = labelled_extra == idx
                if comp[:, 0].any() or comp[:, -1].any():
                    continue
                keep_extra |= comp
            grown |= keep_extra
        grown = ndimage.binary_fill_holes(
            ndimage.binary_closing(grown, iterations=max(8, close_iterations // 2))
        )
        logger.info(
            "Recovered truncated subject mask: fg %.1f%% → %.1f%%",
            100.0 * hard.mean(),
            100.0 * grown.mean(),
        )
    elif float(grown.mean()) > float(hard.mean()) + 0.005:
        logger.info(
            "Filled subject-mask holes: fg %.1f%% → %.1f%%",
            100.0 * hard.mean(),
            100.0 * grown.mean(),
        )

    labelled, count = ndimage.label(grown)
    if count > 1:
        sizes = np.bincount(labelled.ravel())
        sizes[0] = 0
        grown = labelled == int(sizes.argmax())

    alpha = np.where(grown, 255, 0).astype(np.uint8)
    return SubjectMask(
        alpha=alpha,
        model=mask.model,
        foreground_fraction=float(grown.mean()),
    )


def recolour_reclaimed_subject_pixels(
    image: Image.Image,
    original_mask: SubjectMask,
    recovered_mask: SubjectMask,
) -> Image.Image:
    """Warm up pale torso pixels that rembg dropped then recovery reclaimed.

    Portrait chests often match the background wash in the source illustration.
    Once they are back inside the subject mask, snap them toward the nearest
    core-subject colour so they read as fur rather than leftover background.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    original = align_mask(harden_mask(original_mask), image.size, firm=True).binary
    recovered = align_mask(harden_mask(recovered_mask), image.size, firm=True).binary
    reclaimed = recovered & ~original
    core = original & recovered
    if not reclaimed.any() or not core.any():
        return Image.fromarray(rgb, mode="RGB")

    lab = rgb_to_lab(rgb)
    # Sample core subject labs.
    ys, xs = np.nonzero(core)
    idx = np.linspace(0, len(ys) - 1, min(48, len(ys))).astype(int)
    samples = lab[ys[idx], xs[idx]]
    flat_reclaim = lab[reclaimed]
    # Nearest sample per reclaimed pixel.
    dmin = np.full(flat_reclaim.shape[0], np.inf)
    nearest = np.zeros(flat_reclaim.shape[0], dtype=np.int32)
    for i in range(0, len(samples), 8):
        chunk = samples[i : i + 8]
        d = np.sqrt(((flat_reclaim[:, None, :] - chunk[None, :, :]) ** 2).sum(-1))
        local = d.argmin(axis=1)
        local_d = d[np.arange(len(flat_reclaim)), local]
        better = local_d < dmin
        dmin[better] = local_d[better]
        nearest[better] = i + local[better]
    # Only recolour pixels that were still bg-like (far from core).
    far = dmin > 12.0
    if not far.any():
        return Image.fromarray(rgb, mode="RGB")
    target_yx = (ys[idx][nearest[far]], xs[idx][nearest[far]])
    ry, rx = np.nonzero(reclaimed)
    ry, rx = ry[far], rx[far]
    rgb[ry, rx] = rgb[target_yx[0], target_yx[1]]
    logger.info("Recoloured %d reclaimed subject px toward core fur tones", int(far.sum()))
    return Image.fromarray(rgb, mode="RGB")


def _contrasting_background_family(
    subject_mean: np.ndarray,
    subject_palette: np.ndarray,
    *,
    min_delta_e: float,
) -> np.ndarray:
    """Return 2–4 cool abstract swatches, all ≥ min_delta_e from the subject."""
    from .palette import colour_distance_matrix

    mean = np.asarray(subject_mean, dtype=np.float64).reshape(3)
    mean_l = float(0.2126 * mean[0] + 0.7152 * mean[1] + 0.0722 * mean[2])
    # Avoid near-black behind warm animals with dark paw/nose paints — that
    # recreates the p04 "dark merges with dark" failure mode.
    if mean_l >= 90:
        pool = np.array(
            [
                [70, 95, 125],
                [55, 75, 105],
                [100, 120, 140],
                [120, 145, 170],
                [45, 65, 90],
            ],
            dtype=np.uint8,
        )
    elif mean_l <= 50:
        # Keep the pool cool — cream/warm greys compete with animal coats.
        pool = np.array(
            [
                [230, 236, 245],
                [210, 220, 235],
                [200, 210, 220],
                [180, 195, 215],
                [160, 180, 205],
            ],
            dtype=np.uint8,
        )
    else:
        pool = np.array(
            [
                [210, 222, 235],
                [180, 200, 220],
                [140, 165, 195],
                [230, 236, 245],
                [100, 125, 155],
                [160, 185, 210],
            ],
            dtype=np.uint8,
        )

    subj = np.asarray(subject_palette, dtype=np.uint8).reshape(-1, 3)
    kept: list[np.ndarray] = []
    for cand in pool:
        if subj.size == 0:
            kept.append(cand)
            continue
        stacked = np.vstack([cand, subj])
        if float(colour_distance_matrix(stacked)[0, 1:].min()) >= float(min_delta_e):
            kept.append(cand)
    if not kept:
        fallback = _contrasting_background_rgb(
            subject_mean, subject_palette, min_delta_e=min_delta_e
        )
        kept = [fallback]
    # Prefer a small multi-block set (light / mid / deep cool).
    if len(kept) > 4:
        kept = kept[:4]
    return np.asarray(kept, dtype=np.uint8)


def _contrasting_background_rgb(
    subject_mean: np.ndarray,
    subject_palette: np.ndarray,
    *,
    min_delta_e: float,
) -> np.ndarray:
    """Pick a flat background RGB at least ``min_delta_e`` from subject paints."""
    family = _contrasting_background_family(
        subject_mean, subject_palette, min_delta_e=min_delta_e
    )
    return family[0]


def _luminance(rgb: np.ndarray) -> float:
    rgb = np.asarray(rgb, dtype=np.float64).reshape(3)
    return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


def _nearest_family_swatch(rgb: np.ndarray, family: np.ndarray) -> np.ndarray:
    """Map an original bg paint onto the closest-luminance family swatch."""
    target_l = _luminance(rgb)
    best = family[0]
    best_d = abs(_luminance(best) - target_l)
    for cand in family[1:]:
        d = abs(_luminance(cand) - target_l)
        if d < best_d:
            best = cand
            best_d = d
    return best


def _is_warm_coat_like_rgb(rgb: np.ndarray) -> bool:
    """True for cream/tan/gold paints that must not survive as background."""
    r, g, b = (int(x) for x in np.asarray(rgb, dtype=np.int16).reshape(3))
    if r < 120:
        return False
    warm = r > g + 6 and r > b + 8
    cream = r >= 170 and g >= 140 and b >= 100 and r >= b + 8
    return bool(warm or cream)


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

    Warm cream/tan background fills are always offenders (even when Lab ΔE to
    every subject swatch is ≥ ``min_delta_e``) — those blocks look like
    leftover sheet/path geometry behind golden coats.

    Offending background pixels are remapped onto a small cool abstract family
    (not a single flat wash) so multi-block backgrounds can survive; subject
    pixels of a previously shared paint are left intact.
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
    bg_colours = {int(c) for c in np.unique(work[~mask])}
    # Global: never share a paint index, and never keep a bg paint within
    # min_delta_e of the subject (muddy sources like cream-on-linen / dog-on-
    # field otherwise keep merging even outside the silhouette band).
    offenders.update(subj_colours & bg_colours)
    for c in bg_colours:
        if c in subj_colours:
            continue
        min_de = min(float(dist[c, s]) for s in subj_colours)
        if min_de < float(min_delta_e) or _is_warm_coat_like_rgb(pal[c]):
            offenders.add(c)
    # Local reinforcement: anything left in the separation band that still
    # sits too close after palette edits is also an offender.
    if band.any():
        for colour in np.unique(work[band]):
            c = int(colour)
            if c in subj_colours:
                continue
            min_de = min(float(dist[c, s]) for s in subj_colours)
            if min_de < float(min_delta_e) or _is_warm_coat_like_rgb(pal[c]):
                offenders.add(c)

    if not offenders:
        return work, pal

    family = _contrasting_background_family(
        subj_mean, subj_pal, min_delta_e=float(min_delta_e)
    )
    # Reuse existing safe palette entries first, then append family swatches.
    safe_indices: list[int] = []
    for idx, rgb in enumerate(pal):
        if idx in subj_colours or idx in offenders:
            continue
        if _is_warm_coat_like_rgb(rgb):
            continue
        stacked = np.vstack([rgb, subj_pal])
        if float(colour_distance_matrix(stacked)[0, 1:].min()) >= float(min_delta_e):
            safe_indices.append(int(idx))
    family_indices: list[int] = list(safe_indices)
    for swatch in family:
        if len(family_indices) >= 4:
            break
        # Skip if an existing safe entry is already very close.
        if family_indices:
            existing = pal[family_indices]
            stacked = np.vstack([swatch, existing])
            if float(colour_distance_matrix(stacked)[0, 1:].min()) < 6.0:
                continue
        pal = np.vstack([pal, swatch.reshape(1, 3)]).astype(np.uint8)
        family_indices.append(int(pal.shape[0] - 1))
    if not family_indices:
        pal = np.vstack([pal, family[0].reshape(1, 3)]).astype(np.uint8)
        family_indices = [int(pal.shape[0] - 1)]

    family_rgbs = pal[family_indices]
    remapped = 0
    for c in sorted(offenders):
        hit = (~mask) & (work == c)
        if not hit.any():
            continue
        target_rgb = _nearest_family_swatch(pal[c], family_rgbs)
        # Resolve to palette index (exact match in family_indices).
        new_idx = family_indices[0]
        best = 1e9
        for idx in family_indices:
            d = float(np.sum((pal[idx].astype(np.int16) - target_rgb.astype(np.int16)) ** 2))
            if d < best:
                best = d
                new_idx = idx
        work[hit] = new_idx
        remapped += int(hit.sum())

    work, pal = compact_palette(work, pal)
    logger.info(
        "Subject/bg separation: remapped %d bg px onto %d cool swatches "
        "(shared paints + ≈%.1fmm band, ΔE≥%.1f, %dpx band)",
        remapped,
        len(family_indices),
        separation_mm,
        min_delta_e,
        band_px,
    )
    return work, pal
