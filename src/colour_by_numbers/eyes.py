"""Preserve eye features on portrait and full-body subjects (animals, people)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .palette import ANIMAL_CATEGORIES, rgb_to_lab
from .print_resolution import DEFAULT_EYE_REGION_MM, min_region_size_for_a4_mm
from .simplify import is_colourable_block

PORTRAIT_EYE_CATEGORIES = frozenset(ANIMAL_CATEGORIES) | frozenset(
    {"people", "portraits"}
)


def portrait_subject(category: str | None) -> bool:
    return (category or "") in PORTRAIT_EYE_CATEGORIES


def face_region_mask(
    shape: tuple[int, int],
    subject_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Head-biased band where eyes usually sit.

    Portrait crops use the classic upper-centre band. Full-body / side poses
    often put the head on one side of the subject bbox, so we also include the
    forward (left/right) thirds and pick the zone with the strongest local
    dark-on-light contrast when a subject mask is available.
    """
    h, w = shape
    if subject_mask is not None and subject_mask.shape == shape and subject_mask.any():
        ys, xs = np.where(subject_mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
    else:
        y0, y1 = 0, h - 1
        x0, x1 = 0, w - 1
    bh = max(1, y1 - y0 + 1)
    bw = max(1, x1 - x0 + 1)

    # Always include the upper band (classic portraits).
    mask = np.zeros((h, w), dtype=bool)
    fy1 = y0 + max(1, int(bh * 0.48))
    fx0 = x0 + int(bw * 0.08)
    fx1 = x1 - int(bw * 0.08)
    mask[y0 : fy1 + 1, max(0, fx0) : min(w, fx1 + 1)] = True

    # Full-body / elongated subjects: also allow left and right head priors.
    if bw >= int(bh * 1.15) or bh >= int(h * 0.55):
        mid_y0 = y0 + int(bh * 0.08)
        mid_y1 = y0 + max(mid_y0 + 1, int(bh * 0.72))
        left_x1 = x0 + max(1, int(bw * 0.42))
        right_x0 = x1 - max(1, int(bw * 0.42))
        mask[mid_y0 : mid_y1 + 1, x0 : min(w, left_x1 + 1)] = True
        mask[mid_y0 : mid_y1 + 1, max(0, right_x0) : x1 + 1] = True

    if subject_mask is not None and subject_mask.shape == shape:
        mask &= subject_mask
    return mask


def _iter_components(labels: np.ndarray, colour: int, structure: np.ndarray):
    labeled, n = ndimage.label(labels == colour, structure=structure)
    if n == 0:
        return
    for comp_id in range(1, n + 1):
        yield labeled == comp_id


def _component_geometry(component: np.ndarray) -> tuple[float, float, int, int, float]:
    """Return ``(cx, cy, bw, bh, fill)`` for a boolean component."""
    ys, xs = np.where(component)
    if xs.size == 0:
        return 0.0, 0.0, 0, 0, 0.0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    area = int(xs.size)
    fill = float(area) / float(max(1, bw * bh))
    return float(xs.mean()), float(ys.mean()), bw, bh, fill


def _local_darkness_score(
    component: np.ndarray,
    luma_map: np.ndarray,
    *,
    ring: int = 6,
) -> float:
    """How much darker the blob is than a ring of surrounding pixels."""
    if not component.any():
        return 0.0
    dilated = ndimage.binary_dilation(component, iterations=max(1, ring))
    ring_mask = dilated & ~component
    if not ring_mask.any():
        return 0.0
    inside = float(luma_map[component].mean())
    outside = float(luma_map[ring_mask].mean())
    return outside - inside


def compute_eye_protection_mask(
    labels: np.ndarray,
    palette: np.ndarray,
    *,
    category: str | None,
    subject_mask: np.ndarray | None = None,
    min_region_mm: float = 8.0,
    eye_region_mm: float = DEFAULT_EYE_REGION_MM,
) -> np.ndarray:
    """Pixels to exempt from aggressive absorption (pupils + nearby sclera).

    Prefers compact, locally-dark blobs in a head-biased ROI so full-body and
    side views keep real eyes instead of large ear/fur shadows.
    """
    protected = np.zeros(labels.shape, dtype=bool)
    if not portrait_subject(category):
        return protected

    h, w = labels.shape
    face = face_region_mask((h, w), subject_mask)
    if not face.any():
        return protected

    relaxed = min_region_size_for_a4_mm(w, h, min_mm=eye_region_mm)
    full = min_region_size_for_a4_mm(w, h, min_mm=min_region_mm)
    # Tiny pupils on wide shots are often only a few dozen pixels after quantize.
    min_area = max(4, min(12, relaxed.min_area_px // 8))
    max_area = max(full.min_area_px * 2, int(h * w * 0.012), min_area + 1)

    lab = rgb_to_lab(palette)
    luma = lab[:, 0]
    structure = np.ones((3, 3), dtype=bool)
    luma_map = luma[np.clip(labels, 0, len(luma) - 1)]

    dark_candidates: list[tuple[float, np.ndarray, tuple[float, float]]] = []
    light_components: list[tuple[np.ndarray, tuple[float, float]]] = []

    for colour in np.unique(labels):
        idx = int(colour)
        lightness = float(luma[idx]) if idx < len(luma) else 128.0
        for component in _iter_components(labels, idx, structure):
            if not np.any(component & face):
                continue
            # Keep only the part inside the face prior so body shadows don't win.
            component = component & face
            area = int(component.sum())
            if area < min_area or area > max_area:
                continue
            cx, cy, bw, bh, fill = _component_geometry(component)
            if bw < 2 or bh < 2:
                continue
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect > 2.8 or fill < 0.28:
                continue
            if lightness <= 42.0:
                contrast = _local_darkness_score(component, luma_map)
                if contrast < 6.0 and area > relaxed.min_area_px:
                    # Large weakly-contrasted dark fur — skip.
                    continue
                # Prefer compact, high-contrast, mid-sized eye blobs.
                compactness = fill / aspect
                score = contrast * compactness * (1.0 + min(area, 80) / 40.0)
                # Mild prior: eyes sit in the upper/mid part of the search band.
                face_ys = np.where(face)[0]
                face_y0 = int(face_ys.min()) if face_ys.size else 0
                score *= 1.15 - 0.35 * ((cy - face_y0) / max(1.0, float(h)))
                dark_candidates.append((score, component, (cx, cy)))
            elif lightness >= 76.0:
                light_components.append((component, (cx, cy)))

    if not dark_candidates:
        # Fall back: absolute-dark components in the face band (legacy path).
        for colour in np.unique(labels):
            idx = int(colour)
            lightness = float(luma[idx]) if idx < len(luma) else 128.0
            if lightness > 38.0:
                continue
            for component in _iter_components(labels, idx, structure):
                part = component & face
                area = int(part.sum())
                if area < min_area or area > max_area:
                    continue
                cx, cy, bw, bh, fill = _component_geometry(part)
                aspect = max(bw, bh) / max(1, min(bw, bh))
                if aspect > 3.0 or fill < 0.2:
                    continue
                dark_candidates.append((float(area), part, (cx, cy)))

    if not dark_candidates:
        return protected

    dark_candidates.sort(key=lambda item: item[0], reverse=True)
    # Keep a shortlist of the strongest blobs, then pick a horizontal pair or one.
    shortlist = dark_candidates[:8]
    pupils: list[np.ndarray] = []

    if len(shortlist) >= 2:
        best_pair: tuple[float, np.ndarray, np.ndarray] | None = None
        for i, (score_a, comp_a, (ax, ay)) in enumerate(shortlist):
            for score_b, comp_b, (bx, by) in shortlist[i + 1 :]:
                dx = abs(ax - bx)
                dy = abs(ay - by)
                if dx < max(6.0, w * 0.03) or dx > w * 0.55:
                    continue
                if dy > max(18.0, h * 0.12):
                    continue
                pair_score = score_a + score_b + dx * 0.02
                if best_pair is None or pair_score > best_pair[0]:
                    best_pair = (pair_score, comp_a, comp_b)
        if best_pair is not None:
            pupils = [best_pair[1], best_pair[2]]

    if not pupils:
        # Profile / looking-away: protect the single strongest eye-like blob.
        pupils = [shortlist[0][1]]

    for pupil in pupils:
        protected |= pupil
        # Small halo so neighbouring iris/lid wedges stay with the eye.
        protected |= ndimage.binary_dilation(pupil, iterations=1)

    span = max(w, h) * 0.18
    for light_comp, (lx, ly) in light_components:
        for pupil in pupils:
            py, px = ndimage.center_of_mass(pupil)
            if ((lx - px) ** 2 + (ly - py) ** 2) ** 0.5 <= span:
                protected |= light_comp
                break

    return protected


def relaxed_eye_thresholds(
    width: int,
    height: int,
    *,
    eye_region_mm: float = DEFAULT_EYE_REGION_MM,
) -> tuple[int, int, float]:
    region = min_region_size_for_a4_mm(width, height, min_mm=eye_region_mm)
    return (
        region.min_width_px,
        region.min_height_px,
        float(region.min_inscribed_diameter_px),
    )


def emphasize_protected_pupils(
    labels: np.ndarray,
    palette: np.ndarray,
    protected: np.ndarray,
    *,
    target_rgb: tuple[int, int, int] = (22, 18, 14),
    max_lightness: float = 55.0,
) -> tuple[np.ndarray, np.ndarray]:
    """No-op: forced pupil recolouring distorted plates and misplaced “eyes”.

    Kept as a stable API stub so call sites can stay unchanged. Subject
    integrity is handled by dual preserve/background simplify instead.
    """
    del protected, target_rgb, max_lightness
    return labels.astype(np.int32, copy=True), palette.astype(np.uint8, copy=True)


def component_protected(
    component: np.ndarray,
    protected: np.ndarray,
    *,
    min_width_px: int,
    min_height_px: int,
    min_inscribed_px: float,
    relaxed_width_px: int,
    relaxed_height_px: int,
    relaxed_inscribed_px: float,
) -> bool:
    """True when a component should stay as a colourable eye fill.

    Smaller protected pupils are left for the caller to harvest as black detail
    ink instead of numbered speckles.
    """
    overlap = protected & component
    if not overlap.any():
        return False
    if is_colourable_block(
        component,
        min_width_px=min_width_px,
        min_height_px=min_height_px,
        min_inscribed_px=min_inscribed_px,
    ):
        return True
    return is_colourable_block(
        component,
        min_width_px=relaxed_width_px,
        min_height_px=relaxed_height_px,
        min_inscribed_px=relaxed_inscribed_px,
    )
