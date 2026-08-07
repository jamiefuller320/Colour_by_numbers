"""Preserve paired eye features on portrait subjects (animals, people)."""

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
    """Upper-centre band of the subject (or frame) where eyes usually sit."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if subject_mask is not None and subject_mask.shape == shape and subject_mask.any():
        ys, xs = np.where(subject_mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
    else:
        y0, y1 = 0, h - 1
        x0, x1 = 0, w - 1
    bh = max(1, y1 - y0 + 1)
    bw = max(1, x1 - x0 + 1)
    fy1 = y0 + max(1, int(bh * 0.48))
    fx0 = x0 + int(bw * 0.10)
    fx1 = x1 - int(bw * 0.10)
    mask[y0 : fy1 + 1, max(0, fx0) : min(w, fx1 + 1)] = True
    return mask


def _iter_components(labels: np.ndarray, colour: int, structure: np.ndarray):
    labeled, n = ndimage.label(labels == colour, structure=structure)
    if n == 0:
        return
    for comp_id in range(1, n + 1):
        yield labeled == comp_id


def compute_eye_protection_mask(
    labels: np.ndarray,
    palette: np.ndarray,
    *,
    category: str | None,
    subject_mask: np.ndarray | None = None,
    min_region_mm: float = 8.0,
    eye_region_mm: float = DEFAULT_EYE_REGION_MM,
) -> np.ndarray:
    """Pixels to exempt from aggressive absorption (paired pupils + sclera)."""
    protected = np.zeros(labels.shape, dtype=bool)
    if not portrait_subject(category):
        return protected

    h, w = labels.shape
    face = face_region_mask((h, w), subject_mask)
    if not face.any():
        return protected

    relaxed = min_region_size_for_a4_mm(w, h, min_mm=eye_region_mm)
    full = min_region_size_for_a4_mm(w, h, min_mm=min_region_mm)
    min_area = max(4, relaxed.min_area_px // 2)
    max_area = max(full.min_area_px * 4, int(h * w * 0.03), min_area + 1)

    lab = rgb_to_lab(palette)
    luma = lab[:, 0]
    structure = np.ones((3, 3), dtype=bool)
    dark_components: list[tuple[np.ndarray, tuple[float, float]]] = []
    light_components: list[tuple[np.ndarray, tuple[float, float]]] = []

    for colour in np.unique(labels):
        idx = int(colour)
        lightness = float(luma[idx]) if idx < len(luma) else 128.0
        for component in _iter_components(labels, idx, structure):
            if not np.any(component & face):
                continue
            area = int(component.sum())
            if area < min_area or area > max_area:
                continue
            cy, cx = ndimage.center_of_mass(component)
            if lightness <= 38.0:
                dark_components.append((component, (float(cx), float(cy))))
            elif lightness >= 76.0:
                light_components.append((component, (float(cx), float(cy))))

    if len(dark_components) >= 2:
        dark_sorted = sorted(dark_components, key=lambda item: item[1][0])
        pupils = [dark_sorted[0][0], dark_sorted[-1][0]]
        for pupil in pupils:
            protected |= pupil
        span = max(w, h) * 0.20
        for light_comp, (lx, ly) in light_components:
            for pupil in pupils:
                py, px = ndimage.center_of_mass(pupil)
                if ((lx - px) ** 2 + (ly - py) ** 2) ** 0.5 <= span:
                    protected |= light_comp
                    break
        return protected

    if len(dark_components) == 1 and light_components:
        protected |= dark_components[0][0]
        for light_comp, _ in light_components:
            protected |= light_comp
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
    """True when a component should be kept for eye definition."""
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
    if is_colourable_block(
        component,
        min_width_px=relaxed_width_px,
        min_height_px=relaxed_height_px,
        min_inscribed_px=relaxed_inscribed_px,
    ):
        return True
    overlap_frac = float(overlap.sum()) / max(1, int(component.sum()))
    return overlap_frac >= 0.45
