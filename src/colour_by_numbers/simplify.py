"""Reduce palette-image complexity into coherent colouring-book regions.

Photographic quantization creates tens of thousands of speckles. This module
turns that noisy label map into a small set of large, simply shaped regions:

1. Majority-filter smoothing (removes salt-and-pepper noise)
2. Morphological opening/closing per colour (kills thin slivers, fills gaps)
3. Absorbing every connected component below a size threshold into the
   neighbouring colour that shares the longest border
4. Continuing absorption until a maximum region count is met
5. Dropping unused palette entries and compacting indices
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class SimplificationStats:
    """Diagnostics for how much complexity was removed."""

    regions_before: int
    regions_after: int
    min_region_area: int
    smooth_radius: int
    passes: int


def count_regions(labels: np.ndarray) -> int:
    """Count 8-connected components across all colour labels."""
    structure = np.ones((3, 3), dtype=bool)
    total = 0
    for colour in np.unique(labels):
        _, n = ndimage.label(labels == colour, structure=structure)
        total += int(n)
    return total


def majority_smooth(labels: np.ndarray, *, radius: int = 2, iterations: int = 1) -> np.ndarray:
    """Replace each pixel with the majority label in its neighbourhood."""
    if radius <= 0 or iterations <= 0:
        return labels.astype(np.int32, copy=True)

    size = radius * 2 + 1
    current = labels.astype(np.int32, copy=True)
    for _ in range(iterations):
        best_votes = np.zeros(current.shape, dtype=np.float32)
        best_labels = current.copy()
        for value in np.unique(current):
            mask = (current == value).astype(np.float32)
            votes = ndimage.uniform_filter(mask, size=size, mode="nearest")
            better = votes > best_votes
            best_labels = np.where(better, value, best_labels)
            best_votes = np.where(better, votes, best_votes)
        current = best_labels.astype(np.int32)
    return current


def smooth_boundaries(labels: np.ndarray, *, sigma: float = 1.5) -> np.ndarray:
    """Soften jagged borders by Gaussian-blurring each colour's membership."""
    if sigma <= 0:
        return labels.astype(np.int32, copy=True)
    colours = np.unique(labels)
    scores = np.stack(
        [
            ndimage.gaussian_filter(
                (labels == colour).astype(np.float32), sigma=sigma
            )
            for colour in colours
        ],
        axis=0,
    )
    return colours[scores.argmax(axis=0)].astype(np.int32)


def morphological_clean(labels: np.ndarray, *, radius: int = 2) -> np.ndarray:
    """Open then close each colour mask to remove slivers and seal small holes."""
    if radius <= 0:
        return labels.astype(np.int32, copy=True)

    structure = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool)
    colours = np.unique(labels)
    # Distance-to-each-colour via binary erosion/dilation on masks.
    # Start from empty and paint cleaned masks; unresolved pixels keep original.
    cleaned = np.full(labels.shape, -1, dtype=np.int32)
    for colour in colours:
        mask = labels == colour
        opened = ndimage.binary_opening(mask, structure=structure)
        closed = ndimage.binary_closing(opened, structure=structure)
        cleaned[closed] = int(colour)

    unresolved = cleaned < 0
    if unresolved.any():
        # Fill gaps from nearest surviving labelled pixel.
        _, indices = ndimage.distance_transform_edt(cleaned < 0, return_indices=True)
        iy, ix = indices
        cleaned[unresolved] = cleaned[iy[unresolved], ix[unresolved]]
        # If still unresolved (empty image edge case), fall back to original.
        still = cleaned < 0
        cleaned[still] = labels[still]
    return cleaned.astype(np.int32)


def _neighbour_colour_votes(labels: np.ndarray, component: np.ndarray) -> np.ndarray:
    """Histogram of colours touching the border of ``component``."""
    dilated = ndimage.binary_dilation(component, structure=np.ones((3, 3), dtype=bool))
    border = dilated & ~component
    if not border.any():
        return np.zeros(0, dtype=np.int64)
    neighbours = labels[border]
    return np.bincount(neighbours.astype(np.int64))


def absorb_thin_regions(
    labels: np.ndarray,
    *,
    min_thickness: float = 4.0,
    max_passes: int = 8,
) -> np.ndarray:
    """Absorb ribbon-like regions whose inscribed diameter is too thin to colour."""
    if min_thickness <= 0:
        return labels.astype(np.int32, copy=True)

    structure = np.ones((3, 3), dtype=bool)
    current = labels.astype(np.int32, copy=True)

    for _ in range(max_passes):
        work = current.copy()
        absorbed = 0
        for colour in np.unique(current):
            labeled, n = ndimage.label(current == colour, structure=structure)
            if n == 0:
                continue
            for comp_id in range(1, n + 1):
                component = labeled == comp_id
                # Inscribed diameter ≈ 2 * max distance-to-border.
                thickness = 2.0 * float(ndimage.distance_transform_edt(component).max())
                if thickness >= min_thickness:
                    continue
                votes = _neighbour_colour_votes(current, component)
                if votes.size == 0:
                    continue
                if colour < votes.size:
                    votes = votes.copy()
                    votes[colour] = 0
                if votes.max() == 0:
                    continue
                work[component] = int(votes.argmax())
                absorbed += 1
        current = work
        if absorbed == 0:
            break
    return current


def absorb_small_regions(
    labels: np.ndarray,
    *,
    min_area: int,
    max_passes: int = 32,
) -> np.ndarray:
    """Merge connected components smaller than ``min_area`` into neighbours.

    Each small region is reassigned to the adjacent colour with the longest
    shared border. Repeats until every remaining region meets the threshold
    (or ``max_passes`` is hit).
    """
    if min_area <= 1:
        return labels.astype(np.int32, copy=True)

    structure = np.ones((3, 3), dtype=bool)
    current = labels.astype(np.int32, copy=True)

    for _ in range(max_passes):
        work = current.copy()
        absorbed = 0
        regions: list[tuple[int, int, int]] = []

        for colour in np.unique(current):
            mask = current == colour
            labeled, n = ndimage.label(mask, structure=structure)
            if n == 0:
                continue
            areas = np.bincount(labeled.ravel())
            for comp_id in range(1, n + 1):
                area = int(areas[comp_id])
                if area < min_area:
                    regions.append((area, int(colour), comp_id))

        if not regions:
            break

        regions.sort(key=lambda item: item[0])
        labeled_by_colour: dict[int, np.ndarray] = {}

        for _area, colour, comp_id in regions:
            if colour not in labeled_by_colour:
                labeled_by_colour[colour] = ndimage.label(
                    current == colour, structure=structure
                )[0]
            labeled = labeled_by_colour[colour]
            component = labeled == comp_id
            if not component.any() or not np.all(current[component] == colour):
                continue

            votes = _neighbour_colour_votes(current, component)
            if votes.size == 0:
                continue
            if colour < votes.size:
                votes = votes.copy()
                votes[colour] = 0
            if votes.size == 0 or votes.max() == 0:
                counts = np.bincount(current.ravel())
                if colour < counts.size:
                    counts = counts.copy()
                    counts[colour] = 0
                if counts.max() == 0:
                    continue
                target = int(counts.argmax())
            else:
                target = int(votes.argmax())

            work[component] = target
            absorbed += 1

        current = work
        if absorbed == 0:
            break

    return current


def limit_region_count(
    labels: np.ndarray,
    *,
    max_regions: int,
    max_passes: int = 32,
) -> np.ndarray:
    """Keep absorbing the smallest regions until ``max_regions`` is reached."""
    if max_regions <= 0:
        return labels.astype(np.int32, copy=True)

    current = labels.astype(np.int32, copy=True)
    structure = np.ones((3, 3), dtype=bool)

    for _ in range(max_passes):
        inventory: list[tuple[int, int, int, np.ndarray]] = []
        for colour in np.unique(current):
            labeled, n = ndimage.label(current == colour, structure=structure)
            if n == 0:
                continue
            areas = np.bincount(labeled.ravel())
            for comp_id in range(1, n + 1):
                inventory.append((int(areas[comp_id]), int(colour), comp_id, labeled))

        if len(inventory) <= max_regions:
            break

        inventory.sort(key=lambda item: item[0])
        excess = len(inventory) - max_regions
        work = current.copy()
        absorbed = 0
        for _area, colour, comp_id, labeled in inventory[: max(excess, 1)]:
            component = labeled == comp_id
            if not component.any() or not np.all(work[component] == colour):
                continue
            votes = _neighbour_colour_votes(work, component)
            if votes.size == 0:
                continue
            if colour < votes.size:
                votes = votes.copy()
                votes[colour] = 0
            if votes.max() == 0:
                continue
            work[component] = int(votes.argmax())
            absorbed += 1
        current = work
        if absorbed == 0:
            break

    return current


def compact_palette(
    labels: np.ndarray,
    palette: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop unused colours and renumber labels to a compact 0..K-1 range.

    Remaining colours stay ordered dark → light.
    """
    used = np.unique(labels)
    subset = palette[used]
    order = np.argsort(subset.mean(axis=1))
    ordered_used = used[order]
    new_palette = palette[ordered_used]

    remap = np.full(int(labels.max()) + 1, -1, dtype=np.int32)
    for new_idx, old_idx in enumerate(ordered_used):
        remap[int(old_idx)] = new_idx
    new_labels = remap[labels]
    return new_labels.astype(np.int32), new_palette.astype(np.uint8)


def merge_low_contrast_neighbours(
    labels: np.ndarray,
    palette: np.ndarray,
    *,
    min_delta_e: float = 18.0,
    max_passes: int = 12,
    max_merge_fraction: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge touching near-duplicate paints when one side is a small region.

    Large adjacent sections are left alone so major subject/background blocks
    keep separate crayon colours; only muddy/small transitions collapse.
    """
    from .palette import colour_distance_matrix

    if min_delta_e <= 0 or palette.shape[0] < 2:
        return labels.astype(np.int32, copy=True), palette

    work = labels.astype(np.int32, copy=True)
    pal = palette.astype(np.uint8, copy=True)
    area_cap = max(64, int(work.size * max_merge_fraction))

    for _ in range(max_passes):
        if pal.shape[0] < 2:
            break
        dist = colour_distance_matrix(pal)
        pairs: list[tuple[int, int]] = []
        for axis in (0, 1):
            if axis == 0:
                a, b = work[:, :-1], work[:, 1:]
            else:
                a, b = work[:-1, :], work[1:, :]
            differ = a != b
            if not differ.any():
                continue
            left = a[differ].ravel()
            right = b[differ].ravel()
            stacked = np.stack(
                [np.minimum(left, right), np.maximum(left, right)], axis=1
            )
            for ca, cb in np.unique(stacked, axis=0):
                pairs.append((int(ca), int(cb)))

        unique_pairs = sorted(set(pairs), key=lambda p: dist[p[0], p[1]])
        merge_pair = None
        for ca, cb in unique_pairs:
            if dist[ca, cb] >= min_delta_e:
                continue
            count_a = int((work == ca).sum())
            count_b = int((work == cb).sum())
            # Only collapse when the smaller paint is a minor muddy patch.
            if min(count_a, count_b) > area_cap:
                continue
            merge_pair = (ca, cb, count_a, count_b)
            break
        if merge_pair is None:
            break

        ca, cb, count_a, count_b = merge_pair
        keep, drop = (ca, cb) if count_a >= count_b else (cb, ca)
        work[work == drop] = keep
        work, pal = compact_palette(work, pal)

    return work, pal


def simplify_labels(
    labels: np.ndarray,
    palette: np.ndarray,
    *,
    min_region_area: int | None = None,
    max_regions: int | None = None,
    smooth_radius: int = 3,
    smooth_iterations: int = 2,
    morph_radius: int = 2,
    boundary_sigma: float = 1.25,
    min_thickness: float | None = None,
    min_adjacent_delta_e: float = 18.0,
    compact: bool = True,
) -> tuple[np.ndarray, np.ndarray, SimplificationStats]:
    """Full simplification pipeline for colour-by-numbers pages.

    Returns ``(labels, palette, stats)``. When ``compact`` is False, palette
    indices are preserved so dual-path results can be merged.
    """
    height, width = labels.shape
    if min_region_area is None:
        min_region_area = max(80, int(width * height * 0.020))
    if max_regions is None:
        max_regions = max(12, min(40, int(palette.shape[0] * 2)))
    if min_thickness is None:
        min_thickness = max(3.0, min(width, height) * 0.015)

    before = count_regions(labels)
    simplified = majority_smooth(
        labels, radius=smooth_radius, iterations=smooth_iterations
    )
    simplified = morphological_clean(simplified, radius=morph_radius)
    simplified = absorb_small_regions(simplified, min_area=min_region_area)
    simplified = absorb_thin_regions(simplified, min_thickness=min_thickness)
    simplified = limit_region_count(simplified, max_regions=max_regions)
    simplified = smooth_boundaries(simplified, sigma=boundary_sigma)
    simplified = absorb_small_regions(simplified, min_area=min_region_area)
    simplified = absorb_thin_regions(simplified, min_thickness=min_thickness)
    simplified = limit_region_count(simplified, max_regions=max_regions)
    if compact:
        simplified, new_palette = compact_palette(simplified, palette)
        if min_adjacent_delta_e and min_adjacent_delta_e > 0:
            simplified, new_palette = merge_low_contrast_neighbours(
                simplified, new_palette, min_delta_e=min_adjacent_delta_e
            )
    else:
        new_palette = palette
    after = count_regions(simplified)

    stats = SimplificationStats(
        regions_before=before,
        regions_after=after,
        min_region_area=min_region_area,
        smooth_radius=smooth_radius,
        passes=smooth_iterations,
    )
    return simplified, new_palette, stats


def simplify_dual(
    labels: np.ndarray,
    palette: np.ndarray,
    subject_mask: np.ndarray,
    *,
    subject_params: dict,
    background_params: dict,
    firm_border: bool = True,
    min_adjacent_delta_e: float = 18.0,
) -> tuple[np.ndarray, np.ndarray, SimplificationStats, SimplificationStats]:
    """Simplify subject pixels with one preset and background with another.

    ``subject_mask`` is a boolean HxW array aligned with ``labels``.
    When ``firm_border`` is True, the hard mask silhouette is preserved with
    no seam softening across the subject edge.
    """
    if subject_mask.shape != labels.shape:
        raise ValueError("subject_mask must match labels shape")

    subject_params = dict(subject_params)
    background_params = dict(background_params)
    # Defer adjacent merge until after dual combine.
    subject_params["min_adjacent_delta_e"] = 0.0
    background_params["min_adjacent_delta_e"] = 0.0
    subject_params["compact"] = False
    background_params["compact"] = False

    subject_labels, _, subject_stats = simplify_labels(
        labels, palette, **subject_params
    )
    background_labels, _, background_stats = simplify_labels(
        labels, palette, **background_params
    )
    if firm_border:
        combined = np.where(subject_mask, subject_labels, background_labels).astype(
            np.int32
        )
    else:
        combined = np.where(subject_mask, subject_labels, background_labels).astype(
            np.int32
        )
        seam = ndimage.binary_dilation(subject_mask, iterations=2) ^ subject_mask
        if seam.any():
            softened = smooth_boundaries(combined, sigma=0.8)
            combined = np.where(seam, softened, combined).astype(np.int32)
    combined, new_palette = compact_palette(combined, palette)
    if min_adjacent_delta_e and min_adjacent_delta_e > 0:
        combined, new_palette = merge_low_contrast_neighbours(
            combined, new_palette, min_delta_e=min_adjacent_delta_e
        )
    return combined, new_palette, subject_stats, background_stats
