"""Subject-aware preprocessing using neural background removal (rembg / U²-Net).

Supports isolating the foreground, cropping so the subject fills a target
fraction of the frame (default 80%), and preparing images for dual-complexity
colour-by-numbers (fine on subject, light on background).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_BG = (248, 248, 248)
DEFAULT_SUBJECT_FILL = 0.80


@dataclass(frozen=True)
class SubjectMask:
    """Foreground alpha mask aligned to an RGB image."""

    alpha: np.ndarray  # HxW uint8
    model: str
    foreground_fraction: float

    @property
    def binary(self) -> np.ndarray:
        return self.alpha >= 128


def rembg_available() -> bool:
    try:
        import rembg  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=4)
def _session(model_name: str):
    from rembg import new_session

    return new_session(model_name)


def estimate_subject_mask(
    image: Image.Image,
    *,
    model_name: str = "u2net",
) -> SubjectMask:
    """Estimate a soft foreground alpha matte for ``image``."""
    if not rembg_available():
        raise RuntimeError(
            "Subject isolation requires rembg. Install with: pip install 'rembg[cpu]'"
        )

    from rembg import remove

    rgb = image.convert("RGB")
    session = _session(model_name)
    cutout = remove(rgb, session=session)
    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")
    alpha = np.asarray(cutout.split()[-1], dtype=np.uint8)
    fraction = float((alpha >= 128).mean())
    logger.info(
        "Subject mask via %s: %.1f%% foreground", model_name, 100.0 * fraction
    )
    return SubjectMask(alpha=alpha, model=model_name, foreground_fraction=fraction)


def silhouette_binary_from_mask(
    mask: SubjectMask,
    *,
    threshold: int = 64,
    close_iterations: int = 5,
    keep_largest: bool = True,
) -> np.ndarray:
    """Firm boolean silhouette from a soft rembg matte.

    Flat colouring plates often need a lower alpha threshold than the default
    harden cut (128) so same-colour subject/background seams still count as
    subject for outline ink.
    """
    from scipy import ndimage

    binary = mask.alpha >= int(threshold)
    if close_iterations > 0:
        binary = ndimage.binary_closing(binary, iterations=int(close_iterations))
        binary = ndimage.binary_fill_holes(binary)
    if keep_largest and binary.any():
        labelled, count = ndimage.label(binary)
        if count > 1:
            sizes = np.bincount(labelled.ravel())
            sizes[0] = 0
            binary = labelled == int(sizes.argmax())
    return binary.astype(bool)


def estimate_silhouette_mask(
    image: Image.Image,
    *,
    model_name: str = "u2net",
    threshold: int = 64,
    min_foreground: float = 0.04,
    max_foreground: float = 0.85,
) -> np.ndarray | None:
    """Return a cleaned subject silhouette, or ``None`` if rembg is unusable."""
    if not rembg_available():
        logger.warning("Silhouette outline skipped: rembg is not installed")
        return None
    try:
        soft = estimate_subject_mask(image, model_name=model_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Silhouette outline skipped: rembg failed (%s)", exc)
        return None
    binary = silhouette_binary_from_mask(soft, threshold=threshold)
    fraction = float(binary.mean())
    if fraction < min_foreground or fraction > max_foreground:
        logger.warning(
            "Silhouette outline skipped: foreground %.1f%% outside %.0f–%.0f%%",
            100.0 * fraction,
            100.0 * min_foreground,
            100.0 * max_foreground,
        )
        return None
    logger.info("Silhouette mask ready: %.1f%% foreground", 100.0 * fraction)
    return binary


def align_mask(
    mask: SubjectMask,
    size: tuple[int, int],
    *,
    resample: Image.Resampling | None = None,
    firm: bool = False,
) -> SubjectMask:
    """Resize a subject mask to an image size ``(width, height)``.

    Use nearest-neighbour (``firm=True``) for binary silhouette masks so
    subject borders stay crisp when scaled.
    """
    width, height = size
    if mask.alpha.shape == (height, width):
        return harden_mask(mask) if firm else mask
    if resample is None:
        resample = Image.Resampling.NEAREST if firm else Image.Resampling.BILINEAR
    alpha_img = Image.fromarray(mask.alpha, mode="L").resize((width, height), resample)
    alpha = np.asarray(alpha_img, dtype=np.uint8)
    aligned = SubjectMask(
        alpha=alpha,
        model=mask.model,
        foreground_fraction=float((alpha >= 128).mean()),
    )
    return harden_mask(aligned) if firm else aligned


def isolate_on_flat_background(
    image: Image.Image,
    mask: SubjectMask | None = None,
    *,
    background: tuple[int, int, int] = DEFAULT_BG,
    model_name: str = "u2net",
) -> tuple[Image.Image, SubjectMask]:
    """Composite the subject onto a flat background colour."""
    rgb = image.convert("RGB")
    if mask is None:
        mask = estimate_subject_mask(rgb, model_name=model_name)
    mask = align_mask(mask, rgb.size)

    base = Image.new("RGB", rgb.size, background)
    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask.alpha, mode="L"))
    base.paste(rgba, mask=rgba.split()[-1])
    return base, mask


def crop_to_subject(
    image: Image.Image,
    mask: SubjectMask,
    *,
    padding_fraction: float = 0.12,
    min_foreground_fraction: float = 0.002,
) -> tuple[Image.Image, SubjectMask]:
    """Crop around the foreground bbox with relative padding."""
    binary = mask.binary
    if float(binary.mean()) < min_foreground_fraction:
        logger.warning("Subject mask too small to crop; leaving full frame.")
        return image, mask

    ys, xs = np.nonzero(binary)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    height, width = binary.shape
    pad_y = int((y1 - y0) * padding_fraction) + 4
    pad_x = int((x1 - x0) * padding_fraction) + 4
    y0 = max(0, y0 - pad_y)
    x0 = max(0, x0 - pad_x)
    y1 = min(height, y1 + pad_y)
    x1 = min(width, x1 + pad_x)

    cropped = image.crop((x0, y0, x1, y1))
    cropped_alpha = mask.alpha[y0:y1, x0:x1]
    cropped_mask = SubjectMask(
        alpha=cropped_alpha,
        model=mask.model,
        foreground_fraction=float((cropped_alpha >= 128).mean()),
    )
    return cropped, cropped_mask


def crop_to_subject_fill(
    image: Image.Image,
    mask: SubjectMask,
    *,
    target_fill: float = DEFAULT_SUBJECT_FILL,
    min_foreground_fraction: float = 0.002,
    pad_colour: tuple[int, int, int] = DEFAULT_BG,
) -> tuple[Image.Image, SubjectMask]:
    """Crop/pad so the subject bounding box fills ``target_fill`` of the frame.

    The crop keeps the source aspect ratio and centres on the subject. If the
    required window extends past the image edge, the missing area is padded.
    """
    if target_fill <= 0 or target_fill > 1:
        raise ValueError("target_fill must be in (0, 1].")

    rgb = image.convert("RGB")
    mask = align_mask(mask, rgb.size)
    binary = mask.binary
    if float(binary.mean()) < min_foreground_fraction:
        logger.warning("Subject mask too small for 80%% fill crop; leaving full frame.")
        return rgb, mask

    ys, xs = np.nonzero(binary)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bw = x1 - x0
    bh = y1 - y0
    width, height = rgb.size
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    current_fill = max(bw / width, bh / height)
    aspect = width / max(height, 1)

    # Minimum crop window so the subject bbox is target_fill of the crop.
    min_w = bw / target_fill
    min_h = bh / target_fill
    crop_h = max(min_h, min_w / aspect)
    crop_w = crop_h * aspect
    if crop_w < min_w:
        crop_w = min_w
        crop_h = crop_w / aspect

    # If the subject already exceeds the target fill, keep the full frame.
    if current_fill >= target_fill:
        logger.info(
            "Subject already fills %.0f%% of frame (target %.0f%%); no crop.",
            100 * current_fill,
            100 * target_fill,
        )
        return rgb, mask

    left = cx - crop_w / 2
    top = cy - crop_h / 2
    right = left + crop_w
    bottom = top + crop_h

    # Integer canvas covering the crop window; pad outside the source.
    pad_left = max(0, int(np.floor(-left)))
    pad_top = max(0, int(np.floor(-top)))
    pad_right = max(0, int(np.ceil(right - width)))
    pad_bottom = max(0, int(np.ceil(bottom - height)))

    if pad_left or pad_top or pad_right or pad_bottom:
        canvas_w = width + pad_left + pad_right
        canvas_h = height + pad_top + pad_bottom
        canvas = Image.new("RGB", (canvas_w, canvas_h), pad_colour)
        canvas.paste(rgb, (pad_left, pad_top))
        alpha_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        alpha_canvas[
            pad_top : pad_top + height, pad_left : pad_left + width
        ] = mask.alpha
        left += pad_left
        top += pad_top
        right += pad_left
        bottom += pad_top
        rgb = canvas
        mask = SubjectMask(
            alpha=alpha_canvas,
            model=mask.model,
            foreground_fraction=float((alpha_canvas >= 128).mean()),
        )

    x0c = int(np.floor(left))
    y0c = int(np.floor(top))
    x1c = int(np.ceil(right))
    y1c = int(np.ceil(bottom))
    x0c = max(0, x0c)
    y0c = max(0, y0c)
    x1c = min(rgb.width, x1c)
    y1c = min(rgb.height, y1c)

    cropped = rgb.crop((x0c, y0c, x1c, y1c))
    cropped_alpha = mask.alpha[y0c:y1c, x0c:x1c]
    cropped_mask = SubjectMask(
        alpha=cropped_alpha,
        model=mask.model,
        foreground_fraction=float((cropped_alpha >= 128).mean()),
    )
    fill = max(
        (x1 - x0) / max(cropped.width, 1),
        (y1 - y0) / max(cropped.height, 1),
    )
    # Recompute fill against the cropped subject bbox.
    ys2, xs2 = np.nonzero(cropped_mask.binary)
    if len(xs2):
        fill = max(
            (xs2.max() - xs2.min() + 1) / max(cropped.width, 1),
            (ys2.max() - ys2.min() + 1) / max(cropped.height, 1),
        )
    logger.info("Subject fill after crop: %.0f%% (target %.0f%%)", 100 * fill, 100 * target_fill)
    return cropped, cropped_mask


def harden_mask(mask: SubjectMask, *, threshold: int = 128) -> SubjectMask:
    """Convert a soft alpha matte into a firm binary subject mask."""
    hard = np.where(mask.alpha >= threshold, 255, 0).astype(np.uint8)
    return SubjectMask(
        alpha=hard,
        model=mask.model,
        foreground_fraction=float((hard >= 128).mean()),
    )


def blend_subject_background(
    subject_image: Image.Image,
    background_image: Image.Image,
    mask: SubjectMask,
    *,
    firm_border: bool = True,
) -> Image.Image:
    """Composite subject over background.

    When ``firm_border`` is True, uses the hard binary mask from the original
    subject crop (no soft alpha), so the silhouette edge stays crisp.
    """
    subject_image = subject_image.convert("RGB")
    background_image = background_image.convert("RGB").resize(
        subject_image.size, Image.Resampling.BILINEAR
    )
    mask = align_mask(mask, subject_image.size)
    if firm_border:
        mask = harden_mask(mask)
        out = np.asarray(background_image, dtype=np.uint8).copy()
        subj = np.asarray(subject_image, dtype=np.uint8)
        out[mask.binary] = subj[mask.binary]
        return Image.fromarray(out, mode="RGB")

    out = background_image.copy()
    rgba = subject_image.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask.alpha, mode="L"))
    out.paste(rgba, mask=rgba.split()[-1])
    return out


def apply_firm_subject_border(
    subject_labels: np.ndarray,
    background_labels: np.ndarray,
    hard_mask: np.ndarray,
) -> np.ndarray:
    """Snap dual label maps to the hard subject silhouette (no blurred seam)."""
    if hard_mask.shape != subject_labels.shape:
        raise ValueError("hard_mask must match subject_labels shape")
    if background_labels.shape != subject_labels.shape:
        raise ValueError("background_labels must match subject_labels shape")
    return np.where(hard_mask, subject_labels, background_labels).astype(np.int32)


def subject_bbox(mask: SubjectMask) -> tuple[int, int, int, int] | None:
    """Return ``(x0, y0, x1, y1)`` for the foreground, or None if empty."""
    binary = mask.binary
    if not binary.any():
        return None
    ys, xs = np.nonzero(binary)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _centroid(binary: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(binary)
    if len(ys) == 0:
        return 0.0, 0.0
    return float(ys.mean()), float(xs.mean())


def _shift_binary(binary: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = binary.shape
    out = np.zeros_like(binary)
    y0_src = max(0, -dy)
    y1_src = min(h, h - dy)
    x0_src = max(0, -dx)
    x1_src = min(w, w - dx)
    y0_dst = max(0, dy)
    x0_dst = max(0, dx)
    if y1_src <= y0_src or x1_src <= x0_src:
        return out
    out[y0_dst : y0_dst + (y1_src - y0_src), x0_dst : x0_dst + (x1_src - x0_src)] = (
        binary[y0_src:y1_src, x0_src:x1_src]
    )
    return out


def mask_from_subject_layer(
    scene: Image.Image,
    subject_only: Image.Image,
    *,
    model_name: str = "u2net",
    firm: bool = True,
    absdiff_boost: bool = True,
) -> SubjectMask:
    """Build a subject mask from a flat-ground companion illustration.

    Primary signal: rembg on ``subject_only`` (easy on a studio ground).
    The mask is resized to the scene and centroid-aligned to a quick rembg
    pass on the scene so small fal pose drift does not leave the cutout
    floating. Optional abs-diff against the scene reinforces edges when the
    two plates stay roughly registered.
    """
    from scipy import ndimage

    from .quantize import resize_for_processing

    scene_rgb = scene.convert("RGB")
    only_rgb = subject_only.convert("RGB")
    if only_rgb.size != scene_rgb.size:
        only_rgb = only_rgb.resize(scene_rgb.size, Image.Resampling.BILINEAR)

    only_seg = resize_for_processing(only_rgb, max_size=1024)
    only_mask = estimate_subject_mask(only_seg, model_name=model_name)
    only_mask = align_mask(only_mask, scene_rgb.size, firm=True)
    only_bin = silhouette_binary_from_mask(only_mask, threshold=64)

    scene_seg = resize_for_processing(scene_rgb, max_size=1024)
    try:
        scene_mask = estimate_subject_mask(scene_seg, model_name=model_name)
        scene_mask = align_mask(scene_mask, scene_rgb.size, firm=True)
        scene_bin = silhouette_binary_from_mask(scene_mask, threshold=64)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Scene rembg failed during paired mask (%s); using layer only", exc)
        scene_bin = only_bin

    # Centroid-align the clean layer mask onto the scene silhouette.
    cy_o, cx_o = _centroid(only_bin)
    cy_s, cx_s = _centroid(scene_bin)
    dy = int(round(cy_s - cy_o))
    dx = int(round(cx_s - cx_o))
    aligned = _shift_binary(only_bin, dy, dx) if (dy or dx) else only_bin

    if absdiff_boost:
        a = np.asarray(scene_rgb, dtype=np.int16)
        b = np.asarray(only_rgb, dtype=np.int16)
        # After alignment of the mask we still absdiff the unshifted pixels —
        # use difference mainly to fill holes inside the aligned silhouette.
        delta = np.mean(np.abs(a - b), axis=-1)
        # Pixels that differ and sit near the aligned subject reinforce it.
        near = ndimage.binary_dilation(aligned, iterations=6)
        boost = near & (delta >= 18)
        aligned = aligned | boost

    aligned = ndimage.binary_fill_holes(
        ndimage.binary_closing(aligned, iterations=5)
    )
    # Keep the largest component.
    labelled, count = ndimage.label(aligned)
    if count > 1:
        sizes = np.bincount(labelled.ravel())
        sizes[0] = 0
        aligned = labelled == int(sizes.argmax())

    # Prefer layer mask when it still overlaps the scene rembg well; otherwise
    # fall back to the scene rembg (pose may have drifted too far).
    inter = float((aligned & scene_bin).sum())
    union = float((aligned | scene_bin).sum()) or 1.0
    iou = inter / union
    if iou < 0.25 and scene_bin.any():
        logger.warning(
            "Paired-layer mask IoU≈%.2f with scene rembg; falling back to scene mask",
            iou,
        )
        aligned = scene_bin

    if firm:
        alpha = np.where(aligned, 255, 0).astype(np.uint8)
    else:
        alpha = (aligned.astype(np.uint8) * 255)
    logger.info(
        "Paired-layer subject mask: fg %.1f%% (IoU≈%.2f vs scene rembg, shift %+d,%+d)",
        100.0 * aligned.mean(),
        iou,
        dy,
        dx,
    )
    return SubjectMask(
        alpha=alpha,
        model=f"paired:{model_name}",
        foreground_fraction=float(aligned.mean()),
    )


def prepare_subject_image(
    image: Image.Image,
    *,
    mode: str = "dual",
    model_name: str = "u2net",
    background: tuple[int, int, int] = DEFAULT_BG,
    autocrop: bool = True,
    padding_fraction: float = 0.12,
    subject_fill: float = DEFAULT_SUBJECT_FILL,
    firm_border: bool = True,
    segment_max_size: int = 1024,
    colour_refine: bool = True,
    subject_mask: SubjectMask | None = None,
) -> tuple[Image.Image, SubjectMask | None]:
    """Prepare an image for colour-by-numbers with optional subject isolation.

    Segmentation runs on a downscaled copy for speed, but the mask is mapped
    back and the crop is taken from the **full-resolution** source so native
    print DPI is preserved. When ``colour_refine`` is True, silhouette pixels
    near the edge are snapped using subject vs background colour contrast.

    When ``subject_mask`` is provided (e.g. from a paired subject-only fal
    layer), rembg on ``image`` is skipped and that mask is used instead.

    Modes:
      - ``off``: unchanged image
      - ``isolate``: flat background + optional fill crop
      - ``dual``: keep scene, crop so subject fills ``subject_fill`` of frame
        (default 80%) for fine-on-subject / light-on-background processing
      - ``mask-only``: return mask without changing pixels
    """
    from .quantize import resize_for_processing

    mode = mode.lower().strip()
    rgb = image.convert("RGB")
    if mode in {"off", "none", "false", "0"}:
        return rgb, None

    if subject_mask is not None:
        mask = align_mask(subject_mask, rgb.size, firm=firm_border)
        if firm_border:
            mask = harden_mask(mask)
    else:
        # rembg on a moderate canvas, then lift the mask to native resolution.
        seg = resize_for_processing(rgb, max_size=segment_max_size)
        mask = estimate_subject_mask(seg, model_name=model_name)
        if firm_border:
            mask = harden_mask(mask)
        mask = align_mask(mask, rgb.size, firm=firm_border)
    if colour_refine:
        from .contrast import refine_mask_by_colour

        mask = refine_mask_by_colour(rgb, mask)

    if mode == "mask-only":
        return rgb, mask

    if mode in {"dual", "hybrid", "split"}:
        if autocrop:
            cropped, mask = crop_to_subject_fill(
                rgb, mask, target_fill=subject_fill, pad_colour=background
            )
        else:
            cropped, mask = rgb, mask
        if colour_refine:
            from .contrast import refine_mask_by_colour

            mask = refine_mask_by_colour(cropped, mask)
        if firm_border:
            mask = harden_mask(mask)
        return cropped, mask

    if mode not in {"isolate", "on", "true", "1"}:
        raise ValueError(
            f"Unknown subject mode {mode!r}; use off, isolate, dual, or mask-only"
        )

    # Isolate uses a hard paste when firm borders are requested.
    if firm_border:
        mask = harden_mask(mask)
        base = Image.new("RGB", rgb.size, background)
        out = np.asarray(base, dtype=np.uint8).copy()
        src = np.asarray(rgb, dtype=np.uint8)
        out[mask.binary] = src[mask.binary]
        isolated = Image.fromarray(out, mode="RGB")
    else:
        isolated, mask = isolate_on_flat_background(
            rgb, mask, background=background, model_name=model_name
        )
    if autocrop:
        isolated, mask = crop_to_subject_fill(
            isolated, mask, target_fill=subject_fill, pad_colour=background
        )
    if colour_refine:
        from .contrast import refine_mask_by_colour

        mask = refine_mask_by_colour(isolated, mask)
    if firm_border:
        mask = harden_mask(mask)
    return isolated, mask
