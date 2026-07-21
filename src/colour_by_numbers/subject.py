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


def align_mask(mask: SubjectMask, size: tuple[int, int]) -> SubjectMask:
    """Resize a subject mask to an image size ``(width, height)``."""
    width, height = size
    if mask.alpha.shape == (height, width):
        return mask
    alpha_img = Image.fromarray(mask.alpha, mode="L").resize(
        (width, height), Image.Resampling.BILINEAR
    )
    alpha = np.asarray(alpha_img, dtype=np.uint8)
    return SubjectMask(
        alpha=alpha,
        model=mask.model,
        foreground_fraction=float((alpha >= 128).mean()),
    )


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


def blend_subject_background(
    subject_image: Image.Image,
    background_image: Image.Image,
    mask: SubjectMask,
) -> Image.Image:
    """Alpha-composite subject over background using the subject mask."""
    subject_image = subject_image.convert("RGB")
    background_image = background_image.convert("RGB").resize(
        subject_image.size, Image.Resampling.BILINEAR
    )
    mask = align_mask(mask, subject_image.size)
    out = background_image.copy()
    rgba = subject_image.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask.alpha, mode="L"))
    out.paste(rgba, mask=rgba.split()[-1])
    return out


def prepare_subject_image(
    image: Image.Image,
    *,
    mode: str = "dual",
    model_name: str = "u2net",
    background: tuple[int, int, int] = DEFAULT_BG,
    autocrop: bool = True,
    padding_fraction: float = 0.12,
    subject_fill: float = DEFAULT_SUBJECT_FILL,
) -> tuple[Image.Image, SubjectMask | None]:
    """Prepare an image for colour-by-numbers with optional subject isolation.

    Modes:
      - ``off``: unchanged image
      - ``isolate``: flat background + optional padding crop
      - ``dual``: keep scene, crop so subject fills ``subject_fill`` of frame
        (default 80%) for fine-on-subject / light-on-background processing
      - ``mask-only``: return mask without changing pixels
    """
    mode = mode.lower().strip()
    rgb = image.convert("RGB")
    if mode in {"off", "none", "false", "0"}:
        return rgb, None
    if mode == "mask-only":
        return rgb, estimate_subject_mask(rgb, model_name=model_name)

    mask = estimate_subject_mask(rgb, model_name=model_name)

    if mode in {"dual", "hybrid", "split"}:
        if autocrop:
            cropped, mask = crop_to_subject_fill(
                rgb, mask, target_fill=subject_fill, pad_colour=background
            )
        else:
            cropped, mask = rgb, mask
        # Upscale the cropped plate so the page is still printably large.
        return cropped, mask

    if mode not in {"isolate", "on", "true", "1"}:
        raise ValueError(
            f"Unknown subject mode {mode!r}; use off, isolate, dual, or mask-only"
        )

    isolated, mask = isolate_on_flat_background(
        rgb, mask, background=background, model_name=model_name
    )
    if autocrop:
        isolated, mask = crop_to_subject_fill(
            isolated, mask, target_fill=subject_fill, pad_colour=background
        )
    return isolated, mask
