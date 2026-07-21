"""Subject-aware preprocessing using neural background removal (rembg / U²-Net).

Colour-by-numbers fails when a tiny subject sits in a huge sky, or when clutter
shares the subject's colours. This module estimates a foreground mask, replaces
the background with a flat fill, and optionally crops tightly around the subject
so it dominates the page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_BG = (248, 248, 248)


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
    if mask.alpha.shape != (rgb.height, rgb.width):
        alpha_img = Image.fromarray(mask.alpha, mode="L").resize(
            rgb.size, Image.Resampling.BILINEAR
        )
        mask = SubjectMask(
            alpha=np.asarray(alpha_img, dtype=np.uint8),
            model=mask.model,
            foreground_fraction=float((np.asarray(alpha_img) >= 128).mean()),
        )

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
    """Crop around the foreground bbox so small subjects fill the frame.

    If the mask is empty or tiny, returns the inputs unchanged.
    """
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


def prepare_subject_image(
    image: Image.Image,
    *,
    mode: str = "isolate",
    model_name: str = "u2net",
    background: tuple[int, int, int] = DEFAULT_BG,
    autocrop: bool = True,
    padding_fraction: float = 0.12,
) -> tuple[Image.Image, SubjectMask | None]:
    """Prepare an image for colour-by-numbers with optional subject isolation.

    Modes:
      - ``off``: return the image unchanged
      - ``isolate``: flat background + optional autocrop (recommended)
      - ``mask-only``: return mask diagnostics without changing pixels
    """
    mode = mode.lower().strip()
    if mode in {"off", "none", "false", "0"}:
        return image.convert("RGB"), None
    if mode == "mask-only":
        return image.convert("RGB"), estimate_subject_mask(
            image, model_name=model_name
        )
    if mode not in {"isolate", "on", "true", "1"}:
        raise ValueError(
            f"Unknown subject mode {mode!r}; use off, isolate, or mask-only"
        )

    isolated, mask = isolate_on_flat_background(
        image, background=background, model_name=model_name
    )
    if autocrop:
        isolated, mask = crop_to_subject(
            isolated, mask, padding_fraction=padding_fraction
        )
    return isolated, mask
