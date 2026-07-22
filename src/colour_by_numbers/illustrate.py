"""Turn reference photos into colouring-book-ready flat illustrations.

This is the local generation backend: it does not invent pixels with a
diffusion model. Instead it builds a balanced artistic plate from a real
reference (subject isolation, flat fills from the standard palette, firm
ink outline, clean background) so the downstream colour-by-numbers step
starts from illustration-like art rather than raw photography.

A pluggable ``backend`` hook is reserved for future API generators
(OpenAI / Replicate / etc.) when credentials are available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .palette import (
    STANDARD_PALETTE_32,
    nearest_palette_indices,
    select_active_palette,
)
from .quantize import prefilter_for_regions, resize_for_processing
from .subject import (
    SubjectMask,
    align_mask,
    harden_mask,
    prepare_subject_image,
)

logger = logging.getLogger(__name__)

DEFAULT_ILLUSTRATION_SIZE = 1600
DEFAULT_PAGE_BACKGROUND = (248, 248, 252)
AVAILABLE_ILLUSTRATION_BACKENDS = ("local_stylize", "openai", "replicate")


@dataclass(frozen=True)
class IllustrationResult:
    """A generated colouring-ready illustration and its provenance."""

    image: Image.Image
    backend: str
    subject_type_label: str | None = None
    reference_url: str | None = None
    reference_title: str | None = None
    n_colours: int = 16
    prompt: str | None = None
    notes: str = ""


def illustration_prompt(subject_type_label: str, *, category: str | None = None) -> str:
    """Text prompt used by API backends (and recorded for local runs)."""
    subject = subject_type_label.strip()
    style = (
        "children's colouring book illustration, thick clean black outlines, "
        "flat cel fills, limited palette, high subject-background contrast, "
        "simple shapes, no photorealism, no text, white background"
    )
    if category == "aircraft":
        return f"{subject} side view, clear silhouette, {style}"
    if category in {"flowers", "birds"}:
        return f"{subject} centred portrait, {style}"
    return f"{subject} portrait, centred subject, {style}"


def _smooth_flat(image: Image.Image, *, radius: float = 2.4) -> Image.Image:
    """Strong cartoon prefilter: blur + median-like smooth."""
    work = prefilter_for_regions(image, blur_radius=radius)
    work = work.filter(ImageFilter.SMOOTH_MORE)
    work = work.filter(ImageFilter.SMOOTH_MORE)
    return work


def _ink_outline(
    mask: SubjectMask,
    *,
    size: tuple[int, int],
    width: int = 3,
) -> Image.Image:
    """Binary ink outline from the firm subject silhouette."""
    from scipy import ndimage

    hard = align_mask(harden_mask(mask), size, firm=True).binary
    eroded = ndimage.binary_erosion(hard, iterations=max(1, width))
    edge = hard & ~eroded
    if width > 1:
        edge = ndimage.binary_dilation(edge, iterations=width - 1)
    outline = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    outline[:] = 255
    outline[edge] = (20, 20, 20)
    return Image.fromarray(outline, mode="RGB")


def stylize_reference_to_illustration(
    reference: Image.Image,
    *,
    n_colours: int = 16,
    output_size: int = DEFAULT_ILLUSTRATION_SIZE,
    background: tuple[int, int, int] = DEFAULT_PAGE_BACKGROUND,
    outline_width: int = 3,
    subject_model: str = "u2net",
    subject_type_label: str | None = None,
    category: str | None = None,
) -> IllustrationResult:
    """Build a flat illustrated plate from one real reference photo."""
    rgb = reference.convert("RGB")
    # Work from a generous canvas so the plate is A4-friendly.
    working = resize_for_processing(rgb, max_size=max(output_size, 1200))
    prepared, mask = prepare_subject_image(
        working,
        mode="isolate",
        model_name=subject_model,
        autocrop=True,
        subject_fill=0.82,
        firm_border=True,
        colour_refine=True,
    )
    if mask is None:
        prepared = working
        # Full-frame fallback mask (everything is subject).
        alpha = np.full((prepared.height, prepared.width), 255, dtype=np.uint8)
        mask = SubjectMask(alpha=alpha, model="none", foreground_fraction=1.0)

    # Upscale/downscale the isolated plate toward the target illustration size.
    longest = max(prepared.size)
    if longest != output_size:
        scale = output_size / longest
        target = (
            max(1, int(prepared.width * scale)),
            max(1, int(prepared.height * scale)),
        )
        prepared = prepared.resize(target, Image.Resampling.LANCZOS)
        mask = align_mask(harden_mask(mask), prepared.size, firm=True)

    flat = _smooth_flat(prepared, radius=2.6)
    pixels = np.asarray(flat, dtype=np.uint8)
    active = select_active_palette(
        STANDARD_PALETTE_32, n_colours=min(n_colours, 32), image_rgb=pixels
    )
    labels = nearest_palette_indices(pixels, active)
    poster = active[labels]

    # Force a clean flat background outside the firm mask.
    hard = harden_mask(mask).binary
    poster = poster.copy()
    poster[~hard] = np.asarray(background, dtype=np.uint8)

    illustrated = Image.fromarray(poster, mode="RGB")
    ink = _ink_outline(mask, size=illustrated.size, width=outline_width)
    # Composite dark ink onto the flat plate.
    ink_arr = np.asarray(ink)
    out = np.asarray(illustrated).copy()
    ink_pixels = ink_arr[:, :, 0] < 40
    out[ink_pixels] = (20, 20, 20)
    illustrated = Image.fromarray(out, mode="RGB")

    # Mild contrast so flat fills read clearly when printed.
    illustrated = ImageOps.autocontrast(illustrated, cutoff=0.5)

    prompt = (
        illustration_prompt(subject_type_label, category=category)
        if subject_type_label
        else None
    )
    return IllustrationResult(
        image=illustrated,
        backend="local_stylize",
        subject_type_label=subject_type_label,
        n_colours=int(len(np.unique(labels))),
        prompt=prompt,
        notes=(
            "Local stylize: isolate subject, map to standard palette, "
            "flat background, firm ink outline."
        ),
    )


def generate_illustration_openai(
    prompt: str,
    *,
    api_key: str | None = None,
    size: str = "1024x1024",
) -> IllustrationResult:
    """Optional OpenAI Images backend (requires OPENAI_API_KEY)."""
    import os

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OpenAI backend requested but OPENAI_API_KEY is not set. "
            "Use backend='local_stylize' or export an API key."
        )
    try:
        import urllib.request
        import json
        import base64
        import io
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("urllib unavailable") from exc

    payload = json.dumps(
        {
            "model": "gpt-image-1",
            "prompt": prompt,
            "size": size,
            "n": 1,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/images/generations",
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    b64 = body["data"][0].get("b64_json")
    if not b64:
        raise RuntimeError("OpenAI response missing image data")
    image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return IllustrationResult(
        image=image,
        backend="openai",
        prompt=prompt,
        notes="Generated via OpenAI Images API.",
    )


def generate_illustration(
    reference: Image.Image | None = None,
    *,
    subject_type_label: str | None = None,
    category: str | None = None,
    backend: str = "local_stylize",
    n_colours: int = 16,
    output_size: int = DEFAULT_ILLUSTRATION_SIZE,
    openai_api_key: str | None = None,
) -> IllustrationResult:
    """Generate a colouring-ready illustration via the selected backend."""
    backend = backend.lower().strip()
    if backend not in AVAILABLE_ILLUSTRATION_BACKENDS:
        raise ValueError(
            f"Unknown illustration backend {backend!r}; "
            f"choose one of {AVAILABLE_ILLUSTRATION_BACKENDS}"
        )

    prompt = (
        illustration_prompt(subject_type_label or "subject", category=category)
        if subject_type_label or backend != "local_stylize"
        else None
    )

    if backend == "openai":
        return generate_illustration_openai(prompt or "colouring book illustration", api_key=openai_api_key)

    if backend == "replicate":
        raise RuntimeError(
            "Replicate backend is reserved but not configured in this environment. "
            "Use backend='local_stylize' or 'openai'."
        )

    if reference is None:
        raise ValueError("local_stylize backend requires a reference photo")
    return stylize_reference_to_illustration(
        reference,
        n_colours=n_colours,
        output_size=output_size,
        subject_type_label=subject_type_label,
        category=category,
    )
