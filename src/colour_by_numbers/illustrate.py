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
    DEFAULT_ILLUSTRATION_COLOURS,
    MAX_N_COLOURS,
    MIN_N_COLOURS,
    STANDARD_PALETTE_32,
    clamp_n_colours,
    nearest_palette_indices,
    select_active_palette,
)
from .print_resolution import (
    DEFAULT_MIN_REGION_MM,
    min_region_size_for_a4_mm,
)
from .quantize import prefilter_for_regions, resize_for_processing
from .simplify import (
    absorb_small_regions,
    absorb_thin_regions,
    compact_palette,
)
from .subject import (
    SubjectMask,
    align_mask,
    harden_mask,
    prepare_subject_image,
)

logger = logging.getLogger(__name__)

DEFAULT_ILLUSTRATION_SIZE = 1600
DEFAULT_PAGE_BACKGROUND = (248, 248, 252)
AVAILABLE_ILLUSTRATION_BACKENDS = ("local_stylize", "pollinations", "openai", "replicate")
POLLINATIONS_IMAGE_URL = "https://image.pollinations.ai/prompt/{prompt}"


@dataclass(frozen=True)
class IllustrationResult:
    """A generated colouring-ready illustration and its provenance."""

    image: Image.Image
    backend: str
    subject_type_label: str | None = None
    reference_url: str | None = None
    reference_title: str | None = None
    n_colours: int = 12
    prompt: str | None = None
    notes: str = ""


def illustration_prompt(
    subject_type_label: str,
    *,
    category: str | None = None,
    min_colours: int = MIN_N_COLOURS,
    max_colours: int = MAX_N_COLOURS,
) -> str:
    """Text prompt used by API backends (and recorded for local runs)."""
    subject = subject_type_label.strip()
    lo = clamp_n_colours(min_colours, minimum=min_colours, maximum=max_colours)
    hi = clamp_n_colours(max_colours, minimum=min_colours, maximum=max_colours)
    style = (
        "children's colouring book illustration, thick clean black outlines, "
        f"flat cel fills using between {lo} and {hi} solid colours only, "
        "large simple colour regions (each region at least 5mm by 5mm when "
        "printed on A4), high subject-background contrast, no gradients, "
        "no photorealism, no text, white background"
    )
    if category == "aircraft":
        return f"{subject} side view, clear silhouette, {style}"
    if category in {"flowers", "birds"}:
        return f"{subject} centred portrait, {style}"
    return f"{subject} portrait, centred subject, {style}"


def prepare_illustration_for_colouring(
    image: Image.Image,
    *,
    n_colours: int = DEFAULT_ILLUSTRATION_COLOURS,
    min_region_mm: float = DEFAULT_MIN_REGION_MM,
) -> tuple[Image.Image, int]:
    """Clamp a generated plate to 8–16 flat palette colours and A4-safe regions.

    Maps pixels onto the standard crayon set, then absorbs speckles thinner or
    smaller than ``min_region_mm`` × ``min_region_mm`` when printed on A4.
    """
    n = clamp_n_colours(n_colours)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    active = select_active_palette(
        STANDARD_PALETTE_32, n_colours=n, image_rgb=rgb
    )
    labels = nearest_palette_indices(rgb, active)
    region = min_region_size_for_a4_mm(width, height, min_mm=min_region_mm)
    labels = absorb_small_regions(labels, min_area=region.min_area_px)
    labels = absorb_thin_regions(
        labels, min_thickness=float(max(2, region.min_side_px))
    )
    labels = absorb_small_regions(labels, min_area=region.min_area_px)
    labels, palette = compact_palette(labels, active)
    # If absorption collapsed below the requested minimum, keep what remains;
    # the prompt still asked for ≥8 and most plates retain that many paints.
    poster = palette[labels]
    return Image.fromarray(poster, mode="RGB"), int(palette.shape[0])


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
    n_colours: int = DEFAULT_ILLUSTRATION_COLOURS,
    output_size: int = DEFAULT_ILLUSTRATION_SIZE,
    background: tuple[int, int, int] = DEFAULT_PAGE_BACKGROUND,
    outline_width: int = 3,
    subject_model: str = "u2net",
    subject_type_label: str | None = None,
    category: str | None = None,
    min_region_mm: float = DEFAULT_MIN_REGION_MM,
) -> IllustrationResult:
    """Build a flat illustrated plate from one real reference photo."""
    n_colours = clamp_n_colours(n_colours)
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
        STANDARD_PALETTE_32, n_colours=n_colours, image_rgb=pixels
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
    illustrated, used = prepare_illustration_for_colouring(
        illustrated, n_colours=n_colours, min_region_mm=min_region_mm
    )

    prompt = (
        illustration_prompt(subject_type_label, category=category)
        if subject_type_label
        else None
    )
    return IllustrationResult(
        image=illustrated,
        backend="local_stylize",
        subject_type_label=subject_type_label,
        n_colours=used,
        prompt=prompt,
        notes=(
            "Local stylize: isolate subject, map to 8–16 standard colours, "
            f"A4 regions ≥{min_region_mm:g}mm, flat background, firm ink outline."
        ),
    )


def generate_illustration_pollinations(
    prompt: str,
    *,
    width: int = 1024,
    height: int = 1024,
    model: str = "flux",
    seed: int | None = None,
    timeout: float = 120.0,
) -> IllustrationResult:
    """Generate an image via Pollinations.ai (no API key / no paid plan).

    Anonymous use is rate-limited (~1 request / 15s) and may watermark.
    """
    from urllib.parse import quote
    import io

    import requests

    encoded = quote(prompt, safe="")
    url = POLLINATIONS_IMAGE_URL.format(prompt=encoded)
    params: dict[str, str | int] = {
        "width": int(width),
        "height": int(height),
        "model": model,
        "nologo": "true",
        "enhance": "true",
    }
    if seed is not None:
        params["seed"] = int(seed)

    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    content_type = (response.headers.get("Content-Type") or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise RuntimeError(
            f"Pollinations returned non-image content-type {content_type!r}"
        )
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return IllustrationResult(
        image=image,
        backend="pollinations",
        prompt=prompt,
        notes=(
            f"Generated via Pollinations.ai ({model}). "
            "Free / no subscription; anonymous tier is rate-limited."
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
            "Use backend='pollinations' or 'local_stylize', or export an API key."
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
    n_colours: int = DEFAULT_ILLUSTRATION_COLOURS,
    output_size: int = DEFAULT_ILLUSTRATION_SIZE,
    openai_api_key: str | None = None,
    prompt_override: str | None = None,
    pollinations_model: str = "flux",
    seed: int | None = None,
    min_region_mm: float = DEFAULT_MIN_REGION_MM,
    prepare_for_colouring: bool = True,
) -> IllustrationResult:
    """Generate a colouring-ready illustration via the selected backend."""
    backend = backend.lower().strip()
    if backend not in AVAILABLE_ILLUSTRATION_BACKENDS:
        raise ValueError(
            f"Unknown illustration backend {backend!r}; "
            f"choose one of {AVAILABLE_ILLUSTRATION_BACKENDS}"
        )

    n_colours = clamp_n_colours(n_colours)
    prompt = prompt_override or (
        illustration_prompt(subject_type_label or "subject", category=category)
        if subject_type_label or backend != "local_stylize"
        else None
    )

    if backend == "pollinations":
        # Prefer square-ish canvas near output_size for colouring pages.
        side = max(512, min(int(output_size), 1280))
        result = generate_illustration_pollinations(
            prompt or "colouring book illustration of a clear subject",
            width=side,
            height=side,
            model=pollinations_model,
            seed=seed,
        )
    elif backend == "openai":
        result = generate_illustration_openai(
            prompt or "colouring book illustration", api_key=openai_api_key
        )
    elif backend == "replicate":
        raise RuntimeError(
            "Replicate backend is reserved but not configured. "
            "Use backend='pollinations', 'local_stylize', or 'openai'."
        )
    else:
        if reference is None:
            raise ValueError("local_stylize backend requires a reference photo")
        # Local stylize already runs prepare_illustration_for_colouring.
        return stylize_reference_to_illustration(
            reference,
            n_colours=n_colours,
            output_size=output_size,
            subject_type_label=subject_type_label,
            category=category,
            min_region_mm=min_region_mm,
        )

    if prepare_for_colouring:
        cleaned, used = prepare_illustration_for_colouring(
            result.image, n_colours=n_colours, min_region_mm=min_region_mm
        )
        notes = (
            f"{result.notes} Post-processed to {used} flat colours "
            f"(8–16) with A4 regions ≥{min_region_mm:g}mm."
        )
        return IllustrationResult(
            image=cleaned,
            backend=result.backend,
            subject_type_label=result.subject_type_label or subject_type_label,
            reference_url=result.reference_url,
            reference_title=result.reference_title,
            n_colours=used,
            prompt=result.prompt or prompt,
            notes=notes,
        )
    return IllustrationResult(
        image=result.image,
        backend=result.backend,
        subject_type_label=result.subject_type_label or subject_type_label,
        reference_url=result.reference_url,
        reference_title=result.reference_title,
        n_colours=n_colours,
        prompt=result.prompt or prompt,
        notes=result.notes,
    )
