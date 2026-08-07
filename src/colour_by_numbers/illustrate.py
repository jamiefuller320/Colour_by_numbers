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
from PIL import Image, ImageFilter, ImageOps

from .discover import (
    CATEGORY_NEGATIVE_CUES,
    disambiguate_subject_label,
    subject_kind_frame,
)
from .palette import (
    DEFAULT_ILLUSTRATION_COLOURS,
    EARTHY_CATEGORIES,
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
    enforce_colourable_blocks,
    merge_adjacent_same_colour,
    normalize_specular_highlights,
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
AVAILABLE_ILLUSTRATION_BACKENDS = (
    "fal",
    "local_stylize",
    "pollinations",
    "openai",
    "replicate",
)
POLLINATIONS_IMAGE_URL = "https://image.pollinations.ai/prompt/{prompt}"
POLLINATIONS_GEN_IMAGE_URL = "https://gen.pollinations.ai/image/{prompt}"
FAL_RUN_URL = "https://fal.run/{model_id}"
DEFAULT_FAL_MODEL = "fal-ai/flux/schnell"
FAL_MODEL_ALIASES = {
    "schnell": "fal-ai/flux/schnell",
    "flux": "fal-ai/flux/schnell",
    "flux-schnell": "fal-ai/flux/schnell",
    "dev": "fal-ai/flux/dev",
    "flux-dev": "fal-ai/flux/dev",
}


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
    subject = disambiguate_subject_label(subject_type_label, category=category)
    lo = clamp_n_colours(min_colours, minimum=min_colours, maximum=max_colours)
    hi = clamp_n_colours(max_colours, minimum=min_colours, maximum=max_colours)
    style = (
        "children's colouring book illustration, thick clean black outlines, "
        "smooth colour-region boundaries, "
        f"flat cel fills using between {lo} and {hi} solid colours only, "
        f"large simple colour regions (each colourable block at least "
        f"{DEFAULT_MIN_REGION_MM:g}mm wide and {DEFAULT_MIN_REGION_MM:g}mm high "
        f"when printed on A4, with finer detail as black line drawing), "
        "high subject-background contrast, no gradients, "
        "no photorealism, no text, white background, "
        "full subject in frame with a small margin, not over-cropped"
    )
    animal_detail = (
        "large expressive matching eyes, each eye with a separate dark pupil and "
        "lighter iris or sclera fill distinct from surrounding fur "
        "(at least two distinct colour regions per eye), "
        "clearly defined nose and muzzle with visible nostrils and wrinkles "
        "where applicable, sharp facial feature definition, "
        "clear value steps between head, neck and body for depth, "
        "warm natural colours"
    )
    bird_detail = (
        "species-accurate plumage colours and markings, "
        "large expressive matching eyes with separate dark pupil and lighter "
        "iris fill distinct from feathers, clearly defined beak "
        "(not a mammal nose), sharp feature definition"
    )
    people_detail = (
        "clear facial features, large expressive eyes with separate dark pupils "
        "and lighter iris or sclera fills (at least two colour regions per eye), "
        "defined nose and mouth, natural skin tones"
    )
    vehicle_detail = (
        "complete vehicle silhouette in frame, separate colour regions for "
        "body panels, windows, and structural parts"
    )
    negative = CATEGORY_NEGATIVE_CUES.get(category or "", "")
    negative_suffix = f", {negative}" if negative else ""
    kind = subject_kind_frame(category)
    kind_prefix = f"{kind}. " if kind else ""
    if category == "aircraft":
        return (
            f"{kind_prefix}{subject} side view, clear silhouette, {vehicle_detail}"
            f"{negative_suffix}, {style}"
        )
    if category == "flowers":
        return (
            f"{kind_prefix}{subject} whole flower centred, petals and centre "
            f"disk clearly recognisable, species-typical colours, stem or leaves "
            f"visible if needed for identity{negative_suffix}, {style}"
        )
    if category == "birds":
        return (
            f"{kind_prefix}{subject} centred portrait, {bird_detail}"
            f"{negative_suffix}, {style}"
        )
    if category in {"people", "portraits"}:
        return (
            f"{kind_prefix}{subject} portrait, centred face, {people_detail}"
            f"{negative_suffix}, {style}"
        )
    if category in EARTHY_CATEGORIES:
        return (
            f"{kind_prefix}{subject} portrait, centred subject, {animal_detail}"
            f"{negative_suffix}, {style}"
        )
    if category in {"cars", "boats"}:
        return (
            f"{kind_prefix}{subject} side view, clear silhouette, {vehicle_detail}"
            f"{negative_suffix}, {style}"
        )
    return f"{kind_prefix}{subject} portrait, centred subject{negative_suffix}, {style}"


def prepare_illustration_for_colouring(
    image: Image.Image,
    *,
    n_colours: int = DEFAULT_ILLUSTRATION_COLOURS,
    min_region_mm: float = DEFAULT_MIN_REGION_MM,
    category: str | None = None,
) -> tuple[Image.Image, int]:
    """Clamp a generated plate to 8–16 flat palette colours and A4-safe regions.

    Colourable blocks must be at least ``min_region_mm`` wide **and** high on
    A4 (enough for a circular tip of that diameter). Smaller features are kept
    as black line detail instead of numbered fills. For animal categories, dark
    pixels stay on warm neutrals / browns.
    """
    n = clamp_n_colours(n_colours)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    active = select_active_palette(
        STANDARD_PALETTE_32,
        n_colours=n,
        image_rgb=rgb,
        category=category,
    )
    labels = nearest_palette_indices(rgb, active, category=category)
    region = min_region_size_for_a4_mm(width, height, min_mm=min_region_mm)
    tip = float(max(2, region.min_inscribed_diameter_px))
    # Merge split same-colour islands (e.g. both eye whites) before the
    # colourable-block rules. Do not morphologically open/thicken strokes —
    # that flattens fine illustration detail.
    labels = merge_adjacent_same_colour(labels, bridge_px=max(2.0, tip * 0.6))
    labels = absorb_small_regions(labels, min_area=region.min_area_px)
    labels = absorb_thin_regions(labels, min_thickness=tip)
    from .eyes import compute_eye_protection_mask, portrait_subject, relaxed_eye_thresholds

    eye_protected = compute_eye_protection_mask(
        labels,
        active,
        category=category,
        min_region_mm=min_region_mm,
    )
    eye_relaxed = (
        relaxed_eye_thresholds(width, height) if portrait_subject(category) else None
    )
    labels, _hl_detail = normalize_specular_highlights(
        labels,
        active,
        min_width_px=region.min_width_px,
        min_height_px=region.min_height_px,
        min_inscribed_px=tip,
        protected=eye_protected if eye_protected.any() else None,
        protected_relaxed=eye_relaxed,
    )
    labels, detail = enforce_colourable_blocks(
        labels,
        min_width_px=region.min_width_px,
        min_height_px=region.min_height_px,
        min_inscribed_px=tip,
        protected=eye_protected if eye_protected.any() else None,
        protected_relaxed=eye_relaxed,
    )
    labels, palette = compact_palette(labels, active)
    poster = palette[labels]
    if detail.any():
        poster = poster.copy()
        poster[detail] = (18, 18, 18)
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
    outline_width: int = 2,
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
        STANDARD_PALETTE_32,
        n_colours=n_colours,
        image_rgb=pixels,
        category=category,
    )
    labels = nearest_palette_indices(pixels, active, category=category)
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
    out[ink_pixels] = (18, 18, 18)
    illustrated = Image.fromarray(out, mode="RGB")

    # Mild contrast so flat fills read clearly when printed.
    illustrated = ImageOps.autocontrast(illustrated, cutoff=0.5)
    illustrated, used = prepare_illustration_for_colouring(
        illustrated,
        n_colours=n_colours,
        min_region_mm=min_region_mm,
        category=category,
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


def resolve_fal_model_id(model: str | None = None) -> str:
    """Map short aliases (schnell/dev) to fal endpoint ids."""
    raw = (model or DEFAULT_FAL_MODEL).strip()
    return FAL_MODEL_ALIASES.get(raw.lower(), raw)


def generate_illustration_fal(
    prompt: str,
    *,
    width: int = 1024,
    height: int = 1024,
    model: str = DEFAULT_FAL_MODEL,
    seed: int | None = None,
    api_key: str | None = None,
    timeout: float = 180.0,
    num_inference_steps: int = 4,
) -> IllustrationResult:
    """Generate an image via fal.ai (Flux). Requires ``FAL_KEY``.

    Uses the synchronous ``fal.run`` HTTP API (no extra SDK dependency).
    """
    import io
    import os

    import requests

    key = (api_key or os.environ.get("FAL_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "fal backend requested but FAL_KEY is not set. "
            "Create a key at https://fal.ai/dashboard/keys and export FAL_KEY."
        )

    model_id = resolve_fal_model_id(model)
    url = FAL_RUN_URL.format(model_id=model_id)
    payload: dict = {
        "prompt": prompt,
        "image_size": {"width": int(width), "height": int(height)},
        "num_images": 1,
        "enable_safety_checker": True,
        "output_format": "png",
        "num_inference_steps": int(num_inference_steps),
    }
    if seed is not None:
        payload["seed"] = int(seed)

    response = requests.post(
        url,
        headers={
            "Authorization": f"Key {key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if not response.ok:
        detail = (response.text or "")[:300]
        raise RuntimeError(
            f"fal.ai request failed (HTTP {response.status_code}): {detail}"
        )
    body = response.json()
    images = body.get("images") or []
    if not images:
        raise RuntimeError("fal.ai response missing images")
    image_url = images[0].get("url")
    if not image_url:
        raise RuntimeError("fal.ai response missing image URL")

    image_response = requests.get(image_url, timeout=timeout)
    image_response.raise_for_status()
    image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
    return IllustrationResult(
        image=image,
        backend="fal",
        prompt=prompt,
        notes=(
            f"Generated via fal.ai ({model_id}). "
            "Production primary backend; requires FAL_KEY."
        ),
    )


def _pollinations_error_message(response) -> str:
    """Turn Pollinations JSON / wrapped 402 pollen errors into a clear message."""
    text = (response.text or "").strip()
    try:
        import json

        data = json.loads(text)
    except Exception:  # noqa: BLE001
        return text[:300] or f"HTTP {response.status_code}"

    parts: list[str] = []
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            parts.append(str(err["message"]))
        elif isinstance(err, str) and err:
            parts.append(err)
        if data.get("message"):
            parts.append(str(data["message"]))
    message = " | ".join(parts) if parts else text[:300]
    lower = message.lower()
    if (
        "insufficient balance" in lower
        or "payment_required" in lower
        or "pollen" in lower
    ):
        return (
            "Pollinations needs Pollen credit and an API key "
            "(enter.pollinations.ai → set POLLINATIONS_API_KEY). "
            f"Server said: {message[:220]}"
        )
    if "authentication required" in lower or "unauthorized" in lower:
        return (
            "Pollinations authentication required. Export POLLINATIONS_API_KEY "
            f"from enter.pollinations.ai. Server said: {message[:220]}"
        )
    return message[:300] or f"HTTP {response.status_code}"


def generate_illustration_pollinations(
    prompt: str,
    *,
    width: int = 1024,
    height: int = 1024,
    model: str = "flux",
    seed: int | None = None,
    timeout: float = 120.0,
    api_key: str | None = None,
) -> IllustrationResult:
    """Legacy Pollinations backend (optional fallback).

    Prefer ``fal``. Pollinations expects an API key and Pollen balance for
    reliable use; without a key the anonymous host often fails with a wrapped
    HTTP 500 / 402.
    """
    import io
    import os
    from urllib.parse import quote

    import requests

    key = (api_key or os.environ.get("POLLINATIONS_API_KEY") or "").strip()
    encoded = quote(prompt, safe="")
    if key:
        url = POLLINATIONS_GEN_IMAGE_URL.format(prompt=encoded)
    else:
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
    headers: dict[str, str] = {}
    if key:
        params["key"] = key
        headers["Authorization"] = f"Bearer {key}"

    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    content_type = (response.headers.get("Content-Type") or "").lower()
    if not response.ok:
        raise RuntimeError(_pollinations_error_message(response))
    if content_type and not content_type.startswith("image/"):
        raise RuntimeError(
            "Pollinations returned non-image content-type "
            f"{content_type!r}: {_pollinations_error_message(response)}"
        )
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    auth_note = "authenticated gateway" if key else "legacy anonymous host"
    return IllustrationResult(
        image=image,
        backend="pollinations",
        prompt=prompt,
        notes=(
            f"Generated via Pollinations.ai ({model}, {auth_note}) — "
            "legacy fallback. Production primary is fal.ai."
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
            "Use backend='fal' or 'local_stylize', or export an API key."
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
    fal_api_key: str | None = None,
    pollinations_api_key: str | None = None,
    prompt_override: str | None = None,
    fal_model: str = DEFAULT_FAL_MODEL,
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
    if prompt:
        from .plate_critique import seed_prompt_with_plate_lessons

        prompt, _ = seed_prompt_with_plate_lessons(prompt, category=category)

    side = max(512, min(int(output_size), 1280))
    if backend == "fal":
        result = generate_illustration_fal(
            prompt or "colouring book illustration of a clear subject",
            width=side,
            height=side,
            model=fal_model,
            seed=seed,
            api_key=fal_api_key,
        )
    elif backend == "pollinations":
        result = generate_illustration_pollinations(
            prompt or "colouring book illustration of a clear subject",
            width=side,
            height=side,
            model=pollinations_model,
            seed=seed,
            api_key=pollinations_api_key,
        )
    elif backend == "openai":
        result = generate_illustration_openai(
            prompt or "colouring book illustration", api_key=openai_api_key
        )
    elif backend == "replicate":
        raise RuntimeError(
            "Replicate backend is reserved but not configured. "
            "Use backend='fal', 'local_stylize', or 'openai'."
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
            result.image,
            n_colours=n_colours,
            min_region_mm=min_region_mm,
            category=category,
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
