"""High-level orchestration for colour-by-numbers generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .outline import OutlinePage, build_outline_page, composite_page
from .print_resolution import evaluate_print_resolution
from .quantize import (
    QuantizedImage,
    prefilter_for_regions,
    preview_from_labels,
    quantize_colours,
    resize_for_processing,
    upsample_labels,
)
from .search import ImageHit, load_local_image, search_and_download
from .simplify import count_regions, simplify_dual
from .subject import (
    DEFAULT_SUBJECT_FILL,
    SubjectMask,
    align_mask,
    blend_subject_background,
    harden_mask,
    prepare_subject_image,
)


COMPLEXITY_PRESETS: dict[str, dict[str, float | int]] = {
    "raw": {
        "blur_radius": 0.0,
        "structure_size": 900,
        "min_area_fraction": 0.0,
        "max_regions": 10_000,
        "smooth_radius": 0,
        "morph_radius": 0,
        "boundary_sigma": 0.0,
        "line_width": 1,
        "simplify": 0,
    },
    "fine": {
        "blur_radius": 0.6,
        "structure_size": 580,
        "min_area_fraction": 0.0018,
        "max_regions": 110,
        "smooth_radius": 1,
        "morph_radius": 1,
        "boundary_sigma": 0.4,
        "line_width": 1,
        "simplify": 1,
    },
    "light": {
        "blur_radius": 1.0,
        "structure_size": 520,
        "min_area_fraction": 0.003,
        "max_regions": 80,
        "smooth_radius": 1,
        "morph_radius": 1,
        "boundary_sigma": 0.6,
        "line_width": 1,
        "simplify": 1,
    },
    "medium": {
        "blur_radius": 1.4,
        "structure_size": 460,
        "min_area_fraction": 0.005,
        "max_regions": 55,
        "smooth_radius": 2,
        "morph_radius": 1,
        "boundary_sigma": 0.8,
        "line_width": 1,
        "simplify": 1,
    },
    "simple": {
        "blur_radius": 2.4,
        "structure_size": 340,
        "min_area_fraction": 0.012,
        "max_regions": 30,
        "smooth_radius": 2,
        "morph_radius": 2,
        "boundary_sigma": 1.1,
        "line_width": 2,
        "simplify": 1,
    },
}

COMPLEXITY_PRESETS["detailed"] = dict(COMPLEXITY_PRESETS["light"])
COMPLEXITY_PRESETS["balanced"] = dict(COMPLEXITY_PRESETS["medium"])

DEMO_SPREAD_SETTINGS: tuple[str, ...] = ("fine",)
DEMO_SUBJECT_COMPARE: tuple[str, ...] = ("off", "isolate", "dual")


def _preset_simplify_params(
    preset: dict[str, float | int],
    *,
    width: int,
    height: int,
) -> dict:
    fraction = float(preset["min_area_fraction"])
    min_area = 1 if fraction <= 0 else max(20, int(width * height * fraction))
    return {
        "min_region_area": min_area,
        "max_regions": int(preset["max_regions"]),
        "smooth_radius": int(preset["smooth_radius"]),
        "morph_radius": int(preset["morph_radius"]),
        "boundary_sigma": float(preset["boundary_sigma"]),
        "smooth_iterations": 2,
    }


def _override_simplify_params(
    params: dict,
    *,
    zone: str,
    min_region_area: int | None = None,
    max_regions: int | None = None,
    smooth_radius: int | None = None,
    morph_radius: int | None = None,
    boundary_sigma: float | None = None,
    subject_min_region_area: int | None = None,
    subject_max_regions: int | None = None,
    subject_smooth_radius: int | None = None,
    subject_morph_radius: int | None = None,
    subject_boundary_sigma: float | None = None,
    background_min_region_area: int | None = None,
    background_max_regions: int | None = None,
    background_smooth_radius: int | None = None,
    background_morph_radius: int | None = None,
    background_boundary_sigma: float | None = None,
) -> dict:
    """Apply global then zone-specific overrides onto preset simplify params."""
    out = dict(params)
    if min_region_area is not None:
        out["min_region_area"] = min_region_area
    if max_regions is not None:
        out["max_regions"] = max_regions
    if smooth_radius is not None:
        out["smooth_radius"] = smooth_radius
    if morph_radius is not None:
        out["morph_radius"] = morph_radius
    if boundary_sigma is not None:
        out["boundary_sigma"] = boundary_sigma

    if zone == "subject":
        if subject_min_region_area is not None:
            out["min_region_area"] = subject_min_region_area
        if subject_max_regions is not None:
            out["max_regions"] = subject_max_regions
        if subject_smooth_radius is not None:
            out["smooth_radius"] = subject_smooth_radius
        if subject_morph_radius is not None:
            out["morph_radius"] = subject_morph_radius
        if subject_boundary_sigma is not None:
            out["boundary_sigma"] = subject_boundary_sigma
    elif zone == "background":
        if background_min_region_area is not None:
            out["min_region_area"] = background_min_region_area
        if background_max_regions is not None:
            out["max_regions"] = background_max_regions
        if background_smooth_radius is not None:
            out["smooth_radius"] = background_smooth_radius
        if background_morph_radius is not None:
            out["morph_radius"] = background_morph_radius
        if background_boundary_sigma is not None:
            out["boundary_sigma"] = background_boundary_sigma
    return out


@dataclass(frozen=True)
class ColourByNumbersResult:
    """All artefacts produced for one source image."""

    source: Image.Image
    quantized: QuantizedImage
    page: OutlinePage
    printable: Image.Image
    source_hit: ImageHit | None = None
    complexity: str = "fine"
    prepared: Image.Image | None = None
    subject_mask: SubjectMask | None = None
    subject_mode: str = "dual"
    subject_complexity: str | None = None
    background_complexity: str | None = None
    print_dpi: float | None = None
    firm_border: bool = True

    def save(self, output_dir: str | Path, *, stem: str = "colour_by_numbers") -> dict[str, Path]:
        """Write outline, legend, preview, and composite page to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "source": out / f"{stem}_source.png",
            "quantized": out / f"{stem}_quantized.png",
            "outline": out / f"{stem}_outline.png",
            "legend": out / f"{stem}_legend.png",
            "page": out / f"{stem}_page.png",
        }
        self.source.save(paths["source"])
        self.quantized.preview.save(paths["quantized"])
        self.page.outline.save(paths["outline"])
        self.page.legend.save(paths["legend"])
        self.printable.save(paths["page"])
        if self.prepared is not None and self.subject_mode not in {"off", "none"}:
            prepared_path = out / f"{stem}_prepared.png"
            self.prepared.save(prepared_path)
            paths["prepared"] = prepared_path
        return paths


def create_colour_by_numbers(
    image: Image.Image,
    *,
    n_colours: int = 16,
    max_size: int = 900,
    complexity: str = "fine",
    subject_mode: str = "dual",
    subject_model: str = "u2net",
    subject_autocrop: bool = True,
    subject_fill: float = DEFAULT_SUBJECT_FILL,
    subject_complexity: str = "fine",
    background_complexity: str = "light",
    firm_border: bool = True,
    min_a4_dpi: float | None = None,
    subject_blur_radius: float | None = None,
    background_blur_radius: float | None = None,
    min_region_area: int | None = None,
    max_regions: int | None = None,
    subject_min_region_area: int | None = None,
    subject_max_regions: int | None = None,
    background_min_region_area: int | None = None,
    background_max_regions: int | None = None,
    blur_radius: float | None = None,
    structure_size: int | None = None,
    smooth_radius: int | None = None,
    morph_radius: int | None = None,
    boundary_sigma: float | None = None,
    subject_smooth_radius: int | None = None,
    subject_morph_radius: int | None = None,
    subject_boundary_sigma: float | None = None,
    background_smooth_radius: int | None = None,
    background_morph_radius: int | None = None,
    background_boundary_sigma: float | None = None,
    line_width: int | None = None,
    seed: int = 42,
    source_hit: ImageHit | None = None,
) -> ColourByNumbersResult:
    """Quantize an image, simplify regions, and produce a numbered outline page.

    Default ``subject_mode='dual'``:
      1. rembg subject mask (firm binary by default)
      2. crop full-resolution source so the subject fills ``subject_fill``
      3. reject plates below ``min_a4_dpi`` when printed to A4
      4. shared palette of at most ``n_colours`` (default 16)
      5. ``subject_complexity`` on the subject, ``background_complexity`` on bg
    """
    if complexity not in COMPLEXITY_PRESETS:
        raise ValueError(
            f"Unknown complexity {complexity!r}; "
            f"choose one of {sorted(COMPLEXITY_PRESETS)}"
        )
    for name in (subject_complexity, background_complexity):
        if name not in COMPLEXITY_PRESETS:
            raise ValueError(f"Unknown complexity {name!r}")

    mode = subject_mode.lower().strip()
    subject_preset = COMPLEXITY_PRESETS[subject_complexity]
    background_preset = COMPLEXITY_PRESETS[background_complexity]
    uniform_preset = COMPLEXITY_PRESETS[complexity]

    source_rgb = image.convert("RGB")

    # Crop / isolate from the full-resolution source so native print DPI is kept.
    prepared_native, subject_mask = prepare_subject_image(
        source_rgb,
        mode=mode,
        model_name=subject_model,
        autocrop=subject_autocrop,
        subject_fill=subject_fill,
        firm_border=firm_border,
    )
    firm_mask = harden_mask(subject_mask) if subject_mask is not None else None

    print_dpi: float | None = None
    if min_a4_dpi is not None and min_a4_dpi > 0:
        report = evaluate_print_resolution(
            prepared_native.width,
            prepared_native.height,
            min_dpi=min_a4_dpi,
        )
        print_dpi = report.effective_dpi
        if not report.adequate:
            raise ValueError(
                f"Subject plate {prepared_native.width}×{prepared_native.height}px "
                f"is only ~{report.effective_dpi:.0f} DPI on A4; need ≥{min_a4_dpi:.0f} DPI. "
                "Choose a higher-resolution source, or lower the A4 DPI filter."
            )
    else:
        print_dpi = evaluate_print_resolution(
            prepared_native.width, prepared_native.height, min_dpi=150.0
        ).effective_dpi

    # Downscale for colour processing only after the native A4 check.
    prepared = resize_for_processing(prepared_native, max_size=max_size)
    if subject_mask is not None:
        subject_mask = align_mask(subject_mask, prepared.size, firm=firm_border)
    if firm_mask is not None:
        firm_mask = align_mask(firm_mask, prepared.size, firm=True)

    use_dual = mode in {"dual", "hybrid", "split"} and subject_mask is not None

    if use_dual:
        subj_blur = (
            float(subject_preset["blur_radius"])
            if subject_blur_radius is None
            else float(subject_blur_radius)
        )
        bg_blur = (
            float(background_preset["blur_radius"])
            if background_blur_radius is None
            else float(background_blur_radius)
        )
        subject_blurred = prefilter_for_regions(prepared, blur_radius=subj_blur)
        background_blurred = prefilter_for_regions(prepared, blur_radius=bg_blur)
        border_mask = (
            firm_mask if (firm_border and firm_mask is not None) else subject_mask
        )
        quant_input = blend_subject_background(
            subject_blurred,
            background_blurred,
            border_mask,
            firm_border=firm_border,
        )
        struct = int(
            structure_size
            if structure_size is not None
            else min(
                int(subject_preset["structure_size"]),
                int(background_preset["structure_size"]),
                max_size,
            )
        )
        blur = 0.0
        stroke = int(
            line_width
            if line_width is not None
            else max(
                int(subject_preset["line_width"]),
                int(background_preset["line_width"]),
            )
        )
    else:
        preset = uniform_preset
        blur = float(preset["blur_radius"] if blur_radius is None else blur_radius)
        struct = int(
            preset["structure_size"] if structure_size is None else structure_size
        )
        struct = min(struct, max_size)
        stroke = int(preset["line_width"] if line_width is None else line_width)
        quant_input = prepared

    quantized = quantize_colours(
        quant_input,
        n_colours=min(int(n_colours), 16),
        max_size=struct,
        structure_size=struct,
        blur_radius=blur,
        seed=seed,
    )

    height, width = quantized.labels.shape
    used_subject_complexity: str | None = None
    used_background_complexity: str | None = None

    if use_dual:
        active_mask = (
            firm_mask if (firm_border and firm_mask is not None) else subject_mask
        )
        assert active_mask is not None
        resample = (
            Image.Resampling.NEAREST if firm_border else Image.Resampling.BILINEAR
        )
        mask_img = Image.fromarray(active_mask.alpha, mode="L").resize(
            (width, height), resample
        )
        mask_bool = np.asarray(mask_img, dtype=np.uint8) >= 128
        override_kwargs = dict(
            min_region_area=min_region_area,
            max_regions=max_regions,
            smooth_radius=smooth_radius,
            morph_radius=morph_radius,
            boundary_sigma=boundary_sigma,
            subject_min_region_area=subject_min_region_area,
            subject_max_regions=subject_max_regions,
            subject_smooth_radius=subject_smooth_radius,
            subject_morph_radius=subject_morph_radius,
            subject_boundary_sigma=subject_boundary_sigma,
            background_min_region_area=background_min_region_area,
            background_max_regions=background_max_regions,
            background_smooth_radius=background_smooth_radius,
            background_morph_radius=background_morph_radius,
            background_boundary_sigma=background_boundary_sigma,
        )
        subject_params = _override_simplify_params(
            _preset_simplify_params(subject_preset, width=width, height=height),
            zone="subject",
            **override_kwargs,
        )
        background_params = _override_simplify_params(
            _preset_simplify_params(background_preset, width=width, height=height),
            zone="background",
            **override_kwargs,
        )

        labels, palette, subj_stats, _bg_stats = simplify_dual(
            quantized.labels,
            quantized.palette,
            mask_bool,
            subject_params=subject_params,
            background_params=background_params,
            firm_border=firm_border,
        )
        labels = upsample_labels(labels, prepared.size)
        page = build_outline_page(
            labels,
            palette,
            line_width=stroke,
            simplify=False,
        )
        from .simplify import SimplificationStats

        page = OutlinePage(
            outline=page.outline,
            legend=page.legend,
            regions=page.regions,
            palette=page.palette,
            colour_numbers=page.colour_numbers,
            labels=page.labels,
            simplification=SimplificationStats(
                regions_before=subj_stats.regions_before,
                regions_after=count_regions(page.labels),
                min_region_area=subj_stats.min_region_area,
                smooth_radius=subj_stats.smooth_radius,
                passes=subj_stats.passes,
            ),
        )
        complexity_label = f"{subject_complexity}+{background_complexity}"
        used_subject_complexity = subject_complexity
        used_background_complexity = background_complexity
    else:
        preset = uniform_preset
        smooth = int(preset["smooth_radius"] if smooth_radius is None else smooth_radius)
        morph = int(preset["morph_radius"] if morph_radius is None else morph_radius)
        boundary = float(
            preset["boundary_sigma"] if boundary_sigma is None else boundary_sigma
        )
        region_cap = int(preset["max_regions"] if max_regions is None else max_regions)
        do_simplify = bool(int(preset.get("simplify", 1)))
        area = min_region_area
        if area is None:
            fraction = float(preset["min_area_fraction"])
            area = 1 if fraction <= 0 else max(20, int(width * height * fraction))
        page = build_outline_page(
            quantized.labels,
            quantized.palette,
            min_region_area=area,
            max_regions=region_cap,
            line_width=stroke,
            smooth_radius=smooth,
            morph_radius=morph,
            boundary_sigma=boundary,
            output_size=prepared.size,
            simplify=do_simplify,
        )
        complexity_label = complexity

    prepared_out = prepared_native if mode not in {"off", "none"} else None

    simplified_preview = preview_from_labels(page.labels, page.palette)
    quantized = QuantizedImage(
        labels=page.labels,
        palette=page.palette,
        preview=simplified_preview,
    )
    printable = composite_page(page.outline, page.legend)
    return ColourByNumbersResult(
        source=source_rgb,
        quantized=quantized,
        page=page,
        printable=printable,
        source_hit=source_hit,
        complexity=complexity_label,
        prepared=prepared_out,
        subject_mask=subject_mask,
        subject_mode=mode,
        subject_complexity=used_subject_complexity,
        background_complexity=used_background_complexity,
        print_dpi=print_dpi,
        firm_border=firm_border,
    )


def create_from_query(
    query: str,
    *,
    n_colours: int = 16,
    max_size: int = 900,
    pick: int = 0,
    max_results: int = 8,
    min_a4_dpi: float | None = 150.0,
    **kwargs,
) -> ColourByNumbersResult:
    """Search the web for ``query`` and convert a result to colour-by-numbers."""
    image, hit = search_and_download(
        query,
        max_results=max_results,
        pick=pick,
        min_a4_dpi=min_a4_dpi,
    )
    return create_colour_by_numbers(
        image,
        n_colours=n_colours,
        max_size=max_size,
        source_hit=hit,
        min_a4_dpi=min_a4_dpi,
        **kwargs,
    )


def create_from_path(
    path: str | Path,
    *,
    n_colours: int = 16,
    max_size: int = 900,
    **kwargs,
) -> ColourByNumbersResult:
    """Load a local image and convert it to colour-by-numbers."""
    image = load_local_image(str(path))
    return create_colour_by_numbers(
        image,
        n_colours=n_colours,
        max_size=max_size,
        **kwargs,
    )
