"""High-level orchestration for colour-by-numbers generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .outline import OutlinePage, build_outline_page, composite_page
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
    min_region_area: int | None = None,
    max_regions: int | None = None,
    blur_radius: float | None = None,
    structure_size: int | None = None,
    smooth_radius: int | None = None,
    morph_radius: int | None = None,
    line_width: int | None = None,
    seed: int = 42,
    source_hit: ImageHit | None = None,
) -> ColourByNumbersResult:
    """Quantize an image, simplify regions, and produce a numbered outline page.

    Default ``subject_mode='dual'``:
      1. rembg subject mask
      2. crop so the subject fills ``subject_fill`` of the frame (default 80%)
      3. shared palette of at most ``n_colours`` (default 16)
      4. ``subject_complexity`` (fine) on the subject, ``background_complexity``
         (light) on the background
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
    working = resize_for_processing(source_rgb, max_size=max_size)
    prepared, subject_mask = prepare_subject_image(
        working,
        mode=mode,
        model_name=subject_model,
        autocrop=subject_autocrop,
        subject_fill=subject_fill,
    )
    # After an 80% fill crop the plate can be small (tiny subjects). Scale it
    # back up to the working canvas size so outlines stay printable.
    if (
        subject_autocrop
        and mode not in {"off", "none"}
        and max(prepared.size) < max(working.size) * 0.85
    ):
        prepared = prepared.resize(working.size, Image.Resampling.LANCZOS)
        if subject_mask is not None:
            subject_mask = align_mask(subject_mask, prepared.size)
            subject_mask = SubjectMask(
                alpha=subject_mask.alpha,
                model=subject_mask.model,
                foreground_fraction=float((subject_mask.alpha >= 128).mean()),
            )

    use_dual = mode in {"dual", "hybrid", "split"} and subject_mask is not None

    if use_dual:
        # Differential blur: gentler on subject, stronger on background.
        subject_blurred = prefilter_for_regions(
            prepared, blur_radius=float(subject_preset["blur_radius"])
        )
        background_blurred = prefilter_for_regions(
            prepared, blur_radius=float(background_preset["blur_radius"])
        )
        quant_input = blend_subject_background(
            subject_blurred, background_blurred, subject_mask
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
        blur = 0.0  # already applied differentially
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
        mask_img = Image.fromarray(subject_mask.alpha, mode="L").resize(
            (width, height), Image.Resampling.BILINEAR
        )
        mask_bool = np.asarray(mask_img, dtype=np.uint8) >= 128
        subject_params = _preset_simplify_params(
            subject_preset, width=width, height=height
        )
        background_params = _preset_simplify_params(
            background_preset, width=width, height=height
        )
        if min_region_area is not None:
            subject_params["min_region_area"] = min_region_area
        if max_regions is not None:
            subject_params["max_regions"] = max_regions
            background_params["max_regions"] = max_regions
        if smooth_radius is not None:
            subject_params["smooth_radius"] = smooth_radius
        if morph_radius is not None:
            subject_params["morph_radius"] = morph_radius

        labels, palette, subj_stats, _bg_stats = simplify_dual(
            quantized.labels,
            quantized.palette,
            mask_bool,
            subject_params=subject_params,
            background_params=background_params,
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
        boundary = float(preset["boundary_sigma"])
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
        prepared=prepared if mode not in {"off", "none"} else None,
        subject_mask=subject_mask,
        subject_mode=mode,
        subject_complexity=used_subject_complexity,
        background_complexity=used_background_complexity,
    )


def create_from_query(
    query: str,
    *,
    n_colours: int = 16,
    max_size: int = 900,
    pick: int = 0,
    max_results: int = 8,
    **kwargs,
) -> ColourByNumbersResult:
    """Search the web for ``query`` and convert a result to colour-by-numbers."""
    image, hit = search_and_download(query, max_results=max_results, pick=pick)
    return create_colour_by_numbers(
        image,
        n_colours=n_colours,
        max_size=max_size,
        source_hit=hit,
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
