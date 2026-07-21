"""High-level orchestration for colour-by-numbers generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .outline import OutlinePage, build_outline_page, composite_page
from .quantize import (
    QuantizedImage,
    preview_from_labels,
    quantize_colours,
    resize_for_processing,
)
from .search import ImageHit, load_local_image, search_and_download
from .subject import SubjectMask, prepare_subject_image


# Named complexity presets control cartoon prefilter + region absorption.
# Primary ladder is centred on ``fine`` after subject isolation.
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

# Demo: original vs fine without/with subject isolation.
DEMO_SPREAD_SETTINGS: tuple[str, ...] = ("fine",)
DEMO_SUBJECT_COMPARE: tuple[str, ...] = ("off", "isolate")


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
    subject_mode: str = "isolate"

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
    subject_mode: str = "isolate",
    subject_model: str = "u2net",
    subject_autocrop: bool = True,
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

    By default, a neural subject mask (rembg / U²-Net) isolates the foreground
    onto a flat background and crops tightly around it before colour reduction.
    """
    if complexity not in COMPLEXITY_PRESETS:
        raise ValueError(
            f"Unknown complexity {complexity!r}; "
            f"choose one of {sorted(COMPLEXITY_PRESETS)}"
        )
    preset = COMPLEXITY_PRESETS[complexity]

    blur = float(preset["blur_radius"] if blur_radius is None else blur_radius)
    struct = int(preset["structure_size"] if structure_size is None else structure_size)
    struct = min(struct, max_size)
    smooth = int(preset["smooth_radius"] if smooth_radius is None else smooth_radius)
    morph = int(preset["morph_radius"] if morph_radius is None else morph_radius)
    boundary = float(preset["boundary_sigma"])
    stroke = int(preset["line_width"] if line_width is None else line_width)
    region_cap = int(preset["max_regions"] if max_regions is None else max_regions)
    do_simplify = bool(int(preset.get("simplify", 1)))

    source_rgb = image.convert("RGB")
    working = resize_for_processing(source_rgb, max_size=max_size)
    prepared, subject_mask = prepare_subject_image(
        working,
        mode=subject_mode,
        model_name=subject_model,
        autocrop=subject_autocrop,
    )

    quantized = quantize_colours(
        prepared,
        n_colours=n_colours,
        max_size=struct,
        structure_size=struct,
        blur_radius=blur,
        seed=seed,
    )

    height, width = quantized.labels.shape
    if min_region_area is None:
        fraction = float(preset["min_area_fraction"])
        min_region_area = (
            1 if fraction <= 0 else max(20, int(width * height * fraction))
        )

    page = build_outline_page(
        quantized.labels,
        quantized.palette,
        min_region_area=min_region_area,
        max_regions=region_cap,
        line_width=stroke,
        smooth_radius=smooth,
        morph_radius=morph,
        boundary_sigma=boundary,
        output_size=prepared.size,
        simplify=do_simplify,
    )

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
        complexity=complexity,
        prepared=prepared if subject_mode not in {"off", "none"} else None,
        subject_mask=subject_mask,
        subject_mode=subject_mode,
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
