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


# Named complexity presets control cartoon prefilter + region absorption.
# Ordered from least → most aggressive for demo spreads.
COMPLEXITY_PRESETS: dict[str, dict[str, float | int]] = {
    # 16-colour quantize only — no region absorption.
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
    # Gentle cleanup — keeps most photographic structure.
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
    # Default balance of recognisable outline vs segment count.
    "balanced": {
        "blur_radius": 1.8,
        "structure_size": 420,
        "min_area_fraction": 0.008,
        "max_regions": 48,
        "smooth_radius": 2,
        "morph_radius": 1,
        "boundary_sigma": 0.9,
        "line_width": 2,
        "simplify": 1,
    },
    # Stronger merge — fewer, larger colouring regions.
    "simple": {
        "blur_radius": 2.8,
        "structure_size": 320,
        "min_area_fraction": 0.015,
        "max_regions": 28,
        "smooth_radius": 2,
        "morph_radius": 2,
        "boundary_sigma": 1.2,
        "line_width": 2,
        "simplify": 1,
    },
}

# Back-compat alias used in earlier docs/UI.
COMPLEXITY_PRESETS["detailed"] = dict(COMPLEXITY_PRESETS["light"])

DEMO_SPREAD_SETTINGS: tuple[str, ...] = ("raw", "light", "balanced", "simple")


@dataclass(frozen=True)
class ColourByNumbersResult:
    """All artefacts produced for one source image."""

    source: Image.Image
    quantized: QuantizedImage
    page: OutlinePage
    printable: Image.Image
    source_hit: ImageHit | None = None
    complexity: str = "balanced"

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
        return paths


def create_colour_by_numbers(
    image: Image.Image,
    *,
    n_colours: int = 16,
    max_size: int = 900,
    complexity: str = "balanced",
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
    """Quantize an image, simplify regions, and produce a numbered outline page."""
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

    # Keep a high-res canvas for the final page, but quantize/simplify on a
    # smaller structure canvas so only large shapes survive.
    output_image = resize_for_processing(image.convert("RGB"), max_size=max_size)
    quantized = quantize_colours(
        output_image,
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
        output_size=output_image.size,
        simplify=do_simplify,
    )

    # Preview should match the simplified (upsampled) regions used for the outline.
    simplified_preview = preview_from_labels(page.labels, page.palette)
    quantized = QuantizedImage(
        labels=page.labels,
        palette=page.palette,
        preview=simplified_preview,
    )

    printable = composite_page(page.outline, page.legend)
    return ColourByNumbersResult(
        source=image.convert("RGB"),
        quantized=quantized,
        page=page,
        printable=printable,
        source_hit=source_hit,
        complexity=complexity,
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
