"""High-level orchestration for colour-by-numbers generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .outline import OutlinePage, build_outline_page, composite_page
from .quantize import QuantizedImage, quantize_colours
from .search import ImageHit, load_local_image, search_and_download


@dataclass(frozen=True)
class ColourByNumbersResult:
    """All artefacts produced for one source image."""

    source: Image.Image
    quantized: QuantizedImage
    page: OutlinePage
    printable: Image.Image
    source_hit: ImageHit | None = None

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
    min_region_area: int | None = None,
    line_width: int = 1,
    seed: int = 42,
    source_hit: ImageHit | None = None,
) -> ColourByNumbersResult:
    """Quantize an image and produce a numbered outline page."""
    quantized = quantize_colours(
        image,
        n_colours=n_colours,
        max_size=max_size,
        seed=seed,
    )
    page = build_outline_page(
        quantized.labels,
        quantized.palette,
        min_region_area=min_region_area,
        line_width=line_width,
    )
    printable = composite_page(page.outline, page.legend)
    return ColourByNumbersResult(
        source=image.convert("RGB"),
        quantized=quantized,
        page=page,
        printable=printable,
        source_hit=source_hit,
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
