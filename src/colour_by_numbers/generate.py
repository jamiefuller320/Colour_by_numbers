"""End-to-end illustration-first colouring-page generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from PIL import Image

from .contrast import estimate_centre_border_contrast
from .discover import (
    SubjectType,
    discover_subject_types,
    pick_subject_type,
    search_images_for_type,
)
from .illustrate import (
    DEFAULT_ILLUSTRATION_SIZE,
    IllustrationResult,
    generate_illustration,
)
from .pipeline import ColourByNumbersResult, create_colour_by_numbers
from .print_resolution import evaluate_print_resolution
from .search import ImageHit, download_image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratedPage:
    """Illustration-first run: illustration + colour-by-numbers artefacts."""

    illustration: IllustrationResult
    result: ColourByNumbersResult
    subject_type: SubjectType
    reference_hit: ImageHit | None = None


def gather_reference_hits(
    subject_type: SubjectType,
    *,
    max_results: int = 6,
    min_a4_dpi: float | None = 120.0,
) -> list[ImageHit]:
    """Fetch candidate reference photos for a concrete subject type."""
    return search_images_for_type(
        subject_type,
        max_results=max_results,
        min_a4_dpi=min_a4_dpi,
        contrast_bias=True,
    )


def select_best_reference(
    hits: list[ImageHit],
    *,
    min_a4_dpi: float | None = 120.0,
    min_contrast: float = 18.0,
) -> tuple[Image.Image, ImageHit, float]:
    """Download candidates and pick the strongest contrast / resolution ref."""
    if not hits:
        raise RuntimeError("No reference hits to select from")

    errors: list[str] = []
    scored: list[tuple[float, Image.Image, ImageHit]] = []
    for hit in hits:
        try:
            image = download_image(hit.url)
            if min_a4_dpi is not None and min_a4_dpi > 0:
                report = evaluate_print_resolution(
                    image.width, image.height, min_dpi=min_a4_dpi
                )
                if not report.adequate:
                    errors.append(
                        f"{hit.url}: ~{report.effective_dpi:.0f} DPI below {min_a4_dpi}"
                    )
                    continue
            contrast = estimate_centre_border_contrast(image)
            if contrast < min_contrast:
                errors.append(f"{hit.url}: contrast ΔE≈{contrast:.1f}")
                continue
            # Prefer both contrast and megapixels.
            score = contrast + 0.002 * (image.width * image.height) ** 0.5
            scored.append((score, image, hit))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{hit.url}: {exc}")
            logger.warning("Reference download failed: %s", exc)

    if not scored:
        raise RuntimeError(
            "Could not download a suitable reference photo.\n" + "\n".join(errors)
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    score, image, hit = scored[0]
    logger.info(
        "Selected reference ΔE-score≈%.1f (%sx%s) %s",
        score,
        image.width,
        image.height,
        hit.url,
    )
    return image, hit, score


def generate_colouring_page(
    query: str,
    *,
    subject_type: str | None = None,
    type_pick: int = 0,
    discover_types: bool = True,
    backend: str = "local_stylize",
    n_colours: int = 16,
    illustration_colours: int = 16,
    illustration_size: int = DEFAULT_ILLUSTRATION_SIZE,
    max_references: int = 6,
    complexity: str = "fine",
    subject_mode: str = "off",
    min_a4_dpi: float | None = None,
    openai_api_key: str | None = None,
    **pipeline_kwargs,
) -> GeneratedPage:
    """Discover type → gather references → illustrate → colour-by-numbers.

    Default ``subject_mode='off'`` because the illustration is already isolated
    on a flat background with ink outlines; dual rembg is usually unnecessary.
    """
    discovery = discover_subject_types(
        query,
        probe_search=discover_types and subject_type is None,
    )
    chosen = pick_subject_type(
        discovery, type_name=subject_type, pick=type_pick
    )

    reference_hit: ImageHit | None = None
    reference_image: Image.Image | None = None

    if backend == "local_stylize":
        hits = gather_reference_hits(
            chosen, max_results=max_references, min_a4_dpi=120.0
        )
        reference_image, reference_hit, _ = select_best_reference(hits)

    illustration = generate_illustration(
        reference_image,
        subject_type_label=chosen.label,
        category=chosen.category,
        backend=backend,
        n_colours=illustration_colours,
        output_size=illustration_size,
        openai_api_key=openai_api_key,
    )
    if reference_hit is not None:
        illustration = IllustrationResult(
            image=illustration.image,
            backend=illustration.backend,
            subject_type_label=illustration.subject_type_label,
            reference_url=reference_hit.url,
            reference_title=reference_hit.title,
            n_colours=illustration.n_colours,
            prompt=illustration.prompt,
            notes=illustration.notes,
        )

    # Illustrations are already flat; keep A4 filter off unless requested.
    result = create_colour_by_numbers(
        illustration.image,
        n_colours=n_colours,
        complexity=complexity,
        subject_mode=subject_mode,
        palette_mode="standard",
        firm_border=True,
        colour_refine=False,
        min_a4_dpi=min_a4_dpi,
        source_hit=reference_hit,
        **pipeline_kwargs,
    )
    result = ColourByNumbersResult(
        source=result.source,
        quantized=result.quantized,
        page=result.page,
        printable=result.printable,
        source_hit=result.source_hit,
        complexity=result.complexity,
        prepared=result.prepared,
        subject_mask=result.subject_mask,
        subject_mode=result.subject_mode,
        subject_complexity=result.subject_complexity,
        background_complexity=result.background_complexity,
        print_dpi=result.print_dpi,
        firm_border=result.firm_border,
        palette_mode=result.palette_mode,
        subject_bg_contrast=result.subject_bg_contrast,
        min_adjacent_delta_e=result.min_adjacent_delta_e,
        subject_type_label=chosen.label,
        subject_type_query=chosen.search_query,
    )
    return GeneratedPage(
        illustration=illustration,
        result=result,
        subject_type=chosen,
        reference_hit=reference_hit,
    )
