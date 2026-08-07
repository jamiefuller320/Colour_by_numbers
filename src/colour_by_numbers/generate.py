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
from .feedback import FeedbackLoopResult, run_subject_feedback_loop
from .illustrate import (
    DEFAULT_ILLUSTRATION_SIZE,
    AVAILABLE_ILLUSTRATION_BACKENDS,
    IllustrationResult,
    generate_illustration,
    illustration_prompt,
    prepare_illustration_for_colouring,
)
from .palette import DEFAULT_ILLUSTRATION_COLOURS, MAX_N_COLOURS, clamp_n_colours
from .style_presets import DEFAULT_STYLE, resolve_style_preset
from .pipeline import ColourByNumbersResult, create_colour_by_numbers
from .print_resolution import evaluate_print_resolution
from .quality import (
    PHASE_B_MIN_REGION_MM,
    PHASE_B_PRIMARY_BACKEND,
    PlateQualityReport,
    assert_plate_quality,
    evaluate_plate_quality,
)
from .search import ImageHit, download_image

logger = logging.getLogger(__name__)

# Phase B lock: rights-safe text-to-image is the default publish path.
DEFAULT_ILLUSTRATION_BACKEND = PHASE_B_PRIMARY_BACKEND
assert DEFAULT_ILLUSTRATION_BACKEND in AVAILABLE_ILLUSTRATION_BACKENDS


@dataclass(frozen=True)
class GeneratedPage:
    """Illustration-first run: illustration + colour-by-numbers artefacts."""

    illustration: IllustrationResult
    result: ColourByNumbersResult
    subject_type: SubjectType
    reference_hit: ImageHit | None = None
    quality: PlateQualityReport | None = None
    feedback: FeedbackLoopResult | None = None


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
    backend: str = DEFAULT_ILLUSTRATION_BACKEND,
    n_colours: int | None = None,
    illustration_colours: int | None = None,
    illustration_size: int = DEFAULT_ILLUSTRATION_SIZE,
    max_references: int = 6,
    complexity: str | None = None,
    subject_mode: str = "off",
    min_a4_dpi: float | None = None,
    min_region_mm: float | None = None,
    min_adjacent_delta_e: float | None = None,
    style: str = DEFAULT_STYLE,
    openai_api_key: str | None = None,
    fal_api_key: str | None = None,
    pollinations_api_key: str | None = None,
    prompt_override: str | None = None,
    fal_model: str = "fal-ai/flux/schnell",
    pollinations_model: str = "flux",
    seed: int | None = None,
    check_quality: bool = True,
    require_quality: bool = False,
    subject_feedback: bool = False,
    critique_mode: str = "rules",
    max_feedback_attempts: int = 3,
    lessons_file: str | None = None,
    record_lessons: bool = True,
    **pipeline_kwargs,
) -> GeneratedPage:
    """Discover type → gather references → illustrate → colour-by-numbers.

    Default ``subject_mode='off'`` because the illustration is already isolated
    on a flat background with ink outlines; dual rembg is usually unnecessary.

    ``style`` selects a difficulty band (``simple`` / ``standard`` / ``vibrant``).
    ``standard`` is the Phase B kids gate; ``vibrant`` is the adult end-goal band
    (denser fills, up to 32 colours, cooler shadows, exact plate-colour
    preserve, and a high-region ``vibrant`` complexity pass).

    Phase B: default backend is ``fal`` (Flux via fal.ai; needs ``FAL_KEY``).
    When ``check_quality`` is True, attach a ``PlateQualityReport``. When
    ``require_quality`` is True, fail the run if the checklist does not pass.

    When ``subject_feedback`` is True (API backends only), run a critique →
    revise → retry loop that asks whether the plate is recognisable as the
    requested subject and how the prompt should improve. Lessons are stored
    for later runs of the same subject.
    """
    preset = resolve_style_preset(style)
    discovery = discover_subject_types(
        query,
        probe_search=discover_types and subject_type is None,
    )
    chosen = pick_subject_type(
        discovery, type_name=subject_type, pick=type_pick
    )

    reference_hit: ImageHit | None = None
    reference_image: Image.Image | None = None
    if illustration_colours is None:
        illustration_colours = preset.n_colours
    if n_colours is None:
        n_colours = preset.n_colours
    if min_region_mm is None:
        min_region_mm = preset.min_region_mm
    illustration_colours = clamp_n_colours(
        illustration_colours, maximum=preset.max_colours
    )
    n_colours = clamp_n_colours(n_colours, maximum=preset.max_colours)

    if backend == "local_stylize":
        hits = gather_reference_hits(
            chosen, max_results=max_references, min_a4_dpi=120.0
        )
        reference_image, reference_hit, _ = select_best_reference(hits)

    feedback_result: FeedbackLoopResult | None = None
    effective_prompt = prompt_override
    use_feedback = (
        subject_feedback
        and backend != "local_stylize"
        and max_feedback_attempts > 0
    )

    if use_feedback:
        base_prompt = prompt_override or illustration_prompt(
            chosen.label,
            category=chosen.category,
            style_preset=preset.name,
            min_region_mm=min_region_mm,
        )

        def _generate_once(prompt: str) -> Image.Image:
            # Skip colouring prep inside the loop; apply once on the winner.
            one = generate_illustration(
                None,
                subject_type_label=chosen.label,
                category=chosen.category,
                backend=backend,
                n_colours=illustration_colours,
                output_size=illustration_size,
                openai_api_key=openai_api_key,
                fal_api_key=fal_api_key,
                pollinations_api_key=pollinations_api_key,
                prompt_override=prompt,
                fal_model=fal_model,
                pollinations_model=pollinations_model,
                seed=seed,
                min_region_mm=min_region_mm,
                prepare_for_colouring=False,
                style=preset.name,
            )
            return one.image

        feedback_result = run_subject_feedback_loop(
            subject_label=chosen.label,
            category=chosen.category,
            initial_prompt=base_prompt,
            generate_fn=_generate_once,
            critique_mode=critique_mode,
            max_attempts=max_feedback_attempts,
            api_key=openai_api_key,
            lessons_file=lessons_file,
            record=record_lessons,
        )
        effective_prompt = feedback_result.prompt
        # Final plate: prepare the accepted (or last) image for colouring.
        cleaned, used = prepare_illustration_for_colouring(
            feedback_result.image,
            n_colours=illustration_colours,
            min_region_mm=min_region_mm,
            category=chosen.category,
            palette_mode=preset.palette_mode,
            cool_shadows=preset.cool_shadows,
            max_colours=preset.max_colours,
        )
        notes = feedback_result.notes
        illustration = IllustrationResult(
            image=cleaned,
            backend=backend,
            subject_type_label=chosen.label,
            n_colours=used,
            prompt=effective_prompt,
            notes=notes,
        )
    else:
        illustration = generate_illustration(
            reference_image,
            subject_type_label=chosen.label,
            category=chosen.category,
            backend=backend,
            n_colours=illustration_colours,
            output_size=illustration_size,
            openai_api_key=openai_api_key,
            fal_api_key=fal_api_key,
            pollinations_api_key=pollinations_api_key,
            prompt_override=effective_prompt,
            fal_model=fal_model,
            pollinations_model=pollinations_model,
            seed=seed,
            min_region_mm=min_region_mm,
            style=preset.name,
        )
    if reference_hit is not None:
        illustration = IllustrationResult(
            image=illustration.image,
            backend=illustration.backend,
            subject_type_label=illustration.subject_type_label or chosen.label,
            reference_url=reference_hit.url,
            reference_title=reference_hit.title,
            n_colours=illustration.n_colours,
            prompt=illustration.prompt,
            notes=illustration.notes,
        )
    elif illustration.subject_type_label is None:
        illustration = IllustrationResult(
            image=illustration.image,
            backend=illustration.backend,
            subject_type_label=chosen.label,
            reference_url=illustration.reference_url,
            reference_title=illustration.reference_title,
            n_colours=illustration.n_colours,
            prompt=illustration.prompt,
            notes=illustration.notes,
        )

    # Illustrations are already flat; keep A4 filter off unless requested.
    # Strip keys we set explicitly so CLI **kwargs cannot collide.
    for key in (
        "palette_mode",
        "palette_category",
        "firm_border",
        "colour_refine",
        "min_adjacent_delta_e",
        "complexity",
    ):
        pipeline_kwargs.pop(key, None)
    pipeline_kwargs.setdefault("min_region_mm", min_region_mm)
    # Illustrations often reuse fill colours in the abstract background; keep the
    # subject silhouette inked so colourists can still see the form.
    pipeline_kwargs.setdefault("silhouette_outline", True)
    resolved_complexity = complexity or preset.complexity
    resolved_delta_e = (
        float(min_adjacent_delta_e)
        if min_adjacent_delta_e is not None
        else float(preset.min_adjacent_delta_e)
    )
    if preset.pipeline_palette_mode:
        pipeline_palette = preset.pipeline_palette_mode
    elif preset.palette_mode in {"adaptive", "free", "exact", "preserve"}:
        pipeline_palette = (
            "exact" if preset.palette_mode in {"adaptive", "exact", "preserve"} else "free"
        )
    else:
        pipeline_palette = "standard"
    result = create_colour_by_numbers(
        illustration.image,
        n_colours=n_colours,
        complexity=resolved_complexity,
        subject_mode=subject_mode,
        palette_mode=pipeline_palette,
        palette_category=None if preset.cool_shadows else chosen.category,
        firm_border=True,
        colour_refine=False,
        min_a4_dpi=min_a4_dpi,
        min_adjacent_delta_e=resolved_delta_e,
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

    quality: PlateQualityReport | None = None
    if check_quality or require_quality:
        quality_kwargs = dict(
            colour_plate=illustration.image,
            min_region_mm=min_region_mm,
            min_colours=preset.min_colours,
            max_colours=preset.max_colours,
        )
        if require_quality:
            quality = assert_plate_quality(result, **quality_kwargs)
        else:
            quality = evaluate_plate_quality(result, **quality_kwargs)
            if not quality.passed:
                logger.warning("Plate quality gate:\n%s", quality.summary())

    return GeneratedPage(
        illustration=illustration,
        result=result,
        subject_type=chosen,
        reference_hit=reference_hit,
        quality=quality,
        feedback=feedback_result,
    )
