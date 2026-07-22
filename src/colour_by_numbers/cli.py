"""Command-line interface for colour-by-numbers generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import create_from_path, create_from_query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colour-by-numbers",
        description=(
            "Search the web for an image (or load a local file), reduce it to "
            "a limited palette, simplify regions, and export a numbered outline "
            "for colouring books."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-q",
        "--query",
        help="Descriptive search terms, e.g. 'aircraft' or 'dogs'",
    )
    source.add_argument(
        "-i",
        "--input",
        help="Path to a local image instead of searching the web",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output",
        help="Directory for generated files (default: output)",
    )
    parser.add_argument(
        "-n",
        "--colours",
        type=int,
        default=32,
        help="Number of colours from the palette (default: 32)",
    )
    parser.add_argument(
        "--palette-mode",
        choices=["standard", "free"],
        default="standard",
        help="standard = fixed 32-colour set; free = adaptive median-cut",
    )
    parser.add_argument(
        "--min-adjacent-delta-e",
        type=float,
        default=18.0,
        help="Merge touching sections closer than this Lab ΔE (default: 18)",
    )
    parser.add_argument(
        "--colour-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Snap subject mask using colour contrast (default: on)",
    )
    parser.add_argument(
        "--min-subject-bg-contrast",
        type=float,
        default=None,
        help="Reject plates below this subject/background ΔE (optional)",
    )
    parser.add_argument(
        "--no-contrast-bias",
        action="store_true",
        help="Do not bias web search toward high-contrast subject photos",
    )
    parser.add_argument(
        "--complexity",
        choices=["raw", "fine", "light", "medium", "simple", "detailed", "balanced"],
        default="fine",
        help="Region complexity for off/isolate modes (default: fine).",
    )
    parser.add_argument(
        "--subject",
        choices=["dual", "isolate", "off"],
        default="dual",
        help=(
            "Subject engine (default: dual). dual = mask + 80%% fill crop + "
            "fine on subject / light on background; isolate = flat background; "
            "off = full frame."
        ),
    )
    parser.add_argument(
        "--subject-fill",
        type=float,
        default=0.80,
        help="Target subject bbox fill of the frame after crop (default: 0.80)",
    )
    parser.add_argument(
        "--subject-complexity",
        default="fine",
        help="Complexity preset for subject pixels in dual mode (default: fine)",
    )
    parser.add_argument(
        "--background-complexity",
        default="light",
        help="Complexity preset for background pixels in dual mode (default: light)",
    )
    parser.add_argument(
        "--subject-model",
        default="u2net",
        help="rembg model name (default: u2net)",
    )
    parser.add_argument(
        "--no-subject-crop",
        action="store_true",
        help="Keep full frame after subject isolation (no autocrop)",
    )
    parser.add_argument(
        "--firm-border",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a hard binary subject mask for firm silhouette borders (default: on)",
    )
    parser.add_argument(
        "--min-a4-dpi",
        type=float,
        default=150.0,
        help="Reject plates below this effective DPI when printed to A4 (default: 150)",
    )
    parser.add_argument(
        "--no-a4-filter",
        action="store_true",
        help="Disable the A4 print-resolution filter",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=900,
        help="Longest edge in pixels for colour processing (default: 900)",
    )
    parser.add_argument(
        "--structure-size",
        type=int,
        default=None,
        help="Override structure/quantize canvas size",
    )
    parser.add_argument(
        "--pick",
        type=int,
        default=0,
        help="Which search result to use, 0-based (default: 0)",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=None,
        help="Outline stroke thickness in pixels (default: from complexity preset)",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=None,
        help="Hard cap on connected regions after simplification",
    )
    parser.add_argument(
        "--min-region-area",
        type=int,
        default=None,
        help="Minimum region area in pixels",
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=None,
        help="Uniform prefilter blur radius",
    )
    parser.add_argument(
        "--subject-blur-radius",
        type=float,
        default=None,
        help="Subject-zone prefilter blur (dual mode)",
    )
    parser.add_argument(
        "--background-blur-radius",
        type=float,
        default=None,
        help="Background-zone prefilter blur (dual mode)",
    )
    parser.add_argument(
        "--smooth-radius",
        type=int,
        default=None,
        help="Region smooth radius override",
    )
    parser.add_argument(
        "--morph-radius",
        type=int,
        default=None,
        help="Morphology radius override",
    )
    parser.add_argument(
        "--boundary-sigma",
        type=float,
        default=None,
        help="Boundary smoothing sigma override",
    )
    parser.add_argument(
        "--stem",
        default="colour_by_numbers",
        help="Filename prefix for outputs",
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="Discover and list specific types for a broad --query, then exit",
    )
    parser.add_argument(
        "--type",
        dest="subject_type",
        default=None,
        help="Concrete type within a category, e.g. 'golden retriever' or 'pug'",
    )
    parser.add_argument(
        "--type-pick",
        type=int,
        default=0,
        help="Which discovered type to use, 0-based (default: 0 = top ranked)",
    )
    parser.add_argument(
        "--no-discover",
        action="store_true",
        help="Skip type discovery; search the raw --query string",
    )
    parser.add_argument(
        "--illustrate",
        action="store_true",
        help=(
            "Illustration-first mode: discover type, gather references, "
            "build a flat colouring-book illustration, then convert"
        ),
    )
    parser.add_argument(
        "--illustration-backend",
        choices=["local_stylize", "openai", "replicate"],
        default="local_stylize",
        help="Illustration generator backend (default: local_stylize)",
    )
    parser.add_argument(
        "--illustration-size",
        type=int,
        default=1600,
        help="Longest edge of the generated illustration (default: 1600)",
    )
    parser.add_argument(
        "--illustration-colours",
        type=int,
        default=16,
        help="Flat fills used while stylising the illustration (default: 16)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output)
    min_a4 = None if args.no_a4_filter else args.min_a4_dpi

    if args.list_types:
        if not args.query:
            print("--list-types requires --query", file=sys.stderr)
            return 2
        from .discover import discover_subject_types

        discovery = discover_subject_types(args.query, probe_search=True)
        print(f"Query: {discovery.original_query}")
        print(f"Category: {discovery.category or '—'}")
        print(f"Reason: {discovery.reason}")
        for i, item in enumerate(discovery.types):
            ev = f" · e.g. {item.evidence[0]}" if item.evidence else ""
            print(
                f"  {i}. {item.label}  (score {item.score:.2f}) "
                f"→ “{item.search_query}”{ev}"
            )
        return 0

    common = dict(
        n_colours=args.colours,
        palette_mode=args.palette_mode,
        min_adjacent_delta_e=args.min_adjacent_delta_e,
        colour_refine=args.colour_refine,
        min_subject_bg_contrast=args.min_subject_bg_contrast,
        max_size=args.max_size,
        complexity=args.complexity,
        subject_mode=args.subject,
        subject_model=args.subject_model,
        subject_autocrop=not args.no_subject_crop,
        subject_fill=args.subject_fill,
        subject_complexity=args.subject_complexity,
        background_complexity=args.background_complexity,
        firm_border=args.firm_border,
        min_a4_dpi=min_a4,
        structure_size=args.structure_size,
        line_width=args.line_width,
        max_regions=args.max_regions,
        min_region_area=args.min_region_area,
        blur_radius=args.blur_radius,
        subject_blur_radius=args.subject_blur_radius,
        background_blur_radius=args.background_blur_radius,
        smooth_radius=args.smooth_radius,
        morph_radius=args.morph_radius,
        boundary_sigma=args.boundary_sigma,
    )

    if args.illustrate:
        if not args.query:
            print("--illustrate requires --query", file=sys.stderr)
            return 2
        from .generate import generate_colouring_page

        page = generate_colouring_page(
            args.query,
            subject_type=args.subject_type,
            type_pick=args.type_pick,
            discover_types=not args.no_discover,
            backend=args.illustration_backend,
            illustration_colours=args.illustration_colours,
            illustration_size=args.illustration_size,
            n_colours=args.colours,
            complexity=args.complexity,
            subject_mode="off",
            palette_mode=args.palette_mode,
            min_adjacent_delta_e=args.min_adjacent_delta_e,
            firm_border=args.firm_border,
            max_size=args.max_size,
            line_width=args.line_width,
            max_regions=args.max_regions,
            min_region_area=args.min_region_area,
            structure_size=args.structure_size,
        )
        result = page.result
        stem_base = page.subject_type.label
        stem = args.stem if args.stem != "colour_by_numbers" else (
            stem_base.strip().lower().replace(" ", "_")[:40] or args.stem
        )
        paths = result.save(output_dir, stem=stem)
        illustration_path = output_dir / f"{stem}_illustration.png"
        page.illustration.image.save(illustration_path)
        paths["illustration"] = illustration_path
        print(f"Illustration backend: {page.illustration.backend}")
        print(f"Subject type: {page.subject_type.label}")
        if page.illustration.prompt:
            print(f"Prompt: {page.illustration.prompt}")
        if page.illustration.reference_url:
            print(f"Reference: {page.illustration.reference_url}")
        print(f"Complexity: {result.complexity}")
        print(f"Palette colours: {result.quantized.n_colours}")
        print(f"Numbered regions: {len(result.page.regions)}")
        print("Wrote:")
        for label, path in paths.items():
            print(f"  {label:12s} {path}")
        return 0

    if args.query:
        result = create_from_query(
            args.query,
            pick=args.pick,
            contrast_bias=not args.no_contrast_bias,
            discover_types=not args.no_discover,
            subject_type=args.subject_type,
            type_pick=args.type_pick,
            **common,
        )
        if result.subject_type_label:
            stem_base = result.subject_type_label
        else:
            stem_base = args.query
        stem = args.stem if args.stem != "colour_by_numbers" else (
            stem_base.strip().lower().replace(" ", "_")[:40] or args.stem
        )
    else:
        result = create_from_path(args.input, **common)
        stem = args.stem if args.stem != "colour_by_numbers" else Path(args.input).stem

    paths = result.save(output_dir, stem=stem)
    print(f"Complexity: {result.complexity}")
    print(f"Subject mode: {result.subject_mode}")
    if result.subject_type_label:
        print(f"Subject type: {result.subject_type_label}")
        if result.subject_type_query:
            print(f"Type search: {result.subject_type_query}")
    print(f"Firm border: {result.firm_border}")
    if result.print_dpi is not None:
        print(f"Effective A4 DPI: {result.print_dpi:.1f}")
    if result.subject_complexity and result.background_complexity:
        print(
            f"Dual pass: subject={result.subject_complexity} "
            f"background={result.background_complexity}"
        )
    if result.subject_mask is not None:
        print(
            f"Subject mask: {result.subject_mask.model} "
            f"({100 * result.subject_mask.foreground_fraction:.1f}% foreground)"
        )
    print(f"Palette mode: {result.palette_mode}")
    print(f"Palette colours: {result.quantized.n_colours}")
    if result.subject_bg_contrast is not None:
        print(f"Subject/bg contrast ΔE: {result.subject_bg_contrast:.1f}")
    print(f"Numbered regions: {len(result.page.regions)}")
    if result.page.simplification is not None:
        stats = result.page.simplification
        print(
            f"Regions simplified: {stats.regions_before} → {stats.regions_after}"
            f" (min area {stats.min_region_area}px)"
        )
    if result.source_hit:
        print(f"Source: {result.source_hit.url}")
        if result.source_hit.title:
            print(f"Title: {result.source_hit.title}")
        if result.source_hit.provider:
            print(f"Provider: {result.source_hit.provider}")
        if result.source_hit.license:
            print(f"Licence: {result.source_hit.license}")
    print("Wrote:")
    for label, path in paths.items():
        print(f"  {label:10s} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
