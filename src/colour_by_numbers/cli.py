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
            "a limited palette, and export a numbered outline for colouring books."
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
        default=16,
        help="Number of colours / palette size (default: 16)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=900,
        help="Longest edge in pixels before processing (default: 900)",
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
        default=1,
        help="Outline stroke thickness in pixels (default: 1)",
    )
    parser.add_argument(
        "--stem",
        default="colour_by_numbers",
        help="Filename prefix for outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output)

    if args.query:
        result = create_from_query(
            args.query,
            n_colours=args.colours,
            max_size=args.max_size,
            pick=args.pick,
            line_width=args.line_width,
        )
        stem = args.stem if args.stem != "colour_by_numbers" else (
            args.query.strip().lower().replace(" ", "_")[:40] or args.stem
        )
    else:
        result = create_from_path(
            args.input,
            n_colours=args.colours,
            max_size=args.max_size,
            line_width=args.line_width,
        )
        stem = args.stem if args.stem != "colour_by_numbers" else Path(args.input).stem

    paths = result.save(output_dir, stem=stem)
    print(f"Palette colours: {result.quantized.n_colours}")
    print(f"Numbered regions: {len(result.page.regions)}")
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
