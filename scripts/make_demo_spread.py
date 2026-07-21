#!/usr/bin/env python3
"""Generate demo spreads: original + 16-colour simplification settings."""

from __future__ import annotations

import argparse
from pathlib import Path

from colour_by_numbers.demo_spread import write_demo_spreads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aircraft",
        default="output/aircraft_source.png",
        help="Path to the aircraft source photo",
    )
    parser.add_argument(
        "--dogs",
        default="output/dogs_source.png",
        help="Path to the dogs source photo",
    )
    parser.add_argument(
        "--output",
        default="output/demo",
        help="Directory for spread images",
    )
    parser.add_argument("--colours", type=int, default=16)
    parser.add_argument("--max-size", type=int, default=700)
    parser.add_argument("--complexity", default="fine")
    args = parser.parse_args()

    sources = {
        "aircraft": Path(args.aircraft),
        "dogs": Path(args.dogs),
    }
    missing = [name for name, path in sources.items() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing source images: {missing}")

    written = write_demo_spreads(
        sources,
        args.output,
        n_colours=args.colours,
        max_size=args.max_size,
        complexity=args.complexity,
    )
    print("Wrote:")
    for label, path in written.items():
        print(f"  {label:20s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
