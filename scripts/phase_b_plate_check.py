"""Offline / live Phase B single-plate quality check.

Examples::

    # Offline synthetic plate (no network)
    python scripts/phase_b_plate_check.py --offline

    # Live Pollinations primary backend (needs network)
    python scripts/phase_b_plate_check.py --query dogs --type pug --require
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colour_by_numbers.pipeline import create_colour_by_numbers  # noqa: E402
from colour_by_numbers.quality import (  # noqa: E402
    PHASE_B_MIN_REGION_MM,
    PHASE_B_PRIMARY_BACKEND,
    PHASE_B_PRIMARY_MODEL,
    assert_plate_quality,
    evaluate_plate_quality,
)


def _synthetic_plate(size: int = 420) -> Image.Image:
    """Subject-heavy flat blocks that pass Phase B + C format-brief gates."""
    image = Image.new("RGB", (size, size), (240, 238, 232))
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 30, 250, 250), fill=(230, 170, 60))
    draw.rectangle((200, 40, 390, 210), fill=(50, 110, 210))
    draw.rectangle((30, 200, 220, 390), fill=(90, 50, 30))
    draw.ellipse((200, 200, 390, 390), fill=(50, 150, 60))
    draw.ellipse((140, 140, 190, 190), fill=(18, 18, 18))
    return image


def run_offline(output: Path, *, require: bool) -> int:
    plate = _synthetic_plate()
    result = create_colour_by_numbers(
        plate,
        n_colours=12,
        max_size=420,
        complexity="simple",
        subject_mode="off",
        palette_mode="standard",
        min_region_mm=PHASE_B_MIN_REGION_MM,
        min_a4_dpi=None,
    )
    report = (
        assert_plate_quality(
            result, colour_plate=plate, min_region_mm=PHASE_B_MIN_REGION_MM
        )
        if require
        else evaluate_plate_quality(
            result, colour_plate=plate, min_region_mm=PHASE_B_MIN_REGION_MM
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    result.save(output, stem="phase_b_offline")
    plate.save(output / "phase_b_offline_colour_plate.png")
    print(report.summary())
    print(f"Primary backend (live): {PHASE_B_PRIMARY_BACKEND} / {PHASE_B_PRIMARY_MODEL}")
    print(f"Wrote artefacts under {output}")
    return 0 if report.passed else 1


def run_live(
    *,
    query: str,
    subject_type: str | None,
    output: Path,
    require: bool,
) -> int:
    from colour_by_numbers.generate import generate_colouring_page
    from colour_by_numbers.quality import PlateQualityError

    try:
        page = generate_colouring_page(
            query,
            subject_type=subject_type,
            discover_types=False,
            backend=PHASE_B_PRIMARY_BACKEND,
            pollinations_model=PHASE_B_PRIMARY_MODEL,
            min_region_mm=PHASE_B_MIN_REGION_MM,
            illustration_size=768,
            check_quality=True,
            require_quality=require,
        )
    except PlateQualityError as exc:
        print(exc.report.summary(), file=sys.stderr)
        return 1

    output.mkdir(parents=True, exist_ok=True)
    stem = (page.subject_type.label or "plate").replace(" ", "_")
    page.result.save(output, stem=stem)
    page.illustration.image.save(output / f"{stem}_illustration.png")
    if page.quality is not None:
        print(page.quality.summary())
    print(f"Backend: {page.illustration.backend}")
    print(f"Wrote artefacts under {output}")
    return 0 if page.quality is None or page.quality.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run the checklist on a synthetic plate (no network)",
    )
    parser.add_argument("--query", default="dogs", help="Category / phrase for live run")
    parser.add_argument("--type", dest="subject_type", default="pug")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/phase_b"),
        help="Directory for plate artefacts",
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="Exit non-zero if the quality gate fails",
    )
    args = parser.parse_args(argv)
    if args.offline:
        return run_offline(args.output, require=args.require)
    return run_live(
        query=args.query,
        subject_type=args.subject_type,
        output=args.output,
        require=args.require,
    )


if __name__ == "__main__":
    raise SystemExit(main())
