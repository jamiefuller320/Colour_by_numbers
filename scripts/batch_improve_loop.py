"""First-slice batch improve loop: assess → learn → regenerate weak slots.

Examples::

    # Assess + collate only (no fal)
    PYTHONPATH=src python3 scripts/batch_improve_loop.py \\
      --set-dir docs/library/sets/dogs-golden-retriever-20260808-5d1654 \\
      --manifest output/golden-retriever-set-v2/manifest.json \\
      --no-regenerate

    # Full loop (re-fals weak slots; needs FAL_KEY)
    PYTHONPATH=src python3 scripts/batch_improve_loop.py \\
      --set-dir docs/library/sets/dogs-golden-retriever-20260808-5d1654 \\
      --manifest output/golden-retriever-set-v2/manifest.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colour_by_numbers.batch_improve import (  # noqa: E402
    format_batch_report_md,
    run_batch_improve_slice,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set-dir",
        type=Path,
        required=True,
        help="Library set directory containing pairs/pNN/",
    )
    parser.add_argument("--set-id", type=str, default=None)
    parser.add_argument("--category", type=str, default="dogs")
    parser.add_argument("--subject", type=str, default="golden retriever")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Original set manifest.json (slot prompts/aspects)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Where to write before/after JSON + report.md",
    )
    parser.add_argument(
        "--critiques",
        type=Path,
        default=Path("data/plate_critiques.jsonl"),
    )
    parser.add_argument(
        "--lessons",
        type=Path,
        default=Path("data/plate_lessons.json"),
    )
    parser.add_argument(
        "--no-regenerate",
        action="store_true",
        help="Assess + collate only; do not call fal",
    )
    parser.add_argument(
        "--max-regenerate",
        type=int,
        default=3,
        help="Max weak slots to re-fal (default 3)",
    )
    parser.add_argument(
        "--known-issues",
        type=Path,
        default=None,
        help="JSON seed of human-known slot issues (rating/issues/notes)",
    )
    parser.add_argument(
        "--force-slots",
        type=str,
        default="",
        help="Comma-separated slots to force-regenerate (e.g. p01,p04,p06)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    force = {s.strip() for s in args.force_slots.split(",") if s.strip()}

    report = run_batch_improve_slice(
        args.set_dir,
        set_id=args.set_id,
        category=args.category,
        subject=args.subject,
        critiques_path=args.critiques,
        lessons_path=args.lessons,
        manifest_path=args.manifest,
        report_dir=args.report_dir,
        regenerate=not args.no_regenerate,
        max_regenerate=args.max_regenerate,
        known_issues_path=args.known_issues,
        force_slots=force or None,
    )
    print(format_batch_report_md(report))
    print(f"\nWrote report under {args.report_dir or Path('output/batch-improve') / report.set_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
