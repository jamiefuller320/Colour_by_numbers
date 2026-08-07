"""Import and collate plate critiques from the Pages review gallery.

Examples::

    # Import an export downloaded from the review page
    python scripts/collate_plate_critiques.py --import review-export.json

    # Rebuild lessons from the JSONL store on disk
    python scripts/collate_plate_critiques.py --rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colour_by_numbers.plate_critique import (  # noqa: E402
    collate_critiques,
    format_collation_report,
    import_critiques_json,
    load_critiques,
    write_lessons_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--import",
        dest="import_path",
        type=Path,
        help="JSON export from the Pages review UI",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("data/plate_critiques.jsonl"),
        help="Critique JSONL store (default: data/plate_critiques.jsonl)",
    )
    parser.add_argument(
        "--lessons",
        type=Path,
        default=Path("data/plate_lessons.json"),
        help="Collated lessons output (default: data/plate_lessons.json)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild lessons from the JSONL store without importing",
    )
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args(argv)

    if args.import_path:
        payload = json.loads(args.import_path.read_text(encoding="utf-8"))
        count = import_critiques_json(payload, path=args.store)
        print(f"Imported {count} critique(s) into {args.store}")

    if not args.import_path and not args.rebuild:
        parser.error("Provide --import FILE or --rebuild")

    critiques = load_critiques(path=args.store)
    report = collate_critiques(critiques, min_count=args.min_count)
    lessons_path = write_lessons_json(report, path=args.lessons)
    print(format_collation_report(report))
    print(f"\nWrote collated lessons to {lessons_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
