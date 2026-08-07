"""Generate a review batch for the GitHub Pages critique gallery.

Writes ``docs/review/manifest.json`` and plate assets under
``docs/review/plates/<id>/``.

Examples::

    # Offline placeholder plates (no network, for UI dev)
    python scripts/generate_review_batch.py --offline

    # Live fal.ai batch (needs FAL_KEY) — one subject per category
    python scripts/generate_review_batch.py --per-category 1

    # Specific subjects
    python scripts/generate_review_batch.py --query dogs --type pug --query cats --type tabby
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colour_by_numbers.generate import generate_colouring_page  # noqa: E402

DOCS_REVIEW = ROOT / "docs" / "review"
CATEGORIES_FILE = ROOT / "docs" / "categories.json"


def _placeholder(subject: str, category: str, size: int = 420) -> Image.Image:
    image = Image.new("RGB", (size, size), (245, 242, 235))
    draw = ImageDraw.Draw(image)
    draw.ellipse((60, 60, size - 60, size - 60), fill=(210, 175, 120), outline=(20, 20, 20), width=3)
    draw.text((size // 2 - 40, size - 48), f"{subject}", fill=(30, 30, 30))
    draw.text((size // 2 - 50, 24), category, fill=(80, 80, 80))
    return image


def _plate_id(category: str, subject: str, index: int = 1) -> str:
    slug = subject.strip().lower().replace(" ", "-")[:40]
    return f"{category}-{slug}-{index:03d}"


def _copy_assets(stem: Path, plate_id: str, manifest_dir: Path) -> dict[str, str]:
    dest = manifest_dir / "plates" / plate_id
    dest.mkdir(parents=True, exist_ok=True)
    mapping = {
        "illustration": "illustration.png",
        "plate": "plate.png",
        "outline": "outline.png",
        "page": "page.png",
    }
    out: dict[str, str] = {}
    for key, name in mapping.items():
        src = stem.parent / f"{stem.name}_{name.replace('.png', '')}.png"
        # generate_colouring_page save uses stem_illustration via separate save
        candidates = [
            stem.parent / f"{stem.name}_{key}.png",
            stem.parent / f"{stem.name}_{name}",
        ]
        if key == "illustration":
            candidates.insert(0, stem.parent / f"{stem.name}_illustration.png")
        if key == "plate":
            candidates.insert(0, stem.parent / f"{stem.name}_quantized.png")
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            continue
        rel = f"plates/{plate_id}/{name}"
        shutil.copy2(found, dest / name)
        out[key] = rel
        svg = found.with_suffix(".svg")
        if key == "plate" and (stem.parent / f"{stem.name}_plate.svg").is_file():
            svg_src = stem.parent / f"{stem.name}_plate.svg"
            (dest / "plate.svg").write_bytes(svg_src.read_bytes())
            out["plate_svg"] = f"plates/{plate_id}/plate.svg"
        if key == "outline" and (stem.parent / f"{stem.name}_outline.svg").is_file():
            svg_src = stem.parent / f"{stem.name}_outline.svg"
            (dest / "outline.svg").write_bytes(svg_src.read_bytes())
            out["outline_svg"] = f"plates/{plate_id}/outline.svg"
    return out


def generate_live(
    jobs: list[tuple[str, str]],
    *,
    output: Path,
    tmp_output: Path,
) -> list[dict]:
    entries: list[dict] = []
    for category, subject in jobs:
        plate_id = _plate_id(category, subject)
        print(f"Generating {plate_id} ({category} / {subject})…")
        page = generate_colouring_page(
            category,
            subject_type=subject,
            discover_types=False,
            backend="fal",
            check_quality=False,
            require_quality=False,
        )
        stem = tmp_output / plate_id
        paths = page.result.save(tmp_output, stem=plate_id)
        page.illustration.image.save(tmp_output / f"{plate_id}_illustration.png")
        images = _copy_assets(stem, plate_id, output)
        entries.append(
            {
                "id": plate_id,
                "category": category,
                "subject": subject,
                "label": page.subject_type.label,
                "backend": page.illustration.backend,
                "prompt": page.illustration.prompt or "",
                "images": images,
            }
        )
        print(f"  wrote {len(images)} assets")
    return entries


def generate_offline(
    jobs: list[tuple[str, str]],
    *,
    output: Path,
    tmp_output: Path,
) -> list[dict]:
    from colour_by_numbers.pipeline import create_colour_by_numbers

    entries: list[dict] = []
    for category, subject in jobs:
        plate_id = _plate_id(category, subject)
        print(f"Placeholder {plate_id} ({category} / {subject})…")
        image = _placeholder(subject, category)
        result = create_colour_by_numbers(
            image,
            n_colours=10,
            complexity="simple",
            subject_mode="off",
            palette_category=category,
        )
        paths = result.save(tmp_output, stem=plate_id)
        images = _copy_assets(tmp_output / plate_id, plate_id, output)
        entries.append(
            {
                "id": plate_id,
                "category": category,
                "subject": subject,
                "label": subject,
                "backend": "offline",
                "prompt": f"placeholder {subject} ({category})",
                "images": images,
            }
        )
    return entries


def default_jobs(per_category: int) -> list[tuple[str, str]]:
    data = json.loads(CATEGORIES_FILE.read_text(encoding="utf-8"))
    jobs: list[tuple[str, str]] = []
    for category, subjects in data.items():
        for subject in subjects[: max(1, per_category)]:
            jobs.append((category, subject))
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DOCS_REVIEW,
        help="Review folder (default: docs/review)",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=1,
        help="Live/offline: subjects per category from categories.json",
    )
    parser.add_argument("--query", action="append", default=[], help="Category (repeatable)")
    parser.add_argument("--type", action="append", dest="subject_type", default=[], help="Subject type")
    parser.add_argument("--offline", action="store_true", help="Placeholder plates (no FAL_KEY)")
    args = parser.parse_args(argv)

    jobs: list[tuple[str, str]] = []
    if args.query:
        types = args.subject_type or [None] * len(args.query)
        if len(types) < len(args.query):
            types.extend([None] * (len(args.query) - len(types)))
        for category, subject in zip(args.query, types):
            cat_data = json.loads(CATEGORIES_FILE.read_text(encoding="utf-8"))
            subject = subject or cat_data.get(category, [category])[0]
            jobs.append((category, subject))
    else:
        jobs = default_jobs(args.per_category)

    args.output.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output / ".tmp"
    tmp_output.mkdir(parents=True, exist_ok=True)

    if args.offline:
        entries = generate_offline(jobs, output=args.output, tmp_output=tmp_output)
    else:
        entries = generate_live(jobs, output=args.output, tmp_output=tmp_output)

    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "issue_tags": __import__("colour_by_numbers.plate_critique", fromlist=["PLATE_ISSUE_TAGS"]).PLATE_ISSUE_TAGS,
        "plates": entries,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path} ({len(entries)} plates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
