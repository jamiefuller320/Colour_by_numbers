"""Run the subject-recognition feedback loop on one type.

Asks (via rules / OpenAI vision / human):
  1. Is this recognisable as the requested subject?
  2. How can the generation prompt improve?

Examples::

    # Offline rules critic + live Pollinations retries (needs network)
    python scripts/subject_feedback_loop.py --query aircraft --type spitfire

    # Vision critic when OPENAI_API_KEY is set
    python scripts/subject_feedback_loop.py --query aircraft --type spitfire \\
      --critique-mode openai

    # Dry-run the rules critic / revise path without generating images
    python scripts/subject_feedback_loop.py --query aircraft --type spitfire --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colour_by_numbers.feedback import (  # noqa: E402
    critique_subject_rules,
    revise_prompt,
    run_subject_feedback_loop,
    seed_prompt_with_lessons,
)
from colour_by_numbers.generate import generate_colouring_page  # noqa: E402
from colour_by_numbers.illustrate import illustration_prompt  # noqa: E402


def _placeholder(prompt: str) -> Image.Image:
    """Tiny stand-in image for dry-run / unit-style loops."""
    del prompt
    image = Image.new("RGB", (256, 256), (230, 230, 230))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 80, 220, 160), fill=(80, 100, 140))
    return image


def run_dry(query: str, subject_type: str, *, lessons: Path) -> int:
    category = query
    prompt = illustration_prompt(subject_type, category=category)
    seeded, used = seed_prompt_with_lessons(
        prompt, subject_label=subject_type, category=category, path=lessons
    )
    critique = critique_subject_rules(
        _placeholder(seeded),
        subject_label=subject_type,
        category=category,
        prompt=seeded,
    )
    revised = revise_prompt(
        seeded, critique, subject_label=subject_type, category=category
    )
    print(f"Subject: {subject_type} ({category})")
    print(f"Base prompt: {prompt[:240]}{'…' if len(prompt) > 240 else ''}")
    print(f"Seeded extras: {used or '—'}")
    print(
        f"Rules critique: recognisable={critique.recognisable} "
        f"confidence={critique.confidence:.2f} passed={critique.passed}"
    )
    for issue in critique.issues:
        print(f"  issue: {issue}")
    for tip in critique.improvements:
        print(f"  improve: {tip}")
    print(f"Revised prompt: {revised[:400]}{'…' if len(revised) > 400 else ''}")
    return 0 if critique.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Category / broad query")
    parser.add_argument("--type", required=True, dest="subject_type", help="Concrete type")
    parser.add_argument(
        "--critique-mode",
        choices=["rules", "openai", "human"],
        default="rules",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/subject_feedback"),
    )
    parser.add_argument(
        "--lessons",
        type=Path,
        default=Path("data/subject_lessons.jsonl"),
        help="JSONL lesson store (default: data/subject_lessons.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Critique/revise the seeded prompt only (no image generation)",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Do not append lessons for this run",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        return run_dry(args.query, args.subject_type, lessons=args.lessons)

    page = generate_colouring_page(
        args.query,
        subject_type=args.subject_type,
        discover_types=False,
        backend="fal",
        subject_feedback=True,
        critique_mode=args.critique_mode,
        max_feedback_attempts=args.max_attempts,
        lessons_file=str(args.lessons),
        record_lessons=not args.no_record,
        check_quality=True,
        require_quality=False,
    )
    stem = args.subject_type.strip().lower().replace(" ", "_")[:40]
    paths = page.result.save(args.output, stem=stem)
    illu = args.output / f"{stem}_illustration.png"
    page.illustration.image.save(illu)
    print(f"Subject: {page.subject_type.label}")
    if page.feedback is not None:
        print(page.feedback.notes)
        for i, attempt in enumerate(page.feedback.attempts, start=1):
            c = attempt.critique
            print(
                f"  attempt {i}: recognisable={c.recognisable} "
                f"confidence={c.confidence:.2f} accepted={attempt.accepted}"
            )
    if page.illustration.prompt:
        print(f"Final prompt: {page.illustration.prompt}")
    if page.quality is not None:
        print(page.quality.summary())
    print("Wrote:")
    print(f"  illustration {illu}")
    for label, path in paths.items():
        print(f"  {label:12s} {path}")
    if page.feedback is not None and not page.feedback.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
