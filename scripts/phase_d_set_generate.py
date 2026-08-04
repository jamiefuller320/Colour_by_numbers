"""Phase D: plan or generate a varied colouring-plate set from one phrase.

Examples::

    # Plan only (no network)
    python scripts/phase_d_set_generate.py --query aircraft --type spitfire \\
      --set-size 6 --plan-only

    # Live Pollinations set (needs network; slow / rate-limited)
    python scripts/phase_d_set_generate.py --query dogs --type pug --set-size 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colour_by_numbers.set_generate import generate_colouring_set  # noqa: E402
from colour_by_numbers.set_plan import plan_colouring_set  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--type", dest="subject_type", default=None)
    parser.add_argument("--set-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set-attempts", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/phase_d_set"),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write plan.json only (no image generation)",
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="Exit non-zero if the set quality gate fails",
    )
    parser.add_argument("--illustration-size", type=int, default=768)
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    plan = plan_colouring_set(
        args.query,
        subject_type=args.subject_type,
        n_plates=args.set_size,
        base_seed=args.seed,
        discover_types=False,
    )

    if args.plan_only:
        path = args.output / "plan.json"
        path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        print(f"Subject: {plan.subject_type.label} ({plan.subject_type.category})")
        for slot in plan.slots:
            print(f"  {slot.index:02d}. {slot.aspect} / {slot.scene}")
        print(f"Wrote {path}")
        return 0

    generated = generate_colouring_set(
        args.query,
        plan=plan,
        attempts_per_slot=args.set_attempts,
        require_plate_quality=True,
        output_dir=args.output,
        backend="pollinations",
        illustration_size=args.illustration_size,
        subject_mode="off",
    )
    print(f"Subject: {generated.plan.subject_type.label}")
    for item in generated.results:
        print(f"  {item.slot.slug}: {item.status} — {item.reason}")
    if generated.quality is not None:
        print(generated.quality.summary())
    print(f"Wrote set under {args.output}")
    if args.require and not generated.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
