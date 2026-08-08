"""Quick fal smoke: generate set-slot illustrations without full outlining.

Usage::

    PYTHONPATH=src python3 scripts/smoke_set_poses.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colour_by_numbers.discover import SubjectType
from colour_by_numbers.illustrate import generate_illustration
from colour_by_numbers.plate_critique import seed_prompt_with_plate_lessons
from colour_by_numbers.set_plan import compose_slot_prompt


def main() -> int:
    out = ROOT / "output" / "fal-pose-smoke"
    out.mkdir(parents=True, exist_ok=True)
    subject = SubjectType(
        label="golden retriever",
        category="dogs",
        search_query="golden retriever",
    )
    slots = [
        (
            "side-profile",
            "side profile",
            "plain background",
            "FULL BODY clear side silhouette, all four legs visible",
            ("side", "full_body", "single", "standing"),
        ),
        (
            "lying-relaxed",
            "lying relaxed",
            "soft blanket",
            "FULL BODY lying stretched along the frame, whole torso and all four legs visible, calm pose",
            ("side", "full_body", "single", "lying", "scene"),
        ),
        (
            "close-portrait",
            "close front portrait",
            "plain background",
            "head and shoulders facing viewer, large expressive eyes",
            ("front", "portrait", "close", "single"),
        ),
    ]
    for i, (slug, aspect, scene, comp, tags) in enumerate(slots, 1):
        prompt = compose_slot_prompt(
            subject,
            aspect=aspect,
            scene=scene,
            composition=comp,
            style_preset="vibrant",
            tags=tags,
        )
        prompt, applied = seed_prompt_with_plate_lessons(
            prompt, category="dogs", style_preset="vibrant"
        )
        print(f"=== {i} {slug} ({len(prompt.split())}w, lessons={len(applied)}) ===")
        print(prompt[:220])
        result = generate_illustration(
            None,
            subject_type_label="golden retriever",
            category="dogs",
            backend="fal",
            style="vibrant",
            prompt_override=prompt,
            n_colours=28,
            output_size=640,
            seed=20 + i,
            prepare_for_colouring=True,
        )
        path = out / f"{i:02d}-{slug}.png"
        result.image.save(path)
        print(f"Wrote {path} size={result.image.size}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
