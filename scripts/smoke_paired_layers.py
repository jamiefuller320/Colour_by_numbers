"""Live fal smoke for paired scene + subject-only layers.

Usage::

    PYTHONPATH=src python3 scripts/smoke_paired_layers.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colour_by_numbers.discover import SubjectType
from colour_by_numbers.illustrate import generate_illustration_pair
from colour_by_numbers.pipeline import create_colour_by_numbers
from colour_by_numbers.set_plan import compose_slot_prompt
from colour_by_numbers.style_presets import STYLE_VIBRANT
from colour_by_numbers.subject import (
    align_mask,
    estimate_subject_mask,
    mask_from_subject_layer,
    silhouette_binary_from_mask,
)


def _overlay(scene: Image.Image, mask_bin: np.ndarray) -> Image.Image:
    arr = np.asarray(scene.convert("RGB")).copy()
    outside = ~mask_bin
    arr[outside] = (arr[outside] * 0.35 + np.array([40, 160, 200]) * 0.65).astype(
        np.uint8
    )
    return Image.fromarray(arr)


def _scene_iou(scene: Image.Image, mask_bin: np.ndarray) -> float:
    scene_mask = estimate_subject_mask(scene.convert("RGB"))
    scene_mask = align_mask(scene_mask, scene.size, firm=True)
    scene_bin = silhouette_binary_from_mask(scene_mask, threshold=64)
    inter = float((mask_bin & scene_bin).sum())
    union = float((mask_bin | scene_bin).sum()) or 1.0
    return inter / union


def run_slot(
    out: Path,
    *,
    slug: str,
    aspect: str,
    scene: str,
    composition: str,
    tags: tuple[str, ...],
    seed: int,
) -> dict:
    subject = SubjectType(
        label="golden retriever",
        category="dogs",
        search_query="golden retriever",
    )
    prompt = compose_slot_prompt(
        subject,
        aspect=aspect,
        scene=scene,
        composition=composition,
        style_preset="vibrant",
        tags=tags,
    )
    print(f"\n=== {slug} seed={seed} ===")
    print(prompt[:240])

    pair = generate_illustration_pair(
        subject_type_label="golden retriever",
        category="dogs",
        backend="fal",
        style="vibrant",
        scene_prompt=prompt,
        n_colours=22,
        output_size=768,
        seed=seed,
    )

    slot_dir = out / slug
    slot_dir.mkdir(parents=True, exist_ok=True)
    pair.scene.image.save(slot_dir / "01-scene.png")
    pair.subject_only.image.save(slot_dir / "02-subject-only.png")

    mask = mask_from_subject_layer(pair.scene.image, pair.subject_only.image)
    iou = _scene_iou(pair.scene.image, mask.binary)
    Image.fromarray(mask.alpha).save(slot_dir / "03-mask.png")
    _overlay(pair.scene.image, mask.binary).save(slot_dir / "04-mask-overlay.png")

    preset = STYLE_VIBRANT
    result = create_colour_by_numbers(
        pair.scene.image,
        n_colours=22,
        complexity=preset.complexity,
        subject_mode=preset.subject_mode,
        subject_complexity=preset.subject_complexity,
        background_complexity=preset.background_complexity,
        palette_mode=preset.pipeline_palette_mode or "exact",
        firm_border=True,
        colour_refine=True,
        min_region_mm=preset.min_region_mm,
        min_adjacent_delta_e=preset.min_adjacent_delta_e,
        max_plate_colours=preset.max_plate_colours,
        min_similar_delta_e=preset.min_similar_delta_e,
        subject_bg_separation_mm=preset.subject_bg_separation_mm,
        min_subject_bg_delta_e=preset.min_subject_bg_delta_e,
        silhouette_outline=True,
        subject_mask=mask,
    )
    result.quantized.preview.save(slot_dir / "05-plate.png")
    result.page.outline.save(slot_dir / "06-outline.png")
    if result.printable is not None:
        result.printable.save(slot_dir / "07-printable.png")

    (slot_dir / "notes.txt").write_text(
        f"slug={slug}\n"
        f"seed={seed}\n"
        f"mask_model={mask.model}\n"
        f"mask_fg={mask.foreground_fraction:.3f}\n"
        f"iou_vs_scene_rembg={iou:.3f}\n"
        f"fallback_likely={iou < 0.25}\n"
        f"final_mask_model={getattr(result.subject_mask, 'model', None)}\n"
        f"final_mask_fg={getattr(result.subject_mask, 'foreground_fraction', None)}\n"
        f"n_colours={result.quantized.n_colours}\n"
        f"scene_prompt={pair.scene_prompt}\n"
        f"subject_only_prompt={pair.subject_only_prompt}\n",
        encoding="utf-8",
    )
    summary = {
        "slug": slug,
        "mask_model": mask.model,
        "fg": round(mask.foreground_fraction, 3),
        "iou": round(iou, 3),
        "fallback_likely": iou < 0.25,
        "out": str(slot_dir),
    }
    print(summary)
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = ROOT / "output" / "paired-layer-smoke"
    out.mkdir(parents=True, exist_ok=True)

    slots = [
        (
            "close-portrait",
            "close front portrait",
            "plain background",
            "head and shoulders facing viewer, large expressive eyes",
            ("front", "portrait", "close", "single"),
            42,
        ),
        (
            "side-profile",
            "side profile",
            "plain background",
            "FULL BODY clear side silhouette, all four legs visible",
            ("side", "full_body", "single", "standing"),
            43,
        ),
    ]
    results = []
    for slug, aspect, scene, comp, tags, seed in slots:
        try:
            results.append(
                run_slot(
                    out,
                    slug=slug,
                    aspect=aspect,
                    scene=scene,
                    composition=comp,
                    tags=tags,
                    seed=seed,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception("FAIL %s", slug)
            results.append({"slug": slug, "error": str(exc)})

    print("\n=== SUMMARY ===")
    for r in results:
        print(r)
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
