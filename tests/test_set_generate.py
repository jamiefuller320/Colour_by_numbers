"""Tests for Phase D set generation (offline / mocked)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from colour_by_numbers.generate import GeneratedPage
from colour_by_numbers.illustrate import IllustrationResult
from colour_by_numbers.pipeline import create_colour_by_numbers
from colour_by_numbers.quality import PHASE_B_MIN_REGION_MM, evaluate_plate_quality
from colour_by_numbers.set_generate import (
    evaluate_set_quality,
    generate_colouring_set,
    hamming_distance,
    image_dhash,
    is_near_duplicate,
)
from colour_by_numbers.set_plan import plan_colouring_set


def _subject_plate(colour: tuple[int, int, int], *, size: int = 420) -> Image.Image:
    image = Image.new("RGB", (size, size), (240, 238, 232))
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 30, 250, 250), fill=colour)
    draw.rectangle((200, 40, 390, 210), fill=(50, 110, 210))
    draw.rectangle((30, 200, 220, 390), fill=(90, 50, 30))
    draw.ellipse((200, 200, 390, 390), fill=(50, 150, 60))
    return image


def _page_from_plate(plate: Image.Image, subject_label: str = "pug") -> GeneratedPage:
    from colour_by_numbers.discover import SubjectType

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
    quality = evaluate_plate_quality(
        result, colour_plate=plate, min_region_mm=PHASE_B_MIN_REGION_MM
    )
    subject = SubjectType(label=subject_label, category="dogs", search_query=subject_label)
    illustration = IllustrationResult(
        image=plate,
        backend="test",
        subject_type_label=subject_label,
        n_colours=result.quantized.n_colours,
        prompt="test",
    )
    return GeneratedPage(
        illustration=illustration,
        result=result,
        subject_type=subject,
        quality=quality,
    )


def test_dhash_identical_images_match() -> None:
    a = _subject_plate((230, 170, 60))
    b = a.copy()
    ha, hb = image_dhash(a), image_dhash(b)
    assert hamming_distance(ha, hb) == 0
    assert is_near_duplicate(ha, [hb])


def test_dhash_distinct_layouts_differ() -> None:
    a = _subject_plate((230, 170, 60))
    b = Image.new("RGB", (420, 420), (240, 238, 232))
    draw = ImageDraw.Draw(b)
    draw.rectangle((20, 200, 400, 380), fill=(20, 120, 200))
    draw.ellipse((40, 40, 180, 180), fill=(200, 40, 40))
    assert hamming_distance(image_dhash(a), image_dhash(b)) > 10


def test_generate_set_accepts_varied_mocked_plates(tmp_path: Path) -> None:
    layouts = [
        # Large centred oval + side block
        lambda d: (
            d.ellipse((40, 40, 320, 320), fill=(230, 170, 60)),
            d.rectangle((300, 60, 400, 360), fill=(50, 110, 210)),
            d.rectangle((40, 300, 280, 400), fill=(90, 50, 30)),
        ),
        # Top band + bottom circle
        lambda d: (
            d.rectangle((30, 30, 390, 160), fill=(200, 80, 80)),
            d.ellipse((80, 180, 340, 400), fill=(80, 160, 90)),
            d.rectangle((30, 180, 100, 400), fill=(60, 90, 180)),
        ),
        # Diagonal-ish blocks
        lambda d: (
            d.rectangle((20, 20, 200, 400), fill=(60, 90, 180)),
            d.ellipse((180, 40, 400, 260), fill=(230, 170, 60)),
            d.rectangle((180, 280, 400, 400), fill=(200, 80, 80)),
        ),
        # Four corner shapes
        lambda d: (
            d.ellipse((20, 20, 200, 200), fill=(80, 160, 90)),
            d.rectangle((220, 20, 400, 200), fill=(200, 80, 80)),
            d.rectangle((20, 220, 200, 400), fill=(50, 110, 210)),
            d.ellipse((220, 220, 400, 400), fill=(230, 170, 60)),
        ),
    ]
    calls: list[str] = []

    def fake_generate(query, **kwargs):
        del query
        idx = len(calls)
        calls.append(kwargs.get("prompt_override") or "")
        plate = Image.new("RGB", (420, 420), (240, 238, 232))
        layouts[idx % len(layouts)](ImageDraw.Draw(plate))
        page = _page_from_plate(plate, subject_label="pug")
        from colour_by_numbers.discover import SubjectType

        return GeneratedPage(
            illustration=page.illustration,
            result=page.result,
            subject_type=SubjectType(
                label="pug", category="dogs", search_query="pug dog"
            ),
            quality=page.quality,
        )

    generated = generate_colouring_set(
        "dogs",
        subject_type="pug",
        n_plates=4,
        discover_types=False,
        attempts_per_slot=1,
        require_plate_quality=True,
        output_dir=tmp_path,
        generate_page_fn=fake_generate,
    )
    assert len(calls) == 4
    assert len(generated.accepted) == 4
    assert generated.quality is not None
    assert generated.quality.passed, generated.quality.summary()
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "plan.json").is_file()


def test_generate_set_rejects_near_duplicates() -> None:
    plate = _subject_plate((230, 170, 60))
    page = _page_from_plate(plate)

    def always_same(query, **kwargs):
        del query, kwargs
        from colour_by_numbers.discover import SubjectType

        return GeneratedPage(
            illustration=page.illustration,
            result=page.result,
            subject_type=SubjectType(
                label="pug", category="dogs", search_query="pug"
            ),
            quality=page.quality,
        )

    generated = generate_colouring_set(
        "dogs",
        subject_type="pug",
        n_plates=3,
        discover_types=False,
        attempts_per_slot=1,
        require_plate_quality=True,
        generate_page_fn=always_same,
    )
    assert len(generated.accepted) == 1
    assert sum(1 for r in generated.results if r.reason.startswith("near-duplicate")) >= 1
    assert generated.quality is not None
    assert not generated.quality.passed


def test_evaluate_set_quality_requires_all_plates() -> None:
    plan = plan_colouring_set(
        "dogs", subject_type="pug", n_plates=2, discover_types=False
    )
    report = evaluate_set_quality(plan, results=[])
    assert not report.passed
    names = {c.name: c for c in report.checks}
    assert not names["accepted_count"].passed
