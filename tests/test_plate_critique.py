"""Tests for plate-level critique collation."""

from __future__ import annotations

import json
from pathlib import Path

from colour_by_numbers.plate_critique import (
    PlateCritique,
    collate_critiques,
    import_critiques_json,
    load_plate_lessons,
    record_plate_critique,
    seed_prompt_with_plate_lessons,
    write_lessons_json,
)


def test_collate_groups_failures_by_tag() -> None:
    critiques = [
        PlateCritique(
            plate_id="dogs-pug-001",
            category="dogs",
            subject="pug",
            rating="fail",
            issues=("nose_detail",),
            notes="No nostrils",
        ),
        PlateCritique(
            plate_id="cats-tabby-001",
            category="cats",
            subject="tabby cat",
            rating="needs_work",
            issues=("eyes", "colours"),
            notes="Flat eyes",
        ),
        PlateCritique(
            plate_id="boats-sail-001",
            category="boats",
            subject="sailboat",
            rating="needs_work",
            issues=("too_simple",),
            notes="Too flat",
        ),
    ]
    report = collate_critiques(critiques)
    assert report.by_tag["nose_detail"] == 1
    assert report.by_tag["eyes"] == 1
    # Mammal face cues must not become global (they polluted vehicle prompts).
    assert not any("nostril" in hint.lower() for hint in report.global_hints)
    assert any("colour" in hint.lower() or "value" in hint.lower() for hint in report.global_hints)
    dog_lesson = next(l for l in report.lessons if l.category == "dogs")
    assert "nostril" in dog_lesson.prompt_hint.lower()
    boat_lesson = next(l for l in report.lessons if l.category == "boats")
    assert "panel" in boat_lesson.prompt_hint.lower() or "structural" in boat_lesson.prompt_hint.lower()


def test_animal_lessons_do_not_seed_vehicle_prompts(tmp_path: Path) -> None:
    lessons_path = tmp_path / "lessons.json"
    lessons_path.write_text(
        json.dumps(
            {
                "global_hints": [
                    "clearly defined nose with visible nostrils and muzzle wrinkles, "
                    "nose as separate colour regions",
                    "full subject in frame with a small margin, centred, not over-cropped",
                ],
                "lessons": [
                    {
                        "category": "dogs",
                        "tag": "nose_detail",
                        "count": 1,
                        "examples": [],
                        "prompt_hint": (
                            "clearly defined nose with visible nostrils and "
                            "muzzle wrinkles, nose as separate colour regions"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    boat_prompt, boat_applied = seed_prompt_with_plate_lessons(
        "sailboat side view", category="boats", path=lessons_path
    )
    assert "nostril" not in boat_prompt.lower()
    assert "muzzle" not in boat_prompt.lower()
    assert any("margin" in h.lower() or "crop" in h.lower() for h in boat_applied)
    dog_prompt, _ = seed_prompt_with_plate_lessons(
        "golden retriever portrait", category="dogs", path=lessons_path
    )
    assert "nostril" in dog_prompt.lower()


def test_import_and_lessons_round_trip(tmp_path: Path) -> None:
    store = tmp_path / "critiques.jsonl"
    payload = {
        "critiques": [
            {
                "plate_id": "dogs-pug-001",
                "category": "dogs",
                "subject": "pug",
                "rating": "fail",
                "issues": ["nose_detail"],
                "notes": "Missing nose detail",
                "suggested_prompt": "defined nostrils",
                "reviewed_at": "2026-08-07T10:00:00+00:00",
            }
        ]
    }
    count = import_critiques_json(payload, path=store)
    assert count == 1
    from colour_by_numbers.plate_critique import load_critiques

    lessons_path = tmp_path / "lessons.json"
    report = collate_critiques(load_critiques(path=store))
    write_lessons_json(report, path=lessons_path)
    hints = load_plate_lessons(category="dogs", path=lessons_path)
    assert hints
    prompt, applied = seed_prompt_with_plate_lessons(
        "pug dog portrait", category="dogs", path=lessons_path
    )
    assert applied
    assert "nostril" in prompt.lower() or "nose" in prompt.lower()


def test_record_plate_critique_appends(tmp_path: Path) -> None:
    store = tmp_path / "critiques.jsonl"
    record_plate_critique(
        PlateCritique(
            plate_id="x",
            category="dogs",
            subject="pug",
            rating="fail",
            issues=("nose_detail",),
        ),
        path=store,
    )
    lines = store.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["plate_id"] == "x"


def test_seed_lessons_skip_conflicts_for_vibrant_full_body(tmp_path: Path) -> None:
    lessons_path = tmp_path / "lessons.json"
    lessons_path.write_text(
        json.dumps(
            {
                "global_hints": [
                    "distinct flat colour blocks with clear value steps between "
                    "neighbouring parts, prefer 12–16 colours",
                    "unmistakable animal of the correct species/breed",
                    "enough distinct colour regions for depth",
                    "extra filler hint four",
                ],
                "lessons": [
                    {
                        "category": "dogs",
                        "tag": "eyes",
                        "count": 2,
                        "prompt_hint": (
                            "both eyes matching, each with separate dark pupil "
                            "and lighter iris fills"
                        ),
                    },
                    {
                        "category": "dogs",
                        "tag": "nose_detail",
                        "count": 2,
                        "prompt_hint": (
                            "clearly defined nose with visible nostrils and "
                            "muzzle wrinkles"
                        ),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    locked = (
        "Wide shot of a golden retriever: FULL BODY side silhouette. "
        "Aspect: side profile. Camera pulled back so the entire golden "
        "retriever is visible. adult vibrant paint-by-numbers kit style"
    )
    prompt, applied = seed_prompt_with_plate_lessons(
        locked,
        category="dogs",
        path=lessons_path,
        style_preset="vibrant",
    )
    assert "12–16" not in prompt
    assert "prefer 12" not in prompt.lower()
    assert "both eyes matching" not in prompt.lower()
    assert "nostril" not in prompt.lower()
    assert len(applied) <= 1
