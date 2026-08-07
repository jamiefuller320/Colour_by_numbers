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
            issues=("eyes",),
            notes="Flat eyes",
        ),
    ]
    report = collate_critiques(critiques)
    assert report.by_tag["nose_detail"] == 1
    assert report.by_tag["eyes"] == 1
    assert any("nostrils" in hint for hint in report.global_hints)


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
