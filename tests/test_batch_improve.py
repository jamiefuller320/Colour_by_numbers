"""Tests for the batch improve auto-tagger and critique conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from colour_by_numbers.batch_improve import (
    assessments_to_critiques,
    auto_tag_plate,
    detect_busy_background,
    detect_cream_background,
    detect_muzzle_fleck,
    detect_silhouette_notch,
    format_batch_report_md,
    BatchImproveReport,
)
from colour_by_numbers.plate_critique import PLATE_ISSUE_TAGS, collate_critiques


def test_new_issue_tags_registered() -> None:
    for tag in (
        "silhouette_notch",
        "cream_background",
        "busy_background",
        "muzzle_fleck",
    ):
        assert tag in PLATE_ISSUE_TAGS


def test_detect_silhouette_notch_on_crown_bite() -> None:
    img = Image.new("RGB", (120, 100), (90, 120, 150))
    draw = ImageDraw.Draw(img)
    draw.ellipse((25, 20, 95, 85), fill=(210, 150, 70))
    # Cool notch cutting into the crown.
    draw.rectangle((50, 15, 70, 35), fill=(70, 100, 140))
    hit = detect_silhouette_notch(img, min_area=8)
    assert hit is not None
    assert hit.tag == "silhouette_notch"


def test_detect_cream_background() -> None:
    img = Image.new("RGB", (100, 100), (230, 210, 170))  # cream bg
    draw = ImageDraw.Draw(img)
    draw.ellipse((30, 25, 70, 75), fill=(180, 110, 50))
    hit = detect_cream_background(img, min_frac=0.05)
    assert hit is not None
    assert hit.tag == "cream_background"


def test_detect_busy_background() -> None:
    img = Image.new("RGB", (160, 120), (80, 110, 140))
    draw = ImageDraw.Draw(img)
    draw.ellipse((50, 30, 110, 100), fill=(200, 140, 60))
    # Many distinct bg tiles.
    colours = [
        (40, 80, 120),
        (60, 100, 130),
        (100, 140, 160),
        (50, 90, 100),
        (70, 120, 90),
        (90, 100, 150),
        (30, 70, 110),
        (110, 130, 170),
        (20, 60, 90),
        (130, 150, 180),
        (45, 85, 125),
        (75, 95, 145),
        (55, 115, 135),
        (35, 75, 105),
        (95, 125, 155),
        (65, 105, 95),
    ]
    i = 0
    for y in range(0, 120, 30):
        for x in range(0, 160, 40):
            if 50 <= x <= 110 and 30 <= y <= 100:
                continue
            draw.rectangle((x, y, x + 38, y + 28), fill=colours[i % len(colours)])
            i += 1
    hit = detect_busy_background(img, max_regions=6)
    assert hit is not None
    assert hit.tag == "busy_background"


def test_detect_muzzle_fleck() -> None:
    img = Image.new("RGB", (100, 100), (100, 130, 160))
    draw = ImageDraw.Draw(img)
    draw.ellipse((20, 15, 80, 85), fill=(210, 160, 90))
    # Nose leather
    draw.ellipse((42, 48, 58, 62), fill=(25, 20, 18))
    # Light fleck on nose
    draw.rectangle((48, 50, 52, 54), fill=(240, 230, 210))
    hit = detect_muzzle_fleck(img, category="dogs")
    assert hit is not None
    assert hit.tag == "muzzle_fleck"


def test_auto_tag_plate_aliases_background() -> None:
    img = Image.new("RGB", (100, 100), (235, 215, 175))
    draw = ImageDraw.Draw(img)
    draw.ellipse((30, 25, 70, 75), fill=(190, 120, 55))
    hits = auto_tag_plate(img, category="dogs", min_score=0.2)
    tags = {h.tag for h in hits}
    assert "cream_background" in tags
    assert "background" in tags


def test_assessments_to_critiques_and_collate(tmp_path: Path) -> None:
    from colour_by_numbers.batch_improve import SlotAssessment, AutoTagHit

    assessments = [
        SlotAssessment(
            plate_id="set/p01",
            slot="p01",
            category="dogs",
            subject="golden retriever",
            rating="fail",
            issues=["silhouette_notch", "outline"],
            hits=[
                AutoTagHit("silhouette_notch", 0.9, "bite"),
                AutoTagHit("outline", 0.7, "alias"),
            ],
            weak=True,
            notes="bite",
        ),
        SlotAssessment(
            plate_id="set/p02",
            slot="p02",
            category="dogs",
            subject="golden retriever",
            rating="needs_work",
            issues=["cream_background", "background"],
            hits=[AutoTagHit("cream_background", 0.6, "cream")],
            weak=True,
            notes="cream",
        ),
    ]
    critiques = assessments_to_critiques(assessments)
    assert len(critiques) == 2
    report = collate_critiques(critiques)
    assert report.by_tag.get("silhouette_notch", 0) >= 1
    dog_hints = " ".join(l.prompt_hint for l in report.lessons if l.category == "dogs")
    assert "silhouette" in dog_hints.lower() or "crown" in dog_hints.lower()


def test_format_batch_report_md() -> None:
    report = BatchImproveReport(
        set_id="demo",
        generated_at="2026-08-09T00:00:00+00:00",
        before=[{"slot": "p01", "rating": "fail", "issues": ["silhouette_notch"]}],
        after=[{"slot": "p01", "rating": "pass", "issues": []}],
        regenerated=["p01"],
        lessons_path="data/plate_lessons.json",
        critiques_appended=1,
        tag_counts_before={"silhouette_notch": 1},
        tag_counts_after={},
        weak_before=["p01"],
        weak_after=[],
    )
    md = format_batch_report_md(report)
    assert "p01" in md
    assert "silhouette_notch" in md
