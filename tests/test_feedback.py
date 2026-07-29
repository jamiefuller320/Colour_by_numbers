"""Tests for the subject-recognition feedback loop."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from colour_by_numbers.feedback import (
    critique_subject_rules,
    load_lessons,
    record_lesson,
    revise_prompt,
    run_subject_feedback_loop,
    seed_prompt_with_lessons,
)
from colour_by_numbers.illustrate import illustration_prompt


def _blank() -> Image.Image:
    image = Image.new("RGB", (64, 64), (200, 200, 200))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 20, 54, 40), fill=(60, 80, 120))
    return image


def test_rules_critic_flags_weak_spitfire_prompt() -> None:
    weak = "spitfire portrait, colouring book"
    critique = critique_subject_rules(
        _blank(),
        subject_label="spitfire",
        category="aircraft",
        prompt=weak,
    )
    assert not critique.passed
    assert critique.improvements
    assert any("elliptical" in tip.lower() or "spitfire" in tip.lower() for tip in critique.improvements)


def test_rules_critic_passes_seeded_spitfire_prompt() -> None:
    base = illustration_prompt("spitfire", category="aircraft")
    seeded, extras = seed_prompt_with_lessons(
        base, subject_label="spitfire", category="aircraft"
    )
    assert extras
    assert "elliptical" in seeded.lower()
    critique = critique_subject_rules(
        _blank(),
        subject_label="spitfire",
        category="aircraft",
        prompt=seeded,
    )
    assert critique.passed
    assert critique.confidence >= 0.7


def test_rules_critic_soft_passes_unknown_subject() -> None:
    critique = critique_subject_rules(
        _blank(),
        subject_label="widget",
        category=None,
        prompt="widget portrait, colouring book",
    )
    assert critique.passed


def test_revise_prompt_adds_feature_cues() -> None:
    weak = "spitfire, colouring book"
    critique = critique_subject_rules(
        _blank(),
        subject_label="spitfire",
        category="aircraft",
        prompt=weak,
    )
    revised = revise_prompt(
        weak, critique, subject_label="spitfire", category="aircraft"
    )
    assert "elliptical" in revised.lower()
    assert "no people" in revised.lower() or "no person" in revised.lower()
    assert "subject kind: aircraft" in revised.lower()


def test_lesson_store_roundtrip(tmp_path: Path) -> None:
    store = tmp_path / "lessons.jsonl"
    critique = critique_subject_rules(
        _blank(),
        subject_label="spitfire",
        category="aircraft",
        prompt="weak",
    )
    record_lesson(
        subject_label="spitfire",
        category="aircraft",
        failed_prompt="weak",
        critique=critique,
        revised_prompt="weak, elliptical wings, propeller",
        accepted=True,
        path=store,
    )
    lessons = load_lessons("spitfire", category="aircraft", path=store)
    assert lessons
    assert any("elliptical" in lesson.lower() for lesson in lessons)


def test_feedback_loop_retries_until_rules_pass(tmp_path: Path) -> None:
    calls: list[str] = []

    def generate_fn(prompt: str) -> Image.Image:
        calls.append(prompt)
        return _blank()

    # Start from a deliberately weak prompt; seeding should strengthen it
    # before attempt 1 so the rules critic can pass without thrashing.
    result = run_subject_feedback_loop(
        subject_label="spitfire",
        category="aircraft",
        initial_prompt="spitfire portrait",
        generate_fn=generate_fn,
        critique_mode="rules",
        max_attempts=3,
        lessons_file=tmp_path / "lessons.jsonl",
        record=True,
    )
    assert calls
    assert "elliptical" in calls[0].lower()
    assert result.passed
    assert result.attempts[-1].accepted


def test_feedback_loop_records_failure_when_exhausted(tmp_path: Path) -> None:
    store = tmp_path / "lessons.jsonl"

    def generate_fn(prompt: str) -> Image.Image:
        del prompt
        return _blank()

    # Aircraft prompt with neither "aircraft" nor "aeroplane" and no known cue.
    result = run_subject_feedback_loop(
        subject_label="mystery plane",
        category="aircraft",
        initial_prompt="portrait only",
        generate_fn=generate_fn,
        critique_mode="rules",
        max_attempts=1,
        lessons_file=store,
        record=True,
    )
    assert not result.passed
    assert store.is_file()
    text = store.read_text(encoding="utf-8")
    assert "mystery plane" in text
    assert '"accepted": false' in text
