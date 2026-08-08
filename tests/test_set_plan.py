"""Tests for Phase D set planner."""

from __future__ import annotations

from colour_by_numbers.set_plan import (
    compose_slot_prompt,
    plan_colouring_set,
    slot_bank_for_category,
)


def test_plan_spitfire_has_unique_aircraft_slots() -> None:
    plan = plan_colouring_set(
        "aircraft",
        subject_type="spitfire",
        n_plates=6,
        base_seed=10,
        discover_types=False,
    )
    assert plan.subject_type.label == "spitfire"
    assert plan.subject_type.category == "aircraft"
    assert plan.n_plates == 6
    keys = {(s.aspect, s.scene) for s in plan.slots}
    assert len(keys) == 6
    assert plan.slots[0].seed == 10
    assert plan.slots[1].seed == 11
    # Balanced aircraft mix should include a side-oriented variation.
    covered = set().union(*(set(s.tags) for s in plan.slots))
    assert "side" in covered or any("side" in s.aspect for s in plan.slots)


def test_slot_prompts_include_aspect_and_identity() -> None:
    plan = plan_colouring_set(
        "dogs",
        subject_type="pug",
        n_plates=3,
        discover_types=False,
    )
    for slot in plan.slots:
        assert "aspect:" in slot.prompt.lower() or "Aspect:" in slot.prompt
        assert "scene:" in slot.prompt.lower() or "Scene:" in slot.prompt
        assert "pug" in slot.prompt.lower()
        assert "subject kind: dog" in slot.prompt
        # Pose / composition must lead so fal does not ignore late tags.
        assert slot.prompt.startswith("COMPOSITION") or slot.prompt.startswith(
            "Wide shot"
        )
        aspect_at = min(
            i
            for i in (
                slot.prompt.lower().find("aspect:"),
                slot.prompt.find("Aspect:"),
            )
            if i >= 0
        )
        assert aspect_at < slot.prompt.index("subject kind:")


def test_compose_slot_prompt_keeps_disambiguation() -> None:
    plan = plan_colouring_set(
        "aircraft",
        subject_type="spitfire",
        n_plates=1,
        discover_types=False,
    )
    prompt = compose_slot_prompt(
        plan.subject_type,
        aspect="hangar",
        scene="hangar doorway",
        composition="framed in doorway",
    )
    assert "Supermarine Spitfire" in prompt
    assert "hangar" in prompt.lower()
    assert "no people" in prompt or "no person" in prompt


def test_full_body_slots_drop_portrait_bias() -> None:
    from colour_by_numbers.discover import SubjectType
    from colour_by_numbers.illustrate import illustration_prompt

    subject = SubjectType(
        label="golden retriever",
        category="dogs",
        search_query="golden retriever",
    )
    # Singles keep the curated portrait house look.
    single = illustration_prompt(
        "golden retriever", category="dogs", style_preset="vibrant"
    )
    assert "portrait, centred subject" in single

    full = compose_slot_prompt(
        subject,
        aspect="side profile",
        scene="plain background",
        composition="FULL BODY clear side silhouette, all four legs visible",
        style_preset="vibrant",
        tags=("side", "full_body", "single", "standing"),
    )
    assert full.startswith("Wide shot")
    assert "portrait, centred subject" not in full
    assert "NOT a headshot" in full
    assert "REMEMBER: wide shot" in full
    assert "Aspect: side profile" in full
    # Keep set full-body prompts short enough that fal keeps the pose lock.
    assert len(full.split()) < 250

    portrait = compose_slot_prompt(
        subject,
        aspect="close front portrait",
        scene="plain background",
        composition="head and shoulders facing viewer",
        style_preset="vibrant",
        tags=("front", "portrait", "close", "single"),
    )
    assert "portrait, centred subject" in portrait
    assert portrait.startswith("COMPOSITION:")
    assert "Wide shot" not in portrait


def test_default_bank_used_for_unknown_category() -> None:
    bank = slot_bank_for_category("custom-unknown")
    assert len(bank) >= 4
    plan = plan_colouring_set(
        "red tractor",
        n_plates=4,
        discover_types=False,
    )
    assert plan.n_plates == 4
    assert len({(s.aspect, s.scene) for s in plan.slots}) == 4


def test_plan_to_dict_roundtrip_fields() -> None:
    plan = plan_colouring_set(
        "dogs", subject_type="pug", n_plates=2, discover_types=False
    )
    data = plan.to_dict()
    assert data["subject_label"] == "pug"
    assert len(data["slots"]) == 2
    assert data["slots"][0]["slug"].startswith("01-")
