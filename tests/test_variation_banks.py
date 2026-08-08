"""Tests for bulk themed prompt variation banks."""

from __future__ import annotations

from colour_by_numbers.set_plan import plan_colouring_set, plan_mixed_colouring_set
from colour_by_numbers.variation_banks import (
    AIRCRAFT_VARIATIONS,
    ANIMAL_VARIATIONS,
    select_varied_slots,
    variation_bank_for_category,
)


def test_animal_bank_covers_requested_mix() -> None:
    tags = set().union(*(slot.tags for slot in ANIMAL_VARIATIONS))
    for needed in (
        "front",
        "side",
        "oblique",
        "sitting",
        "standing",
        "single",
        "group",
        "portrait",
        "scene",
        "close",
    ):
        assert needed in tags


def test_aircraft_bank_covers_flight_states() -> None:
    tags = set().union(*(slot.tags for slot in AIRCRAFT_VARIATIONS))
    for needed in ("on_ground", "takeoff", "airborne", "landing", "side", "oblique"):
        assert needed in tags


def test_select_varied_slots_mixes_tags_for_dogs() -> None:
    picked = select_varied_slots("dogs", 8, seed=3)
    assert len(picked) == 8
    covered = set().union(*(slot.tags for slot in picked))
    # A balanced 8-plate animal set should open several dimensions.
    assert {"front", "side"} & covered
    assert {"portrait", "full_body", "scene"} & covered
    assert {"single", "group"} & covered or "sitting" in covered
    keys = {(s.aspect, s.scene) for s in picked}
    assert len(keys) == 8


def test_select_varied_slots_mixes_aircraft_states() -> None:
    picked = select_varied_slots("aircraft", 8, seed=1)
    covered = set().union(*(slot.tags for slot in picked))
    assert "airborne" in covered
    assert "on_ground" in covered or "takeoff" in covered
    assert len({(s.aspect, s.scene) for s in picked}) == 8


def test_plan_balanced_variety_embeds_tags_in_prompt() -> None:
    plan = plan_colouring_set(
        "dogs",
        subject_type="pug",
        n_plates=6,
        discover_types=False,
        variety="balanced",
        base_seed=5,
    )
    assert plan.n_plates == 6
    assert all(slot.tags for slot in plan.slots)
    assert any(
        "variation tags:" in slot.prompt or "; tags:" in slot.prompt
        for slot in plan.slots
    )
    covered = set().union(*(set(slot.tags) for slot in plan.slots))
    assert len(covered) >= 6


def test_plan_sequential_variety_still_unique() -> None:
    plan = plan_colouring_set(
        "aircraft",
        subject_type="biplane",
        n_plates=6,
        discover_types=False,
        variety="sequential",
    )
    assert len({(s.aspect, s.scene) for s in plan.slots}) == 6
    # Sequential starts at the first aircraft bank entry.
    bank = variation_bank_for_category("aircraft")
    assert plan.slots[0].aspect == bank[0].aspect


def test_mixed_plan_uses_category_appropriate_banks() -> None:
    plan = plan_mixed_colouring_set(
        [("dogs", "pug"), ("aircraft", "biplane")],
        plates_per_subject=4,
        discover_types=False,
        variety="balanced",
    )
    dog_slots = [s for s in plan.slots if s.category == "dogs"]
    air_slots = [s for s in plan.slots if s.category == "aircraft"]
    assert len(dog_slots) == 4 and len(air_slots) == 4
    dog_tags = set().union(*(set(s.tags) for s in dog_slots))
    air_tags = set().union(*(set(s.tags) for s in air_slots))
    assert "sitting" in dog_tags or "standing" in dog_tags
    assert "airborne" in air_tags or "takeoff" in air_tags
