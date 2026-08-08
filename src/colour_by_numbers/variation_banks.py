"""Category-aware prompt variation banks for bulk themed generation.

Each slot carries tags so planners can build a *mixture* of interesting views
(front / side / oblique, sitting / standing, portrait / scene, single / group,
on-ground / takeoff / airborne, …) instead of repeating near-identical poses.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariationSlot:
    """One reusable prompt variation."""

    aspect: str
    scene: str
    composition: str
    tags: frozenset[str]

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.aspect, self.scene, self.composition)


def _slot(
    aspect: str,
    scene: str,
    composition: str,
    *tags: str,
) -> VariationSlot:
    return VariationSlot(
        aspect=aspect,
        scene=scene,
        composition=composition,
        tags=frozenset(tags),
    )


# ---------------------------------------------------------------------------
# Family banks (shared across related categories)
# ---------------------------------------------------------------------------

ANIMAL_VARIATIONS: tuple[VariationSlot, ...] = (
    _slot(
        "close front portrait",
        "plain background",
        "head and shoulders facing viewer, large expressive eyes",
        "front",
        "portrait",
        "close",
        "single",
        "standing",
    ),
    _slot(
        "side profile",
        "plain background",
        "FULL BODY clear side silhouette, all four legs visible, head to tail in frame",
        "side",
        "full_body",
        "single",
        "standing",
    ),
    _slot(
        "oblique three-quarter",
        "simple yard",
        "FULL BODY angled toward viewer, three-quarter standing pose, legs and tail visible",
        "oblique",
        "full_body",
        "single",
        "standing",
        "scene",
    ),
    _slot(
        "sitting front",
        "garden lawn",
        "FULL BODY sitting facing viewer, legs and tail visible where typical",
        "front",
        "full_body",
        "single",
        "sitting",
        "scene",
    ),
    _slot(
        "sitting side",
        "rug",
        "FULL BODY sitting side profile, head to paws in frame",
        "side",
        "full_body",
        "single",
        "sitting",
        "scene",
    ),
    _slot(
        "standing scene",
        "park path",
        "FULL BODY standing in a simple outdoor scene, head to paws in frame",
        "oblique",
        "full_body",
        "single",
        "standing",
        "scene",
    ),
    _slot(
        "close oblique portrait",
        "plain background",
        "head-and-shoulders study, three-quarter view, muzzle and eyes clear",
        "oblique",
        "portrait",
        "close",
        "single",
    ),
    _slot(
        "action stride",
        "open field",
        "FULL BODY mid-stride or playful motion, all legs visible, simple ground",
        "side",
        "full_body",
        "single",
        "action",
        "scene",
    ),
    _slot(
        "pair together",
        "plain background",
        "two FULL BODY subjects side by side, both head-to-paws visible",
        "front",
        "full_body",
        "group",
        "sitting",
    ),
    _slot(
        "group scene",
        "simple meadow",
        "small group of three FULL BODY figures, staggered depth, still colourable",
        "oblique",
        "full_body",
        "group",
        "standing",
        "scene",
    ),
    _slot(
        "lying relaxed",
        "soft blanket",
        "FULL BODY lying down facing viewer, head torso and legs visible, calm pose",
        "front",
        "full_body",
        "single",
        "lying",
        "scene",
    ),
    _slot(
        "looking back",
        "plain background",
        "FULL BODY mostly side-on with over-shoulder glance, not a head crop",
        "side",
        "full_body",
        "single",
        "standing",
    ),
)

HORSE_VARIATIONS: tuple[VariationSlot, ...] = (
    _slot(
        "close front portrait",
        "plain background",
        "horse head and neck facing viewer, clear eyes and muzzle",
        "front",
        "portrait",
        "close",
        "single",
        "standing",
    ),
    _slot(
        "side profile",
        "paddock",
        "full horse side silhouette, all four legs visible",
        "side",
        "full_body",
        "single",
        "standing",
        "scene",
    ),
    _slot(
        "oblique standing",
        "field",
        "three-quarter standing horse, simple ground",
        "oblique",
        "full_body",
        "single",
        "standing",
        "scene",
    ),
    _slot(
        "grazing",
        "meadow",
        "horse grazing head down, full body readable",
        "side",
        "full_body",
        "single",
        "action",
        "scene",
    ),
    _slot(
        "canter",
        "open field",
        "horse in canter mid-stride, simple horizon",
        "side",
        "full_body",
        "single",
        "action",
        "scene",
    ),
    _slot(
        "foal with adult",
        "paddock",
        "adult horse with foal beside, both fully visible",
        "oblique",
        "full_body",
        "group",
        "standing",
        "scene",
    ),
    _slot(
        "close oblique portrait",
        "plain background",
        "horse head three-quarter view",
        "oblique",
        "portrait",
        "close",
        "single",
    ),
    _slot(
        "rear three-quarter",
        "stable yard",
        "horse angled away looking back",
        "oblique",
        "full_body",
        "single",
        "standing",
        "scene",
    ),
)

BIRD_VARIATIONS: tuple[VariationSlot, ...] = (
    _slot(
        "perched side view",
        "branch",
        "bird on a simple branch, clear side profile and beak",
        "side",
        "full_body",
        "single",
        "standing",
        "scene",
    ),
    _slot(
        "close front portrait",
        "plain background",
        "head and chest facing viewer, distinct eye and beak",
        "front",
        "portrait",
        "close",
        "single",
    ),
    _slot(
        "oblique perched",
        "plain background",
        "three-quarter perched pose",
        "oblique",
        "full_body",
        "single",
        "standing",
    ),
    _slot(
        "wings spread airborne",
        "clear sky",
        "bird gliding, wings open, airborne silhouette",
        "side",
        "full_body",
        "single",
        "airborne",
        "action",
        "scene",
    ),
    _slot(
        "taking off",
        "branch",
        "wings raised leaving the perch",
        "oblique",
        "full_body",
        "single",
        "action",
        "scene",
    ),
    _slot(
        "standing ground",
        "simple ground",
        "full body standing bird",
        "side",
        "full_body",
        "single",
        "standing",
        "scene",
    ),
    _slot(
        "pair on branch",
        "branch",
        "two birds perched together, both readable",
        "side",
        "full_body",
        "group",
        "standing",
        "scene",
    ),
    _slot(
        "nesting scene",
        "nest",
        "bird on a simple nest",
        "oblique",
        "full_body",
        "single",
        "sitting",
        "scene",
    ),
)

PERSON_VARIATIONS: tuple[VariationSlot, ...] = (
    _slot(
        "close front portrait",
        "plain background",
        "face and shoulders facing viewer, clear features",
        "front",
        "portrait",
        "close",
        "single",
        "standing",
    ),
    _slot(
        "side profile portrait",
        "plain background",
        "head side profile, clear nose and jaw",
        "side",
        "portrait",
        "close",
        "single",
    ),
    _slot(
        "oblique portrait",
        "plain background",
        "three-quarter face toward viewer",
        "oblique",
        "portrait",
        "close",
        "single",
    ),
    _slot(
        "standing full body front",
        "plain background",
        "person standing facing viewer, full figure",
        "front",
        "full_body",
        "single",
        "standing",
    ),
    _slot(
        "standing side",
        "simple room",
        "full body side profile standing",
        "side",
        "full_body",
        "single",
        "standing",
        "scene",
    ),
    _slot(
        "sitting scene",
        "simple chair",
        "person sitting, full figure readable",
        "oblique",
        "full_body",
        "single",
        "sitting",
        "scene",
    ),
    _slot(
        "pair portrait",
        "plain background",
        "two people side by side, both faces clear",
        "front",
        "portrait",
        "group",
        "standing",
    ),
    _slot(
        "walking scene",
        "park path",
        "person walking, mid-stride, simple background",
        "side",
        "full_body",
        "single",
        "action",
        "scene",
    ),
)

AIRCRAFT_VARIATIONS: tuple[VariationSlot, ...] = (
    _slot(
        "side view airborne",
        "clear sky",
        "full aeroplane side silhouette in flight, gear up",
        "side",
        "airborne",
        "single",
        "scene",
    ),
    _slot(
        "oblique airborne",
        "above soft clouds",
        "three-quarter view banking gently in flight",
        "oblique",
        "airborne",
        "single",
        "action",
        "scene",
    ),
    _slot(
        "front airborne",
        "sky",
        "nose-on flying toward viewer, wings readable",
        "front",
        "airborne",
        "single",
        "scene",
    ),
    _slot(
        "on ground parked",
        "airfield",
        "parked on tarmac, wheels visible, full silhouette",
        "oblique",
        "on_ground",
        "single",
        "scene",
    ),
    _slot(
        "side view on ground",
        "runway",
        "aircraft parked side profile on ground",
        "side",
        "on_ground",
        "single",
        "scene",
    ),
    _slot(
        "takeoff",
        "runway",
        "wheels just leaving the ground, nose slightly up",
        "side",
        "takeoff",
        "single",
        "action",
        "scene",
    ),
    _slot(
        "landing approach",
        "runway approach",
        "low approach, landing gear down",
        "oblique",
        "landing",
        "single",
        "action",
        "scene",
    ),
    _slot(
        "hangar scene",
        "hangar doorway",
        "aeroplane framed in hangar opening, on ground",
        "oblique",
        "on_ground",
        "single",
        "scene",
    ),
    _slot(
        "top plan view",
        "plain background",
        "plan-view silhouette, wings spread",
        "top",
        "on_ground",
        "single",
        "close",
    ),
    _slot(
        "formation pair",
        "clear sky",
        "two aircraft airborne, staggered, both readable",
        "oblique",
        "airborne",
        "group",
        "scene",
    ),
)

LAND_VEHICLE_VARIATIONS: tuple[VariationSlot, ...] = (
    _slot(
        "side view parked",
        "driveway",
        "full vehicle side silhouette, on ground",
        "side",
        "on_ground",
        "single",
        "scene",
    ),
    _slot(
        "oblique front",
        "street",
        "three-quarter front, headlights and side visible",
        "oblique",
        "on_ground",
        "single",
        "scene",
    ),
    _slot(
        "front view",
        "plain background",
        "head-on facing viewer, wheels on ground",
        "front",
        "on_ground",
        "single",
    ),
    _slot(
        "rear three-quarter",
        "road",
        "rear and side visible, parked or slow roll",
        "oblique",
        "on_ground",
        "single",
        "scene",
    ),
    _slot(
        "driving scene",
        "open road",
        "vehicle in motion on a simple road, horizon clear",
        "side",
        "moving",
        "single",
        "action",
        "scene",
    ),
    _slot(
        "garage scene",
        "garage",
        "parked beside a simple garage doorway",
        "oblique",
        "on_ground",
        "single",
        "scene",
    ),
    _slot(
        "top plan view",
        "plain background",
        "plan-view vehicle silhouette",
        "top",
        "on_ground",
        "single",
        "close",
    ),
    _slot(
        "pair parked",
        "car park",
        "two vehicles side by side, both fully visible",
        "oblique",
        "on_ground",
        "group",
        "scene",
    ),
)

WATER_VEHICLE_VARIATIONS: tuple[VariationSlot, ...] = (
    _slot(
        "side view afloat",
        "calm water",
        "full boat side silhouette on water",
        "side",
        "afloat",
        "single",
        "scene",
    ),
    _slot(
        "oblique harbour",
        "harbour",
        "boat angled toward viewer at a simple quay",
        "oblique",
        "docked",
        "single",
        "scene",
    ),
    _slot(
        "front bow view",
        "water",
        "bow facing viewer",
        "front",
        "afloat",
        "single",
        "scene",
    ),
    _slot(
        "under sail",
        "open sea",
        "sailing with simple waves, underway",
        "side",
        "moving",
        "single",
        "action",
        "scene",
    ),
    _slot(
        "docked pier",
        "pier",
        "tied at a simple pier, on water",
        "oblique",
        "docked",
        "single",
        "scene",
    ),
    _slot(
        "top plan view",
        "plain water",
        "plan-view boat silhouette",
        "top",
        "afloat",
        "single",
        "close",
    ),
    _slot(
        "rowing",
        "lake",
        "small boat with oars, underway",
        "side",
        "moving",
        "single",
        "action",
        "scene",
    ),
    _slot(
        "pair at anchor",
        "bay",
        "two boats floating near a simple shoreline",
        "oblique",
        "afloat",
        "group",
        "scene",
    ),
)

FLOWER_VARIATIONS: tuple[VariationSlot, ...] = (
    _slot(
        "centred portrait",
        "plain background",
        "single bloom filling the frame",
        "front",
        "portrait",
        "close",
        "single",
    ),
    _slot(
        "side view stem",
        "plain background",
        "bloom and stem from the side",
        "side",
        "full_body",
        "single",
    ),
    _slot(
        "oblique bloom",
        "plain background",
        "flower angled toward viewer",
        "oblique",
        "portrait",
        "single",
    ),
    _slot(
        "top view",
        "plain background",
        "looking down into the bloom",
        "top",
        "portrait",
        "close",
        "single",
    ),
    _slot(
        "bud and bloom",
        "plain background",
        "open flower beside a bud",
        "oblique",
        "full_body",
        "group",
    ),
    _slot(
        "in a pot scene",
        "simple pot",
        "potted plant, centred",
        "front",
        "full_body",
        "single",
        "scene",
    ),
    _slot(
        "garden patch scene",
        "simple ground",
        "one main flower with minimal leaves",
        "oblique",
        "full_body",
        "single",
        "scene",
    ),
    _slot(
        "bouquet pair",
        "plain background",
        "two stems together, still simple",
        "front",
        "full_body",
        "group",
    ),
)

DEFAULT_VARIATIONS: tuple[VariationSlot, ...] = (
    _slot(
        "close front portrait",
        "plain background",
        "centred subject close-up, clear silhouette",
        "front",
        "portrait",
        "close",
        "single",
    ),
    _slot(
        "side view",
        "plain background",
        "full side profile",
        "side",
        "full_body",
        "single",
    ),
    _slot(
        "oblique three-quarter",
        "simple setting",
        "angled toward the viewer",
        "oblique",
        "full_body",
        "single",
        "scene",
    ),
    _slot(
        "full body scene",
        "simple ground",
        "entire subject visible in a light scene",
        "front",
        "full_body",
        "single",
        "scene",
    ),
    _slot(
        "action pose",
        "open space",
        "subject mid-action, simple ground",
        "side",
        "full_body",
        "single",
        "action",
        "scene",
    ),
    _slot(
        "pair together",
        "plain background",
        "two subjects, both fully visible",
        "front",
        "full_body",
        "group",
    ),
)

CATEGORY_VARIATION_BANK: dict[str, tuple[VariationSlot, ...]] = {
    "dogs": ANIMAL_VARIATIONS,
    "cats": ANIMAL_VARIATIONS,
    "wildlife": ANIMAL_VARIATIONS,
    "animals": ANIMAL_VARIATIONS,
    "pets": ANIMAL_VARIATIONS,
    "farm animals": ANIMAL_VARIATIONS,
    "mammals": ANIMAL_VARIATIONS,
    "horses": HORSE_VARIATIONS,
    "birds": BIRD_VARIATIONS,
    "people": PERSON_VARIATIONS,
    "portraits": PERSON_VARIATIONS,
    "aircraft": AIRCRAFT_VARIATIONS,
    "cars": LAND_VEHICLE_VARIATIONS,
    "boats": WATER_VEHICLE_VARIATIONS,
    "flowers": FLOWER_VARIATIONS,
}

# Tag families used when scoring diversity for a bulk mix.
DIVERSITY_TAG_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"front", "side", "oblique", "top"}),
    frozenset({"portrait", "full_body", "close", "scene"}),
    frozenset({"sitting", "standing", "lying", "action"}),
    frozenset({"single", "group"}),
    frozenset({"on_ground", "takeoff", "airborne", "landing", "docked", "afloat", "moving"}),
)


def variation_bank_for_category(category: str | None) -> tuple[VariationSlot, ...]:
    if category and category in CATEGORY_VARIATION_BANK:
        return CATEGORY_VARIATION_BANK[category]
    return DEFAULT_VARIATIONS


def _tag_coverage(selected: list[VariationSlot]) -> set[str]:
    tags: set[str] = set()
    for slot in selected:
        tags |= set(slot.tags)
    return tags


def _novelty_score(candidate: VariationSlot, covered: set[str]) -> int:
    """How many diversity-relevant new tags this slot would add."""
    score = 0
    for group in DIVERSITY_TAG_GROUPS:
        cand = candidate.tags & group
        if not cand:
            continue
        if not (cand & covered):
            score += 2  # opens a new dimension value
        else:
            score += 0
        # Prefer unused tags inside the group.
        score += len(cand - covered)
    return score


def select_varied_slots(
    category: str | None,
    n: int,
    *,
    seed: int = 0,
    prefer_tags: set[str] | None = None,
) -> list[VariationSlot]:
    """Pick ``n`` slots with a mixture of interesting variations.

    Uses a deterministic greedy pass (seeded order) that maximises new
    viewpoint / pose / framing / grouping / vehicle-state tags, then fills
    remaining slots from the bank with unique aspect+scene keys.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    bank = list(variation_bank_for_category(category))
    if not bank:
        bank = list(DEFAULT_VARIATIONS)

    # Deterministic shuffle by seed without importing random.Random state globally.
    order = list(range(len(bank)))
    # LCG shuffle
    state = int(seed) & 0xFFFFFFFF
    for i in range(len(order) - 1, 0, -1):
        state = (1103515245 * state + 12345) & 0xFFFFFFFF
        j = state % (i + 1)
        order[i], order[j] = order[j], order[i]
    ordered = [bank[i] for i in order]

    preferred = prefer_tags or set()
    selected: list[VariationSlot] = []
    covered: set[str] = set()
    used_keys: set[tuple[str, str]] = set()

    def try_add(slot: VariationSlot) -> bool:
        key = (slot.aspect.lower(), slot.scene.lower())
        if key in used_keys:
            return False
        selected.append(slot)
        used_keys.add(key)
        covered.update(slot.tags)
        return True

    # Pass 1: greedily maximise novelty.
    remaining = list(ordered)
    while len(selected) < n and remaining:
        best_i = 0
        best_score = -1
        for i, slot in enumerate(remaining):
            key = (slot.aspect.lower(), slot.scene.lower())
            if key in used_keys:
                continue
            score = _novelty_score(slot, covered)
            if preferred:
                score += 3 * len(slot.tags & preferred)
            if score > best_score:
                best_score = score
                best_i = i
        slot = remaining.pop(best_i)
        try_add(slot)

    # Pass 2: if bank exhausted before n, wrap with scene variants.
    wrap = 1
    while len(selected) < n:
        base = bank[(len(selected)) % len(bank)]
        variant = VariationSlot(
            aspect=base.aspect,
            scene=f"{base.scene} variant {wrap}",
            composition=base.composition,
            tags=base.tags,
        )
        if not try_add(variant):
            wrap += 1
            if wrap > n + 5:
                break
            continue
        if len(selected) % len(bank) == 0:
            wrap += 1

    return selected[:n]


def bank_as_tuples(category: str | None) -> tuple[tuple[str, str, str], ...]:
    """Legacy (aspect, scene, composition) tuples for older callers."""
    return tuple(slot.as_tuple() for slot in variation_bank_for_category(category))
