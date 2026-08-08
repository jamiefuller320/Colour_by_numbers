"""Phase D set planner: one phrase → N aspect/scene plate slots."""

from __future__ import annotations

from dataclasses import dataclass

from .discover import (
    SubjectType,
    discover_subject_types,
    pick_subject_type,
)
from .illustrate import illustration_prompt

# Category-aware variety banks. Each entry is (aspect, scene, composition_hint).
# Keep scenes simple so format-brief subject-fill stays achievable.
CATEGORY_SLOT_BANK: dict[str, tuple[tuple[str, str, str], ...]] = {
    "aircraft": (
        ("side view", "clear sky", "full aeroplane silhouette, side profile"),
        ("three-quarter view", "airfield", "parked on tarmac, wheels visible"),
        ("front view", "runway", "nose and propeller facing the viewer"),
        ("in flight", "above soft clouds", "banking gently, wheels up"),
        ("takeoff", "runway", "wheels just leaving the ground"),
        ("hangar", "hangar doorway", "aeroplane framed in hangar opening"),
        ("landing", "runway approach", "low approach, landing gear down"),
        ("top view", "plain background", "plan-view silhouette, wings spread"),
    ),
    "dogs": (
        ("portrait", "plain background", "head and shoulders facing viewer"),
        ("sitting", "garden lawn", "full body sitting, tail visible"),
        ("standing side view", "park path", "full body side profile"),
        ("lying down", "rug", "relaxed dog lying facing viewer"),
        ("running", "open field", "dog in mid-stride, simple ground"),
        ("puppy pose", "plain background", "playful sit, oversized paws"),
        ("looking up", "kitchen floor", "head tilted upward"),
        ("three-quarter view", "yard", "standing three-quarter body view"),
    ),
    "cats": (
        ("portrait", "plain background", "head and shoulders facing viewer"),
        ("sitting", "windowsill", "cat sitting upright, full body"),
        ("side view", "garden", "standing side profile"),
        ("curled sleeping", "cushion", "cat curled asleep"),
        ("stretching", "plain background", "cat in long stretch pose"),
        ("crouching", "grass", "alert crouch, simple ground"),
        ("looking back", "plain background", "over-shoulder glance"),
        ("kitten pose", "blanket", "small kitten sitting"),
    ),
    "birds": (
        ("perched side view", "branch", "bird on simple branch, side profile"),
        ("portrait", "plain background", "head and chest facing viewer"),
        ("wings spread", "clear sky", "bird gliding, wings open"),
        ("taking off", "branch", "wings raised leaving perch"),
        ("standing", "ground", "full body standing bird"),
        ("three-quarter view", "plain background", "body angled toward viewer"),
        ("nesting", "nest", "bird on simple nest"),
        ("in flight", "sky", "side flying silhouette"),
    ),
    "flowers": (
        ("centred portrait", "plain background", "single bloom filling the frame"),
        ("side view", "plain background", "bloom and stem from the side"),
        ("bud and bloom", "plain background", "open flower beside a bud"),
        ("top view", "plain background", "looking down into the bloom"),
        ("in a pot", "simple pot", "potted plant, centred"),
        ("bouquet pair", "plain background", "two stems, still simple"),
        ("close petal study", "plain background", "large centred bloom"),
        ("garden patch", "simple ground", "one main flower, minimal leaves"),
    ),
    "cars": (
        ("side view", "road", "full car silhouette, side profile"),
        ("three-quarter front", "driveway", "front and side visible"),
        ("front view", "plain background", "headlights facing viewer"),
        ("rear three-quarter", "street", "rear and side visible"),
        ("parked", "garage", "car beside simple garage"),
        ("on the road", "open road", "driving scene, simple horizon"),
        ("top view", "plain background", "plan-view car silhouette"),
        ("classic pose", "showroom", "centred three-quarter display"),
    ),
    "boats": (
        ("side view", "calm water", "full boat silhouette on water"),
        ("three-quarter view", "harbour", "boat angled toward viewer"),
        ("front view", "water", "bow facing viewer"),
        ("sailing", "open sea", "under sail, simple waves"),
        ("docked", "pier", "tied at a simple pier"),
        ("top view", "plain water", "plan-view boat silhouette"),
        ("rowing", "lake", "small boat with oars"),
        ("at anchor", "bay", "boat floating, simple shoreline"),
    ),
}

DEFAULT_SLOT_BANK: tuple[tuple[str, str, str], ...] = (
    ("portrait", "plain background", "centred subject, clear silhouette"),
    ("side view", "plain background", "full side profile"),
    ("three-quarter view", "simple setting", "angled toward the viewer"),
    ("action pose", "open space", "subject mid-action, simple ground"),
    ("close-up", "plain background", "head or main feature fills frame"),
    ("full body", "simple ground", "entire subject visible"),
    ("looking left", "plain background", "subject facing left"),
    ("looking right", "plain background", "subject facing right"),
)


@dataclass(frozen=True)
class PlateSlot:
    """One planned plate in a colouring set."""

    index: int
    aspect: str
    scene: str
    composition: str
    seed: int
    prompt: str
    # Optional overrides for mixed-theme sets (defaults to the plan subject).
    subject_label: str | None = None
    category: str | None = None

    @property
    def slug(self) -> str:
        subject_bit = f"{self.subject_label}-" if self.subject_label else ""
        raw = f"{self.index:02d}-{subject_bit}{self.aspect}-{self.scene}"
        cleaned = "".join(ch if ch.isalnum() else "-" for ch in raw.lower())
        while "--" in cleaned:
            cleaned = cleaned.replace("--", "-")
        return cleaned.strip("-")[:60]


@dataclass(frozen=True)
class SetPlan:
    """Deterministic plan for N varied plates of one subject."""

    original_query: str
    subject_type: SubjectType
    slots: tuple[PlateSlot, ...]
    style_notes: str = (
        "shared colouring-book style, same subject identity, "
        "flat fills, bold outlines, white or pale background"
    )
    mode: str = "single"  # single | mixed (mixed plans carry per-slot subjects)

    @property
    def n_plates(self) -> int:
        return len(self.slots)

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "subject_label": self.subject_type.label,
            "category": self.subject_type.category,
            "mode": self.mode,
            "style_notes": self.style_notes,
            "slots": [
                {
                    "index": slot.index,
                    "aspect": slot.aspect,
                    "scene": slot.scene,
                    "composition": slot.composition,
                    "seed": slot.seed,
                    "slug": slot.slug,
                    "prompt": slot.prompt,
                    **(
                        {
                            "subject_label": slot.subject_label,
                            "category": slot.category,
                        }
                        if slot.subject_label
                        else {}
                    ),
                }
                for slot in self.slots
            ],
        }


def slot_bank_for_category(category: str | None) -> tuple[tuple[str, str, str], ...]:
    if category and category in CATEGORY_SLOT_BANK:
        return CATEGORY_SLOT_BANK[category]
    return DEFAULT_SLOT_BANK


def compose_slot_prompt(
    subject_type: SubjectType,
    *,
    aspect: str,
    scene: str,
    composition: str,
    style_preset: str | None = None,
) -> str:
    """Base illustration prompt plus unique aspect/scene cues."""
    base = illustration_prompt(
        subject_type.label,
        category=subject_type.category,
        style_preset=style_preset,
    )
    extras = (
        f"aspect: {aspect}, scene: {scene}, {composition}, "
        "same subject identity, distinct pose from other pages in the set, "
        "subject fills most of the frame"
    )
    return f"{base}, {extras}"


def plan_colouring_set(
    query: str,
    *,
    subject_type: str | None = None,
    type_pick: int = 0,
    n_plates: int = 6,
    base_seed: int = 0,
    discover_types: bool = True,
    custom_slots: list[tuple[str, str, str]] | None = None,
    style: str | None = None,
) -> SetPlan:
    """Turn a keyword/phrase into N unique aspect/scene plate slots.

    Discovery runs once; each slot reuses the same subject identity with a
    distinct aspect/scene prompt and seed.
    """
    if n_plates < 1:
        raise ValueError("n_plates must be >= 1")

    discovery = discover_subject_types(
        query,
        probe_search=discover_types and subject_type is None,
    )
    chosen = pick_subject_type(discovery, type_name=subject_type, pick=type_pick)
    bank = list(custom_slots) if custom_slots else list(slot_bank_for_category(chosen.category))
    if not bank:
        bank = list(DEFAULT_SLOT_BANK)

    slots: list[PlateSlot] = []
    for i in range(n_plates):
        aspect, scene, composition = bank[i % len(bank)]
        # If we wrap the bank, nudge the scene label so prompts stay unique.
        if i >= len(bank):
            scene = f"{scene} variant {(i // len(bank)) + 1}"
        seed = int(base_seed) + i
        prompt = compose_slot_prompt(
            chosen,
            aspect=aspect,
            scene=scene,
            composition=composition,
            style_preset=style,
        )
        slots.append(
            PlateSlot(
                index=i + 1,
                aspect=aspect,
                scene=scene,
                composition=composition,
                seed=seed,
                prompt=prompt,
            )
        )

    # Guarantee unique (aspect, scene) pairs in the plan.
    seen: set[tuple[str, str]] = set()
    for slot in slots:
        key = (slot.aspect.lower(), slot.scene.lower())
        if key in seen:
            raise RuntimeError(f"Duplicate slot planned: {key}")
        seen.add(key)

    return SetPlan(
        original_query=query,
        subject_type=chosen,
        slots=tuple(slots),
        mode="single",
    )


def plan_mixed_colouring_set(
    entries: list[tuple[str, str | None]],
    *,
    plates_per_subject: int = 2,
    base_seed: int = 0,
    discover_types: bool = True,
    style: str | None = None,
) -> SetPlan:
    """Plan a mixed-theme set from ``(query, optional_type)`` entries.

    Each subject contributes ``plates_per_subject`` aspect/scene slots. The
    plan's top-level ``subject_type`` is the first entry (for compatibility);
    per-slot ``subject_label`` / ``category`` carry the mixed identity.
    """
    if not entries:
        raise ValueError("entries must be non-empty")
    if plates_per_subject < 1:
        raise ValueError("plates_per_subject must be >= 1")

    slots: list[PlateSlot] = []
    first_chosen: SubjectType | None = None
    index = 1
    for entry_i, (query, type_name) in enumerate(entries):
        discovery = discover_subject_types(
            query,
            probe_search=discover_types and type_name is None,
        )
        chosen = pick_subject_type(discovery, type_name=type_name, pick=0)
        if first_chosen is None:
            first_chosen = chosen
        bank = list(slot_bank_for_category(chosen.category)) or list(DEFAULT_SLOT_BANK)
        for j in range(plates_per_subject):
            aspect, scene, composition = bank[j % len(bank)]
            if j >= len(bank):
                scene = f"{scene} variant {(j // len(bank)) + 1}"
            seed = int(base_seed) + entry_i * 100 + j
            prompt = compose_slot_prompt(
                chosen,
                aspect=aspect,
                scene=scene,
                composition=composition,
                style_preset=style,
            )
            slots.append(
                PlateSlot(
                    index=index,
                    aspect=aspect,
                    scene=scene,
                    composition=composition,
                    seed=seed,
                    prompt=prompt,
                    subject_label=chosen.label,
                    category=chosen.category,
                )
            )
            index += 1

    assert first_chosen is not None
    queries = ", ".join(q for q, _ in entries)
    return SetPlan(
        original_query=queries,
        subject_type=first_chosen,
        slots=tuple(slots),
        mode="mixed",
        style_notes=(
            "mixed-theme colouring set; each slot keeps its own subject identity; "
            "shared house style, flat fills, bold outlines"
        ),
    )
