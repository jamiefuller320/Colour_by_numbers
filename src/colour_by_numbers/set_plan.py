"""Phase D set planner: one phrase → N varied aspect/scene plate slots."""

from __future__ import annotations

from dataclasses import dataclass

from .discover import (
    SubjectType,
    discover_subject_types,
    pick_subject_type,
)
from .illustrate import illustration_prompt
from .variation_banks import (
    VariationSlot,
    bank_as_tuples,
    select_varied_slots,
    variation_bank_for_category,
)

# Back-compat alias: older code/tests import CATEGORY_SLOT_BANK / DEFAULT_SLOT_BANK.
CATEGORY_SLOT_BANK: dict[str, tuple[tuple[str, str, str], ...]] = {
    key: bank_as_tuples(key)
    for key in (
        "aircraft",
        "dogs",
        "cats",
        "birds",
        "flowers",
        "cars",
        "boats",
        "horses",
        "people",
        "portraits",
    )
}

DEFAULT_SLOT_BANK: tuple[tuple[str, str, str], ...] = bank_as_tuples(None)


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
    tags: tuple[str, ...] = ()

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
                    "tags": list(slot.tags),
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
    return bank_as_tuples(category)


def _expand_bank(bank: list[VariationSlot], n: int) -> list[VariationSlot]:
    """Walk a bank in order, wrapping with scene variants when n exceeds length."""
    if not bank:
        raise ValueError("variation bank is empty")
    picked: list[VariationSlot] = []
    for i in range(n):
        base = bank[i % len(bank)]
        scene = base.scene
        if i >= len(bank):
            scene = f"{scene} variant {(i // len(bank)) + 1}"
        picked.append(
            VariationSlot(
                aspect=base.aspect,
                scene=scene,
                composition=base.composition,
                tags=base.tags,
            )
        )
    return picked


def compose_slot_prompt(
    subject_type: SubjectType,
    *,
    aspect: str,
    scene: str,
    composition: str,
    style_preset: str | None = None,
    tags: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Base illustration prompt plus unique aspect/scene cues."""
    base = illustration_prompt(
        subject_type.label,
        category=subject_type.category,
        style_preset=style_preset,
    )
    tag_bit = ""
    if tags:
        tag_bit = f", variation tags: {', '.join(tags)}"
    extras = (
        f"aspect: {aspect}, scene: {scene}, {composition}{tag_bit}, "
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
    variety: str = "balanced",
) -> SetPlan:
    """Turn a keyword/phrase into N unique aspect/scene plate slots.

    ``variety``:
      - ``balanced`` (default): greedy mix across viewpoint / pose / framing /
        grouping / vehicle-state tags for interesting bulk sets
      - ``sequential``: walk the category bank in order (legacy behaviour)
    """
    if n_plates < 1:
        raise ValueError("n_plates must be >= 1")

    discovery = discover_subject_types(
        query,
        probe_search=discover_types and subject_type is None,
    )
    chosen = pick_subject_type(discovery, type_name=subject_type, pick=type_pick)

    if custom_slots is not None:
        bank = [
            VariationSlot(a, s, c, frozenset())
            for a, s, c in custom_slots
        ]
        picked = _expand_bank(bank, n_plates)
    elif (variety or "balanced").lower().strip() == "sequential":
        picked = _expand_bank(
            list(variation_bank_for_category(chosen.category)), n_plates
        )
    else:
        picked = select_varied_slots(
            chosen.category, n_plates, seed=int(base_seed)
        )

    slots: list[PlateSlot] = []
    for i, var in enumerate(picked):
        tags = tuple(sorted(var.tags))
        seed = int(base_seed) + i
        prompt = compose_slot_prompt(
            chosen,
            aspect=var.aspect,
            scene=var.scene,
            composition=var.composition,
            style_preset=style,
            tags=tags,
        )
        slots.append(
            PlateSlot(
                index=i + 1,
                aspect=var.aspect,
                scene=var.scene,
                composition=var.composition,
                seed=seed,
                prompt=prompt,
                tags=tags,
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
        style_notes=(
            "shared colouring-book style, same subject identity, "
            "varied aspects/poses/scenes for a bulk themed set, "
            "flat fills, bold outlines"
        ),
    )


def plan_mixed_colouring_set(
    entries: list[tuple[str, str | None]],
    *,
    plates_per_subject: int = 2,
    base_seed: int = 0,
    discover_types: bool = True,
    style: str | None = None,
    variety: str = "balanced",
) -> SetPlan:
    """Plan a mixed-theme set from ``(query, optional_type)`` entries.

    Each subject contributes ``plates_per_subject`` varied aspect/scene slots.
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
        if (variety or "balanced").lower().strip() == "sequential":
            picked = _expand_bank(
                list(variation_bank_for_category(chosen.category)),
                plates_per_subject,
            )
        else:
            picked = select_varied_slots(
                chosen.category,
                plates_per_subject,
                seed=int(base_seed) + entry_i * 17,
            )
        for j, var in enumerate(picked):
            tags = tuple(sorted(var.tags))
            seed = int(base_seed) + entry_i * 100 + j
            prompt = compose_slot_prompt(
                chosen,
                aspect=var.aspect,
                scene=var.scene,
                composition=var.composition,
                style_preset=style,
                tags=tags,
            )
            slots.append(
                PlateSlot(
                    index=index,
                    aspect=var.aspect,
                    scene=var.scene,
                    composition=var.composition,
                    seed=seed,
                    prompt=prompt,
                    subject_label=chosen.label,
                    category=chosen.category,
                    tags=tags,
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
            "varied category-appropriate aspects/poses; shared house style"
        ),
    )
