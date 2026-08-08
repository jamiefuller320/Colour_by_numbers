"""Plate-level human critique storage and collation for prompt refinement.

Complements :mod:`feedback` (subject recognition) with fine-grained quality
feedback: missing nose detail, weak eyes, muddy colours, etc. Critiques are
stored as JSONL rows and aggregated into reusable prompt additions per category
and issue tag.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CRITIQUES_PATH = Path("data/plate_critiques.jsonl")
DEFAULT_LESSONS_PATH = Path("data/plate_lessons.json")

# Standard issue tags shown in the Pages review UI.
PLATE_ISSUE_TAGS: dict[str, str] = {
    "nose_detail": "Nose / muzzle lacks nostrils, wrinkles, or definition",
    "eyes": "Eyes lack definition or need more colour separation",
    "mouth_detail": "Mouth / lips / tongue detail missing or wrong",
    "ears": "Ear shape or placement wrong",
    "outline": "Outline too thin, thick, or broken",
    "colours": "Palette too flat, muddy, or too few regions",
    "proportions": "Head/body proportions or pose wrong",
    "background": "Background too busy or poor contrast",
    "missing_detail": "Important feature missing (generic)",
    "too_simple": "Over-simplified — lost character",
    "wrong_subject": "Wrong subject or unrecognisable",
    "composition": "Framing / cropping / centre of page",
    "other": "Other (describe in notes)",
}

# Tags that only make sense for animal / people faces. Never become global
# hints applied to boats, aircraft, cars, flowers, etc.
MAMMAL_FACE_TAGS = frozenset({"nose_detail", "mouth_detail", "ears"})
ANIMAL_OR_PEOPLE_TAGS = frozenset({"eyes", "proportions", *MAMMAL_FACE_TAGS})
VEHICLE_CATEGORIES = frozenset({"aircraft", "cars", "boats", "vehicles", "trains"})
FLOWER_CATEGORIES = frozenset({"flowers", "plants"})
PEOPLE_CATEGORIES = frozenset({"people", "portraits"})
MAMMAL_CATEGORIES = frozenset(
    {
        "dogs",
        "cats",
        "horses",
        "wildlife",
        "animals",
        "pets",
        "farm animals",
        "mammals",
    }
)
BIRD_CATEGORIES = frozenset({"birds"})

# Default prompt hints applied when a tag is frequent in collated critiques.
TAG_PROMPT_HINTS: dict[str, str] = {
    "nose_detail": (
        "clearly defined nose with visible nostrils and muzzle wrinkles, "
        "nose as separate colour regions"
    ),
    "eyes": (
        "both eyes matching, each with separate dark pupil and lighter iris "
        "or sclera fills distinct from surrounding fur or feathers"
    ),
    "mouth_detail": "defined mouth and muzzle line with separate colour fills",
    "ears": "clear ear shape and inner-ear colour separation",
    "outline": (
        "bold clean black outlines with smooth region boundaries around "
        "every colour block"
    ),
    "colours": (
        "distinct flat colour blocks with clear value steps between neighbouring "
        "parts, prefer 12–16 colours"
    ),
    "proportions": "accurate breed proportions and recognisable pose",
    "background": "plain white or pale background, strong subject contrast",
    "missing_detail": "include all diagnostic breed or subject features as colour blocks",
    "too_simple": (
        "enough distinct colour regions to show form and depth; preserve "
        "characteristic markings and structural parts"
    ),
    "wrong_subject": (
        "unmistakable subject with breed- or species-accurate colours and "
        "silhouette, no person or wrong entity"
    ),
    "composition": (
        "full subject in frame with a small margin, centred, not over-cropped "
        "or over-enlarged"
    ),
    "other": "smooth colour-region boundaries; clear value separation for depth",
}


def tag_applies_to_category(tag: str, category: str | None) -> bool:
    """Return False for animal-face tags on vehicles/flowers/etc."""
    cat = (category or "").strip().lower()
    if tag in MAMMAL_FACE_TAGS:
        return cat in MAMMAL_CATEGORIES or cat in PEOPLE_CATEGORIES
    if tag == "eyes":
        return (
            cat in MAMMAL_CATEGORIES
            or cat in BIRD_CATEGORIES
            or cat in PEOPLE_CATEGORIES
            or not cat
        )
    if tag == "proportions":
        return cat not in VEHICLE_CATEGORIES and cat not in FLOWER_CATEGORIES
    return True


def prompt_hint_for_tag(tag: str, category: str | None = None) -> str:
    """Category-aware prompt hint for an issue tag."""
    cat = (category or "").strip().lower()
    if tag == "too_simple":
        if cat in VEHICLE_CATEGORIES:
            return (
                "separate colour regions for body panels, windows, wheels or "
                "wings, and other structural parts — not a single flat fill"
            )
        if cat in FLOWER_CATEGORIES:
            return (
                "show petals, centre disk, stem and leaves as separate colour "
                "regions with species-typical colours"
            )
        if cat in BIRD_CATEGORIES:
            return (
                "species-accurate plumage colour blocks (e.g. robin red breast), "
                "defined beak, matching eyes"
            )
        if cat in MAMMAL_CATEGORIES:
            return (
                "preserve characteristic markings and facial features; separate "
                "value steps between head, neck and body"
            )
    if tag == "wrong_subject":
        if cat in BIRD_CATEGORIES:
            return (
                "species-accurate bird colours and markings, clear beak "
                "(not a mammal nose), unmistakable silhouette"
            )
        if cat in FLOWER_CATEGORIES:
            return (
                "recognisable whole flower with petals and centre visible, "
                "species-typical colours, not an abstract crop"
            )
        if cat in MAMMAL_CATEGORIES:
            return (
                "unmistakable animal of the correct species/breed with "
                "accurate colours and silhouette"
            )
    if tag == "colours":
        if cat in MAMMAL_CATEGORIES:
            return (
                "distinct value steps for depth between head, neck and body; "
                "eyes and nose as separate colours from surrounding fur"
            )
        if cat in VEHICLE_CATEGORIES:
            return (
                "enough distinct colour regions for body, windows, trim and "
                "other parts — avoid a single flat fill"
            )
    if tag == "eyes" and cat in BIRD_CATEGORIES:
        return (
            "both eyes matching, each with dark pupil and lighter iris "
            "distinct from surrounding feathers"
        )
    if not tag_applies_to_category(tag, category):
        return ""
    return TAG_PROMPT_HINTS.get(tag, "")


@dataclass(frozen=True)
class PlateCritique:
    """One human review of a generated plate."""

    plate_id: str
    category: str
    subject: str
    rating: str  # pass | fail | needs_work
    issues: tuple[str, ...] = ()
    notes: str = ""
    suggested_prompt: str = ""
    reviewer: str = ""
    reviewed_at: str = ""
    prompt_used: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CollatedLesson:
    """Aggregated guidance for a category + issue tag."""

    category: str
    tag: str
    count: int
    examples: list[str] = field(default_factory=list)
    prompt_hint: str = ""


@dataclass
class CollationReport:
    """Summary after merging critiques."""

    total: int
    by_category: dict[str, int]
    by_tag: dict[str, int]
    lessons: list[CollatedLesson]
    global_hints: list[str]


def critiques_path(path: Path | str | None = None) -> Path:
    return Path(path) if path is not None else DEFAULT_CRITIQUES_PATH


def lessons_store_path(path: Path | str | None = None) -> Path:
    return Path(path) if path is not None else DEFAULT_LESSONS_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalise_rating(value: str) -> str:
    raw = (value or "").strip().lower()
    if raw in {"pass", "ok", "good", "yes"}:
        return "pass"
    if raw in {"fail", "bad", "no", "reject"}:
        return "fail"
    return "needs_work"


def normalise_issues(issues: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not issues:
        return ()
    out: list[str] = []
    for item in issues:
        tag = str(item).strip().lower().replace(" ", "_")
        if tag in PLATE_ISSUE_TAGS and tag not in out:
            out.append(tag)
    return tuple(out)


def record_plate_critique(
    critique: PlateCritique,
    *,
    path: Path | str | None = None,
    append: bool = True,
) -> None:
    """Append one critique row to the JSONL store."""
    store = critiques_path(path)
    store.parent.mkdir(parents=True, exist_ok=True)
    row = critique.to_dict()
    if not row.get("reviewed_at"):
        row["reviewed_at"] = _now_iso()
    mode = "a" if append else "w"
    with store.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_critiques(
    *,
    path: Path | str | None = None,
    category: str | None = None,
    subject: str | None = None,
    plate_id: str | None = None,
) -> list[PlateCritique]:
    store = critiques_path(path)
    if not store.is_file():
        return []
    rows: list[PlateCritique] = []
    for line in store.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if plate_id and data.get("plate_id") != plate_id:
            continue
        if category and str(data.get("category", "")).lower() != category.lower():
            continue
        if subject and str(data.get("subject", "")).lower() != subject.lower():
            continue
        rows.append(
            PlateCritique(
                plate_id=str(data.get("plate_id", "")),
                category=str(data.get("category", "")),
                subject=str(data.get("subject", "")),
                rating=normalise_rating(str(data.get("rating", "needs_work"))),
                issues=normalise_issues(data.get("issues")),
                notes=str(data.get("notes", "")),
                suggested_prompt=str(data.get("suggested_prompt", "")),
                reviewer=str(data.get("reviewer", "")),
                reviewed_at=str(data.get("reviewed_at", "")),
                prompt_used=str(data.get("prompt_used", "")),
            )
        )
    return rows


def import_critiques_json(
    payload: dict | list,
    *,
    path: Path | str | None = None,
    dedupe: bool = True,
) -> int:
    """Import critiques exported from the Pages review UI."""
    items = payload if isinstance(payload, list) else payload.get("critiques", [])
    if not isinstance(items, list):
        raise ValueError("Expected a list of critiques or {critiques: [...]}")

    existing: set[tuple[str, str]] = set()
    if dedupe:
        for row in load_critiques(path=path):
            existing.add((row.plate_id, row.reviewed_at))

    written = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        plate_id = str(item.get("plate_id", "")).strip()
        if not plate_id:
            continue
        reviewed_at = str(item.get("reviewed_at", "")).strip() or _now_iso()
        if dedupe and (plate_id, reviewed_at) in existing:
            continue
        critique = PlateCritique(
            plate_id=plate_id,
            category=str(item.get("category", "")),
            subject=str(item.get("subject", "")),
            rating=normalise_rating(str(item.get("rating", "needs_work"))),
            issues=normalise_issues(item.get("issues")),
            notes=str(item.get("notes", "")).strip(),
            suggested_prompt=str(item.get("suggested_prompt", "")).strip(),
            reviewer=str(item.get("reviewer", "")).strip(),
            reviewed_at=reviewed_at,
            prompt_used=str(item.get("prompt_used", "")).strip(),
        )
        record_plate_critique(critique, path=path)
        existing.add((plate_id, reviewed_at))
        written += 1
    return written


def collate_critiques(
    critiques: list[PlateCritique],
    *,
    min_count: int = 1,
) -> CollationReport:
    """Group critiques into category/tag lessons and global prompt hints."""
    failures = [c for c in critiques if c.rating != "pass"]
    by_category: Counter[str] = Counter()
    by_tag: Counter[str] = Counter()
    tag_notes: dict[tuple[str, str], list[str]] = defaultdict(list)
    pair_counts: Counter[tuple[str, str]] = Counter()

    for critique in failures:
        category = critique.category or "unknown"
        by_category[category] += 1
        for tag in critique.issues:
            by_tag[tag] += 1
            pair_counts[(category, tag)] += 1
            if critique.notes:
                tag_notes[(category, tag)].append(critique.notes)
            if critique.suggested_prompt:
                tag_notes[(category, tag)].append(critique.suggested_prompt)

    lessons: list[CollatedLesson] = []
    for (category, tag), count in sorted(pair_counts.items()):
        if count < min_count:
            continue
        hint = prompt_hint_for_tag(tag, category)
        notes = tag_notes.get((category, tag), [])
        if not hint and not notes:
            continue
        lessons.append(
            CollatedLesson(
                category=category,
                tag=tag,
                count=count,
                examples=notes[:5],
                prompt_hint=hint,
            )
        )

    # Only universal tags become global — never mammal nose/muzzle cues.
    global_hints: list[str] = []
    for tag, count in by_tag.most_common():
        if count < min_count:
            continue
        if tag in ANIMAL_OR_PEOPLE_TAGS:
            continue
        hint = prompt_hint_for_tag(tag, category=None)
        if hint and hint not in global_hints:
            global_hints.append(hint)

    return CollationReport(
        total=len(critiques),
        by_category=dict(by_category),
        by_tag=dict(by_tag),
        lessons=lessons,
        global_hints=global_hints,
    )


def write_lessons_json(
    report: CollationReport,
    *,
    path: Path | str | None = None,
) -> Path:
    """Persist collated lessons for scripts and prompt seeding."""
    store = lessons_store_path(path)
    store.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _now_iso(),
        "total_critiques": report.total,
        "by_category": report.by_category,
        "by_tag": report.by_tag,
        "global_hints": report.global_hints,
        "lessons": [
            {
                "category": lesson.category,
                "tag": lesson.tag,
                "count": lesson.count,
                "examples": lesson.examples,
                "prompt_hint": lesson.prompt_hint,
            }
            for lesson in report.lessons
        ],
    }
    store.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return store


def load_plate_lessons(
    *,
    category: str | None = None,
    path: Path | str | None = None,
    limit: int = 8,
) -> list[str]:
    """Return prompt hints from collated plate lessons for a category.

    Category-specific lessons are preferred. Universal ``global_hints`` are
    added only when they still apply to the category (animal-face cues never
    leak onto vehicles or flowers).
    """
    store = lessons_store_path(path)
    if not store.is_file():
        return []
    try:
        data = json.loads(store.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    hints: list[str] = []
    cat = (category or "").strip().lower()

    for lesson in data.get("lessons", []):
        lesson_cat = str(lesson.get("category", "")).lower()
        if cat and lesson_cat not in {"", cat}:
            continue
        tag = str(lesson.get("tag", "")).strip().lower()
        if tag and not tag_applies_to_category(tag, cat or lesson_cat):
            continue
        hint = str(lesson.get("prompt_hint", "")).strip()
        if not hint:
            # Fall back to a fresh category-aware hint for the tag.
            hint = prompt_hint_for_tag(tag, cat or lesson_cat)
        if hint and hint not in hints:
            hints.append(hint)
        if len(hints) >= limit:
            return hints

    for hint in data.get("global_hints", []):
        text = str(hint).strip()
        if not text:
            continue
        # Reject legacy animal-face globals that polluted vehicle prompts.
        lower = text.lower()
        if any(
            cue in lower
            for cue in ("nostril", "muzzle", "nose as separate", "inner-ear")
        ) and cat and cat not in MAMMAL_CATEGORIES and cat not in PEOPLE_CATEGORIES:
            continue
        if text not in hints:
            hints.append(text)
        if len(hints) >= limit:
            break
    return hints


def seed_prompt_with_plate_lessons(
    prompt: str,
    *,
    category: str | None,
    path: Path | str | None = None,
    style_preset: str | None = None,
) -> tuple[str, list[str]]:
    """Append collated plate-lesson hints not already present in the prompt.

    Skips hints that fight the active style or a set composition lock (e.g.
    ``prefer 12–16 colours`` against vibrant, or extra face-crop cues after a
    FULL BODY lock — those were burying pose variation for fal/Flux).
    """
    prompt_l = prompt.lower()
    style = (style_preset or "").lower().strip()
    locked = prompt_l.startswith("composition")
    vibrant = style == "vibrant" or "vibrant paint-by-numbers" in prompt_l
    full_body = (
        "composition lock" in prompt_l
        or "full body" in prompt_l
        or "entire body" in prompt_l
        or "head to paws" in prompt_l
    )
    applied: list[str] = []
    for hint in load_plate_lessons(category=category, path=path):
        hint_l = hint.lower()
        if hint_l in prompt_l:
            continue
        if vibrant and (
            "12–16" in hint_l
            or "12-16" in hint_l
            or "prefer 12" in hint_l
        ):
            continue
        if full_body and any(
            cue in hint_l
            for cue in (
                "both eyes matching",
                "nostril",
                "muzzle",
                "centred, not over-cropped",
                "over-enlarged",
                "facial features",
            )
        ):
            continue
        # Set prompts already carry identity + style; keep lesson appends tiny.
        if locked and len(applied) >= 2:
            break
        prompt = f"{prompt}, {hint}"
        applied.append(hint)
        prompt_l = prompt.lower()
    prompt = re.sub(r"\s+", " ", prompt)
    prompt = re.sub(r"(,\s*){2,}", ", ", prompt)
    return prompt.strip(" ,"), applied


def format_collation_report(report: CollationReport) -> str:
    lines = [
        f"Plate critiques: {report.total} total",
        f"Failures by category: {report.by_category or '—'}",
        f"Issues by tag: {report.by_tag or '—'}",
        "",
        "Suggested prompt additions:",
    ]
    if report.global_hints:
        for hint in report.global_hints:
            lines.append(f"  • {hint}")
    else:
        lines.append("  (none)")
    if report.lessons:
        lines.append("")
        lines.append("Per-category lessons:")
        for lesson in report.lessons:
            lines.append(
                f"  [{lesson.category}] {lesson.tag} (×{lesson.count}): {lesson.prompt_hint}"
            )
            for example in lesson.examples[:2]:
                lines.append(f"      e.g. {example[:120]}")
    return "\n".join(lines)
