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

# Default prompt hints applied when a tag is frequent in collated critiques.
TAG_PROMPT_HINTS: dict[str, str] = {
    "nose_detail": (
        "clearly defined nose with visible nostrils and muzzle wrinkles, "
        "nose as separate colour regions"
    ),
    "eyes": (
        "large expressive eyes with separate dark pupils and lighter iris or "
        "sclera fills (at least two colour regions per eye)"
    ),
    "mouth_detail": "defined mouth and muzzle line with separate colour fills",
    "ears": "clear ear shape and inner-ear colour separation",
    "outline": "bold clean black outlines around every colour region",
    "colours": "distinct flat colour blocks with clear value steps between neighbours",
    "proportions": "accurate breed proportions and centred portrait composition",
    "background": "plain white or pale background, strong subject contrast",
    "missing_detail": "include all diagnostic breed or subject features as colour blocks",
    "too_simple": "preserve characteristic wrinkles, markings, and facial features",
    "wrong_subject": "unmistakable subject silhouette, no person or wrong entity",
    "composition": "subject fills most of the page, centred portrait",
}


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

    for critique in failures:
        by_category[critique.category or "unknown"] += 1
        for tag in critique.issues:
            by_tag[tag] += 1
            if critique.notes:
                tag_notes[(critique.category, tag)].append(critique.notes)
            if critique.suggested_prompt:
                tag_notes[(critique.category, tag)].append(critique.suggested_prompt)

    lessons: list[CollatedLesson] = []
    for (category, tag), notes in sorted(tag_notes.items()):
        count = by_tag[tag]
        if count < min_count:
            continue
        hint = TAG_PROMPT_HINTS.get(tag, "")
        lessons.append(
            CollatedLesson(
                category=category,
                tag=tag,
                count=count,
                examples=notes[:5],
                prompt_hint=hint,
            )
        )

    global_hints: list[str] = []
    for tag, count in by_tag.most_common():
        if count < min_count:
            continue
        hint = TAG_PROMPT_HINTS.get(tag, "")
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
    """Return prompt hints from collated plate lessons for a category."""
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
        if cat and str(lesson.get("category", "")).lower() not in {"", cat}:
            continue
        hint = str(lesson.get("prompt_hint", "")).strip()
        if hint and hint not in hints:
            hints.append(hint)
        if len(hints) >= limit:
            break

    if not hints:
        for hint in data.get("global_hints", []):
            text = str(hint).strip()
            if text and text not in hints:
                hints.append(text)
            if len(hints) >= limit:
                break
    return hints


def seed_prompt_with_plate_lessons(
    prompt: str,
    *,
    category: str | None,
    path: Path | str | None = None,
) -> tuple[str, list[str]]:
    """Append collated plate-lesson hints not already present in the prompt."""
    prompt_l = prompt.lower()
    applied: list[str] = []
    for hint in load_plate_lessons(category=category, path=path):
        if hint.lower() in prompt_l:
            continue
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
