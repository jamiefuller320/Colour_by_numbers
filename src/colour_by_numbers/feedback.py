"""Subject-recognition feedback loop: critique → revise → retry → learn.

Answers two questions after each generation:

1. Is this recognisable as the requested subject?
2. How should the generation prompt improve?

Critics (``mode``):
- ``rules`` — cheap, offline category/feature heuristics (always available)
- ``openai`` — vision model via ``OPENAI_API_KEY`` (best automatic critic)
- ``human`` — interactive CLI yes/no + free-text advice

Lessons from accepted revisions are appended to a JSONL store and replayed
into future prompts for the same subject/category.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from .discover import (
    CATEGORY_NEGATIVE_CUES,
    disambiguate_subject_label,
    subject_kind_frame,
)

logger = logging.getLogger(__name__)

DEFAULT_LESSONS_PATH = Path("data/subject_lessons.jsonl")
RECOGNITION_PASS_MIN = 0.7

# Distinctive visual cues used when revising weak prompts (offline rules critic).
SUBJECT_FEATURE_CUES: dict[str, dict[str, str]] = {
    "aircraft": {
        "spitfire": (
            "Supermarine Spitfire WWII fighter aeroplane, elliptical wings, "
            "long pointed nose, single three-blade propeller, bubble canopy, "
            "machine-gun ports on wings, RAF fighter silhouette, side view, "
            "aircraft only"
        ),
        "biplane": (
            "vintage biplane aeroplane with two wings stacked, struts and "
            "rigging wires, propeller, open cockpit, side view"
        ),
        "concorde": (
            "Concorde supersonic airliner, slender delta wings, droop nose, "
            "four engines under wings, side view airliner only"
        ),
        "helicopter": (
            "helicopter aircraft with main rotor and tail rotor, skids or "
            "wheels, cockpit bubble, side view"
        ),
    },
    "dogs": {
        "pug": (
            "pug dog, wrinkled face, defined nose with visible nostrils and muzzle "
            "wrinkles, black face mask, curled tail, compact square body, large "
            "round eyes, short muzzle"
        ),
        "dachshund": (
            "dachshund dog, very long body, short legs, long snout, "
            "recognisable sausage-dog silhouette"
        ),
    },
}


@dataclass(frozen=True)
class SubjectCritique:
    """Structured answer to the recognition questions."""

    recognisable: bool
    confidence: float
    issues: tuple[str, ...] = ()
    improvements: tuple[str, ...] = ()
    mode: str = "rules"
    raw: str = ""

    @property
    def passed(self) -> bool:
        return self.recognisable and self.confidence >= RECOGNITION_PASS_MIN


@dataclass(frozen=True)
class FeedbackAttempt:
    """One generate → critique cycle."""

    prompt: str
    critique: SubjectCritique
    accepted: bool


@dataclass
class FeedbackLoopResult:
    """Outcome of the recognition feedback loop."""

    image: Image.Image
    prompt: str
    attempts: list[FeedbackAttempt] = field(default_factory=list)
    lessons_used: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.attempts) and self.attempts[-1].accepted


def lessons_path(path: Path | str | None = None) -> Path:
    return Path(path) if path is not None else DEFAULT_LESSONS_PATH


def load_lessons(
    subject_label: str,
    *,
    category: str | None,
    path: Path | str | None = None,
    limit: int = 5,
) -> list[str]:
    """Return successful prompt-improvement snippets for this subject."""
    store = lessons_path(path)
    if not store.is_file():
        return []
    needle = subject_label.strip().lower()
    cat = (category or "").strip().lower()
    hits: list[str] = []
    for line in store.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("subject", "")).lower() != needle:
            continue
        if cat and str(row.get("category", "")).lower() not in {"", cat}:
            continue
        if not row.get("accepted"):
            continue
        improvement = (row.get("revised_addition") or row.get("improvement") or "").strip()
        if improvement and improvement not in hits:
            hits.append(improvement)
        if len(hits) >= limit:
            break
    return hits


def record_lesson(
    *,
    subject_label: str,
    category: str | None,
    failed_prompt: str,
    critique: SubjectCritique,
    revised_prompt: str,
    accepted: bool,
    path: Path | str | None = None,
) -> None:
    """Append one learning row for future prompt seeding."""
    store = lessons_path(path)
    store.parent.mkdir(parents=True, exist_ok=True)
    addition = _prompt_delta(failed_prompt, revised_prompt)
    row = {
        "subject": subject_label,
        "category": category,
        "failed_prompt": failed_prompt,
        "revised_prompt": revised_prompt,
        "revised_addition": addition,
        "accepted": accepted,
        "critique": asdict(critique),
    }
    with store.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prompt_delta(before: str, after: str) -> str:
    if after.startswith(before):
        return after[len(before) :].strip(" ,.")
    # Prefer newly added comma-separated clauses.
    before_parts = {p.strip() for p in before.split(",") if p.strip()}
    added = [p.strip() for p in after.split(",") if p.strip() and p.strip() not in before_parts]
    return ", ".join(added[:8])


def _feature_cue(subject_label: str, category: str | None) -> str | None:
    """Look up distinctive cues by short label or known disambiguated phrase."""
    if not category:
        return None
    table = SUBJECT_FEATURE_CUES.get(category, {})
    key = subject_label.strip().lower()
    if key in table:
        return table[key]
    # Also match when the caller passed a disambiguated long label.
    for short, cue in table.items():
        if short in key or key in cue.lower():
            return cue
    return None


def critique_subject_rules(
    image: Image.Image,
    *,
    subject_label: str,
    category: str | None,
    prompt: str,
) -> SubjectCritique:
    """Offline critic: strengthen known weak subjects; flag missing cues.

    Pixel heuristics are weak offline, so this critic judges whether the
    *prompt* already carries the diagnostic features needed for recognition.
    Unknown subjects soft-pass (cannot judge without a vision model).
    """
    del image
    subject = disambiguate_subject_label(subject_label, category=category)
    issues: list[str] = []
    improvements: list[str] = []
    prompt_l = prompt.lower()

    cue = _feature_cue(subject_label, category)
    if cue:
        diagnostic_bits = (
            "elliptical wings",
            "propeller",
            "side view",
            "wrinkled",
            "curled tail",
            "delta wings",
            "droop nose",
            "main rotor",
        )
        missing_bits = [
            bit for bit in diagnostic_bits if bit in cue.lower() and bit not in prompt_l
        ]
        # Accept either the full cue text or all of its diagnostic bits.
        cue_covered = cue.lower() in prompt_l or not missing_bits
        if not cue_covered:
            issues.append(
                f"Prompt lacks distinctive {subject_label} features; "
                "model may invent a wrong entity (e.g. person/character)."
            )
            improvements.append(cue)

    negatives = CATEGORY_NEGATIVE_CUES.get(category or "", "")
    if negatives:
        for token in ("no people", "no person", "vehicle only", "animal only"):
            if token in negatives and token not in prompt_l:
                improvements.append(negatives)
                break

    kind = subject_kind_frame(category)
    if kind and "subject kind:" not in prompt_l:
        improvements.append(kind)

    if (
        category == "aircraft"
        and "aeroplane" not in prompt_l
        and "aircraft" not in prompt_l
    ):
        issues.append("Aircraft kind not explicit in prompt.")
        improvements.append(f"{subject}, aeroplane aircraft only, side view silhouette")

    uniq: list[str] = []
    for item in improvements:
        if item and item not in uniq:
            uniq.append(item)

    if issues:
        recognisable = False
        confidence = 0.3
    elif cue is not None:
        # Known subject with diagnostic cues already in the prompt.
        recognisable = True
        confidence = 0.85
    else:
        # Unknown subject: offline rules cannot judge the pixels — soft-pass
        # so we do not thrash retries; prefer openai/human for real checks.
        recognisable = True
        confidence = 0.75
        uniq.append(
            f"emphasise unmistakable {subject} silhouette and diagnostic details"
        )

    return SubjectCritique(
        recognisable=recognisable,
        confidence=confidence,
        issues=tuple(issues),
        improvements=tuple(uniq),
        mode="rules",
    )


def critique_subject_openai(
    image: Image.Image,
    *,
    subject_label: str,
    category: str | None,
    prompt: str,
    api_key: str | None = None,
) -> SubjectCritique:
    """Vision critic using OpenAI chat completions (optional)."""
    import base64
    import io
    import json as json_mod
    import urllib.request

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set for openai critique mode")

    subject = disambiguate_subject_label(subject_label, category=category)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    system = (
        "You evaluate colouring-book illustrations for subject recognition. "
        "Reply with JSON only."
    )
    user_text = (
        f"Requested subject: {subject}\n"
        f"Category: {category or 'unknown'}\n"
        f"Generation prompt used:\n{prompt}\n\n"
        "Questions:\n"
        "1) Is the image recognisable as that subject (not a person/character "
        "or wrong object)?\n"
        "2) How should the generation prompt improve?\n\n"
        "Return JSON: "
        '{"recognisable": bool, "confidence": number, '
        '"issues": [str], "improvements": [str]}'
    )
    payload = {
        "model": os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini"),
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json_mod.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json_mod.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    data = json_mod.loads(content)
    return SubjectCritique(
        recognisable=bool(data.get("recognisable")),
        confidence=float(data.get("confidence", 0.0)),
        issues=tuple(str(x) for x in data.get("issues", [])),
        improvements=tuple(str(x) for x in data.get("improvements", [])),
        mode="openai",
        raw=content,
    )


def critique_subject_human(
    image: Image.Image,
    *,
    subject_label: str,
    category: str | None,
    prompt: str,
) -> SubjectCritique:
    """Interactive critic for local agent/human-in-the-loop runs."""
    del image
    subject = disambiguate_subject_label(subject_label, category=category)
    print(f"\n=== Subject recognition check ===")
    print(f"Requested: {subject} ({category})")
    print(f"Prompt: {prompt[:300]}{'…' if len(prompt) > 300 else ''}")
    answer = input("Is this recognisable as the requested subject? [y/n]: ").strip().lower()
    recognisable = answer in {"y", "yes"}
    issues: list[str] = []
    improvements: list[str] = []
    if not recognisable:
        issue = input("What is wrong with the image? ").strip()
        advice = input("How can the generation prompt improve? ").strip()
        if issue:
            issues.append(issue)
        if advice:
            improvements.append(advice)
    conf_raw = input("Confidence 0-1 [default 0.8/0.3]: ").strip()
    if conf_raw:
        confidence = float(conf_raw)
    else:
        confidence = 0.8 if recognisable else 0.3
    return SubjectCritique(
        recognisable=recognisable,
        confidence=confidence,
        issues=tuple(issues),
        improvements=tuple(improvements),
        mode="human",
    )


def critique_subject(
    image: Image.Image,
    *,
    subject_label: str,
    category: str | None,
    prompt: str,
    mode: str = "rules",
    api_key: str | None = None,
) -> SubjectCritique:
    """Dispatch to the selected critic."""
    mode = mode.lower().strip()
    if mode == "openai":
        try:
            return critique_subject_openai(
                image,
                subject_label=subject_label,
                category=category,
                prompt=prompt,
                api_key=api_key,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI critique failed (%s); falling back to rules", exc)
            return critique_subject_rules(
                image,
                subject_label=subject_label,
                category=category,
                prompt=prompt,
            )
    if mode == "human":
        return critique_subject_human(
            image,
            subject_label=subject_label,
            category=category,
            prompt=prompt,
        )
    return critique_subject_rules(
        image,
        subject_label=subject_label,
        category=category,
        prompt=prompt,
    )


def revise_prompt(
    prompt: str,
    critique: SubjectCritique,
    *,
    subject_label: str,
    category: str | None,
) -> str:
    """Merge critique improvements into a stronger generation prompt."""
    parts = [prompt.strip()]
    subject = disambiguate_subject_label(subject_label, category=category)
    kind = subject_kind_frame(category)
    if kind and kind not in prompt:
        parts.insert(0, kind)

    cue = _feature_cue(subject_label, category)
    if cue and cue not in prompt:
        parts.append(cue)

    for improvement in critique.improvements:
        text = improvement.strip()
        if not text:
            continue
        if text.lower() in ",".join(parts).lower():
            continue
        parts.append(text)

    negatives = CATEGORY_NEGATIVE_CUES.get(category or "", "")
    if negatives and negatives not in ",".join(parts):
        parts.append(negatives)

    # Keep subject phrase visible near the front.
    revised = ", ".join(p for p in parts if p)
    if subject.lower() not in revised.lower():
        revised = f"{subject}, {revised}"
    # Collapse whitespace / duplicate commas.
    revised = re.sub(r"\s+", " ", revised)
    revised = re.sub(r"(,\s*){2,}", ", ", revised)
    return revised.strip(" ,")


def seed_prompt_with_lessons(
    prompt: str,
    *,
    subject_label: str,
    category: str | None,
    path: Path | str | None = None,
    plate_lessons_path: Path | str | None = None,
) -> tuple[str, list[str]]:
    """Apply stored lessons and known feature cues before the first attempt."""
    from .plate_critique import seed_prompt_with_plate_lessons

    extras: list[str] = []
    cue = _feature_cue(subject_label, category)
    if cue and cue.lower() not in prompt.lower():
        prompt = f"{prompt}, {cue}"
        extras.append(cue)

    lessons = load_lessons(subject_label, category=category, path=path)
    for lesson in lessons:
        if lesson.lower() not in prompt.lower():
            prompt = f"{prompt}, {lesson}"
            extras.append(lesson)

    prompt, plate_extras = seed_prompt_with_plate_lessons(
        prompt, category=category, path=plate_lessons_path
    )
    extras.extend(plate_extras)
    return prompt, extras


def run_subject_feedback_loop(
    *,
    subject_label: str,
    category: str | None,
    initial_prompt: str,
    generate_fn,
    critique_mode: str = "rules",
    max_attempts: int = 3,
    api_key: str | None = None,
    lessons_file: Path | str | None = None,
    record: bool = True,
) -> FeedbackLoopResult:
    """Generate with critique/revise retries until recognised or attempts exhausted.

    ``generate_fn(prompt: str) -> Image.Image`` performs one illustration call.
    """
    prompt, used_lessons = seed_prompt_with_lessons(
        initial_prompt,
        subject_label=subject_label,
        category=category,
        path=lessons_file,
    )
    attempts: list[FeedbackAttempt] = []
    image: Image.Image | None = None

    for round_idx in range(max(1, max_attempts)):
        image = generate_fn(prompt)
        critique = critique_subject(
            image,
            subject_label=subject_label,
            category=category,
            prompt=prompt,
            mode=critique_mode,
            api_key=api_key,
        )
        accepted = critique.passed
        attempts.append(
            FeedbackAttempt(prompt=prompt, critique=critique, accepted=accepted)
        )
        logger.info(
            "Feedback round %s: recognisable=%s confidence=%.2f mode=%s",
            round_idx + 1,
            critique.recognisable,
            critique.confidence,
            critique.mode,
        )
        if accepted:
            if record and round_idx > 0:
                record_lesson(
                    subject_label=subject_label,
                    category=category,
                    failed_prompt=attempts[0].prompt,
                    critique=critique,
                    revised_prompt=prompt,
                    accepted=True,
                    path=lessons_file,
                )
            break

        if round_idx + 1 >= max_attempts:
            if record:
                record_lesson(
                    subject_label=subject_label,
                    category=category,
                    failed_prompt=prompt,
                    critique=critique,
                    revised_prompt=revise_prompt(
                        prompt,
                        critique,
                        subject_label=subject_label,
                        category=category,
                    ),
                    accepted=False,
                    path=lessons_file,
                )
            break

        prompt = revise_prompt(
            prompt,
            critique,
            subject_label=subject_label,
            category=category,
        )

    assert image is not None
    notes = (
        f"Subject feedback: {len(attempts)} attempt(s), "
        f"final={'pass' if attempts[-1].accepted else 'fail'} "
        f"via {attempts[-1].critique.mode}"
    )
    return FeedbackLoopResult(
        image=image,
        prompt=prompt,
        attempts=attempts,
        lessons_used=used_lessons,
        notes=notes,
    )
