"""Phase D set generation: run planned slots, reject duplicates / failed plates."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .generate import GeneratedPage, generate_colouring_page
from .quality import PlateQualityReport, evaluate_plate_quality
from .set_plan import PlateSlot, SetPlan, plan_colouring_set

logger = logging.getLogger(__name__)

# 8×8 dHash → 64 bits; Hamming distance at/under this counts as near-duplicate.
DEFAULT_DUPLICATE_HAMMING_MAX = 10


@dataclass(frozen=True)
class SetQualityCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SetQualityReport:
    checks: tuple[SetQualityCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"Set quality: {status}"]
        for check in self.checks:
            mark = "✓" if check.passed else "✗"
            lines.append(f"  {mark} {check.name}: {check.detail}")
        return "\n".join(lines)


@dataclass
class SlotAttemptResult:
    slot: PlateSlot
    status: str  # accepted | rejected | error
    page: GeneratedPage | None = None
    reason: str = ""
    attempts: int = 0
    paths: dict[str, str] = field(default_factory=dict)
    dhash: list[int] | None = None


@dataclass
class GeneratedSet:
    plan: SetPlan
    results: list[SlotAttemptResult]
    quality: SetQualityReport | None = None

    @property
    def accepted(self) -> list[SlotAttemptResult]:
        return [item for item in self.results if item.status == "accepted"]

    @property
    def passed(self) -> bool:
        return self.quality.passed if self.quality is not None else False


def image_dhash(image: Image.Image, *, hash_size: int = 8) -> np.ndarray:
    """Difference hash for cheap near-duplicate detection."""
    gray = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    return diff.astype(np.uint8).reshape(-1)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape:
        raise ValueError("hash shapes must match")
    return int(np.count_nonzero(a != b))


def is_near_duplicate(
    candidate: np.ndarray,
    accepted_hashes: list[np.ndarray],
    *,
    max_distance: int = DEFAULT_DUPLICATE_HAMMING_MAX,
) -> bool:
    return any(
        hamming_distance(candidate, prior) <= max_distance for prior in accepted_hashes
    )


def evaluate_set_quality(
    plan: SetPlan,
    results: list[SlotAttemptResult],
    *,
    required_plates: int | None = None,
) -> SetQualityReport:
    """Aggregate Phase D gate: N accepted, unique slots, plates OK, no dupes."""
    need = required_plates if required_plates is not None else plan.n_plates
    accepted = [item for item in results if item.status == "accepted"]
    checks: list[SetQualityCheck] = []

    checks.append(
        SetQualityCheck(
            name="accepted_count",
            passed=len(accepted) >= need,
            detail=f"{len(accepted)} accepted / {need} required",
        )
    )

    slot_keys = {(s.aspect.lower(), s.scene.lower()) for s in plan.slots}
    checks.append(
        SetQualityCheck(
            name="unique_slot_plan",
            passed=len(slot_keys) == len(plan.slots),
            detail=f"{len(slot_keys)} unique aspect/scene pairs in plan",
        )
    )

    plate_ok = True
    plate_detail = "no accepted plates"
    if accepted:
        failed = [
            item.slot.slug
            for item in accepted
            if item.page is None
            or item.page.quality is None
            or not item.page.quality.passed
        ]
        plate_ok = not failed
        plate_detail = (
            "all accepted plates pass Phase B/C checklist"
            if plate_ok
            else f"failed plates: {', '.join(failed[:6])}"
        )
    checks.append(
        SetQualityCheck(
            name="plates_pass_quality",
            passed=plate_ok and bool(accepted),
            detail=plate_detail,
        )
    )

    hashes = [np.asarray(item.dhash, dtype=np.uint8) for item in accepted if item.dhash]
    dup_pairs = 0
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hamming_distance(hashes[i], hashes[j]) <= DEFAULT_DUPLICATE_HAMMING_MAX:
                dup_pairs += 1
    checks.append(
        SetQualityCheck(
            name="no_near_duplicates",
            passed=dup_pairs == 0,
            detail=f"{dup_pairs} near-duplicate pair(s) among accepted plates",
        )
    )

    if getattr(plan, "mode", "single") == "mixed":
        identity_ok = all(
            item.page is not None
            and (
                item.slot.subject_label is None
                or item.page.subject_type.label == item.slot.subject_label
            )
            for item in accepted
        )
        subjects = sorted(
            {
                (item.slot.subject_label or plan.subject_type.label)
                for item in plan.slots
            }
        )
        identity_detail = (
            f"mixed subjects match slots ({', '.join(subjects)})"
            if identity_ok
            else "one or more plates mismatched their slot subject"
        )
        checks.append(
            SetQualityCheck(
                name="slot_subject_identity",
                passed=identity_ok and bool(accepted),
                detail=identity_detail,
            )
        )
    else:
        identity = plan.subject_type.label
        checks.append(
            SetQualityCheck(
                name="shared_subject_identity",
                passed=bool(identity)
                and all(
                    item.page is not None
                    and item.page.subject_type.label == identity
                    for item in accepted
                ),
                detail=f"subject “{identity}” on all accepted plates",
            )
        )

    return SetQualityReport(checks=tuple(checks))


def _save_slot(
    page: GeneratedPage,
    slot: PlateSlot,
    output_dir: Path,
) -> dict[str, str]:
    slot_dir = output_dir / slot.slug
    slot_dir.mkdir(parents=True, exist_ok=True)
    stem = slot.slug
    paths = page.result.save(slot_dir, stem=stem)
    illustration_path = slot_dir / f"{stem}_illustration.png"
    page.illustration.image.save(illustration_path)
    out = {key: str(path) for key, path in paths.items()}
    out["illustration"] = str(illustration_path)
    if page.quality is not None:
        (slot_dir / f"{stem}_quality.txt").write_text(
            page.quality.summary() + "\n", encoding="utf-8"
        )
    return out


def generate_colouring_set(
    query: str,
    *,
    subject_type: str | None = None,
    type_pick: int = 0,
    n_plates: int = 6,
    base_seed: int = 0,
    discover_types: bool = True,
    attempts_per_slot: int = 3,
    require_plate_quality: bool = True,
    duplicate_hamming_max: int = DEFAULT_DUPLICATE_HAMMING_MAX,
    output_dir: Path | str | None = None,
    plan: SetPlan | None = None,
    generate_page_fn=None,
    library_root: Path | str | None = None,
    library_title: str | None = None,
    **page_kwargs,
) -> GeneratedSet:
    """Generate a varied plate set from one phrase.

    Each slot calls the single-plate generator with a unique prompt/seed.
    Rejects plates that fail the Phase B/C checklist or are near-duplicates of
    an already accepted plate. Retries a slot up to ``attempts_per_slot`` times
    with bumped seeds before marking it rejected.
    """
    active_plan = plan or plan_colouring_set(
        query,
        subject_type=subject_type,
        type_pick=type_pick,
        n_plates=n_plates,
        base_seed=base_seed,
        discover_types=discover_types,
    )
    generate = generate_page_fn or generate_colouring_page
    out: Path | None = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    results: list[SlotAttemptResult] = []
    accepted_hashes: list[np.ndarray] = []

    for slot in active_plan.slots:
        outcome = SlotAttemptResult(slot=slot, status="rejected", reason="not attempted")
        for attempt in range(max(1, attempts_per_slot)):
            seed = slot.seed + attempt * 17
            try:
                slot_subject = slot.subject_label or active_plan.subject_type.label
                page: GeneratedPage = generate(
                    slot_subject,
                    subject_type=slot_subject,
                    discover_types=False,
                    prompt_override=slot.prompt,
                    seed=seed,
                    check_quality=True,
                    require_quality=False,
                    **page_kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Slot %s attempt %s failed: %s", slot.slug, attempt + 1, exc)
                outcome = SlotAttemptResult(
                    slot=slot,
                    status="error",
                    reason=str(exc),
                    attempts=attempt + 1,
                )
                continue

            quality = page.quality
            if quality is None:
                quality = evaluate_plate_quality(
                    page.result, colour_plate=page.illustration.image
                )
                page = GeneratedPage(
                    illustration=page.illustration,
                    result=page.result,
                    subject_type=page.subject_type,
                    reference_hit=page.reference_hit,
                    quality=quality,
                    feedback=page.feedback,
                )

            if require_plate_quality and not quality.passed:
                outcome = SlotAttemptResult(
                    slot=slot,
                    status="rejected",
                    page=page,
                    reason="plate quality gate failed",
                    attempts=attempt + 1,
                )
                continue

            digest = image_dhash(page.illustration.image)
            if is_near_duplicate(
                digest, accepted_hashes, max_distance=duplicate_hamming_max
            ):
                outcome = SlotAttemptResult(
                    slot=slot,
                    status="rejected",
                    page=page,
                    reason="near-duplicate of an accepted plate",
                    attempts=attempt + 1,
                    dhash=digest.tolist(),
                )
                continue

            paths: dict[str, str] = {}
            if out is not None:
                paths = _save_slot(page, slot, out)
            accepted_hashes.append(digest)
            outcome = SlotAttemptResult(
                slot=slot,
                status="accepted",
                page=page,
                reason="ok",
                attempts=attempt + 1,
                paths=paths,
                dhash=digest.tolist(),
            )
            break

        results.append(outcome)
        logger.info(
            "Slot %s → %s (%s)", outcome.slot.slug, outcome.status, outcome.reason
        )

    report = evaluate_set_quality(active_plan, results)
    generated = GeneratedSet(plan=active_plan, results=results, quality=report)

    if out is not None:
        write_set_manifest(generated, out)

    if library_root is not None:
        from .library import AssetLibrary, ingest_generated_set

        style = page_kwargs.get("style")
        record = ingest_generated_set(
            generated,
            library=AssetLibrary(library_root),
            title=library_title,
            style=style if isinstance(style, str) else None,
        )
        logger.info(
            "Ingested set into library %s (%s pairs)",
            record.set_id,
            len(record.pair_ids),
        )
        if out is not None:
            (out / "library_set_id.txt").write_text(record.set_id + "\n", encoding="utf-8")

    return generated


def write_set_manifest(generated: GeneratedSet, output_dir: Path | str) -> Path:
    """Persist plan + per-slot outcomes for book compilation later."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan": generated.plan.to_dict(),
        "set_quality": {
            "passed": generated.quality.passed if generated.quality else False,
            "summary": generated.quality.summary() if generated.quality else "",
            "checks": [
                {
                    "name": check.name,
                    "passed": check.passed,
                    "detail": check.detail,
                }
                for check in (generated.quality.checks if generated.quality else ())
            ],
        },
        "results": [
            {
                "slot": item.slot.slug,
                "index": item.slot.index,
                "aspect": item.slot.aspect,
                "scene": item.slot.scene,
                "status": item.status,
                "reason": item.reason,
                "attempts": item.attempts,
                "paths": item.paths,
                "plate_quality_passed": (
                    None
                    if item.page is None or item.page.quality is None
                    else item.page.quality.passed
                ),
            }
            for item in generated.results
        ],
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if generated.quality is not None:
        (out / "set_quality.txt").write_text(
            generated.quality.summary() + "\n", encoding="utf-8"
        )
    (out / "plan.json").write_text(
        json.dumps(generated.plan.to_dict(), indent=2), encoding="utf-8"
    )
    return path
