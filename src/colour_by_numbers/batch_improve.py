"""Batch assess → learn → regenerate loop (first slice).

Rules-based auto-tagging of common plate failure modes, critique recording,
lesson collation, weak-slot regeneration, and a before/after report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .contrast import _is_warm_coat_like_rgb, _warm_fur_pixels
from .palette import rgb_to_lab
from .plate_critique import (
    AUTO_ISSUE_TAGS,
    PLATE_ISSUE_TAGS,
    PlateCritique,
    collate_critiques,
    load_critiques,
    prompt_hint_for_tag,
    record_plate_critique,
    seed_prompt_with_plate_lessons,
    write_lessons_json,
)
from .set_plan import compose_slot_prompt
from .subject import estimate_subject_mask, harden_mask

logger = logging.getLogger(__name__)

# Primary tags that mark a slot for regeneration. Aliases (outline/nose_detail)
# alone do not force a redraw.
# Structural tags that trigger regeneration. Muzzle flecks are recorded for
# lessons but do not alone force a redraw (vibrant snouts are naturally tiled).
DEFAULT_WEAK_TAGS = frozenset(
    {
        "silhouette_notch",
        "cream_background",
        "busy_background",
    }
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class AutoTagHit:
    tag: str
    score: float
    detail: str


@dataclass
class SlotAssessment:
    """Assessment of one library pair / set slot."""

    plate_id: str
    slot: str
    category: str
    subject: str
    rating: str
    issues: list[str] = field(default_factory=list)
    hits: list[AutoTagHit] = field(default_factory=list)
    quality_passed: bool | None = None
    quality_failures: list[str] = field(default_factory=list)
    notes: str = ""
    weak: bool = False

    def to_dict(self) -> dict:
        return {
            "plate_id": self.plate_id,
            "slot": self.slot,
            "category": self.category,
            "subject": self.subject,
            "rating": self.rating,
            "issues": list(self.issues),
            "hits": [asdict(h) for h in self.hits],
            "quality_passed": self.quality_passed,
            "quality_failures": list(self.quality_failures),
            "notes": self.notes,
            "weak": self.weak,
        }


@dataclass
class BatchImproveReport:
    """Before/after summary for one improve-loop run."""

    set_id: str
    generated_at: str
    before: list[dict]
    after: list[dict]
    regenerated: list[str]
    lessons_path: str
    critiques_appended: int
    tag_counts_before: dict[str, int]
    tag_counts_after: dict[str, int]
    weak_before: list[str]
    weak_after: list[str]
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _estimate_subject_binary(image: Image.Image) -> np.ndarray:
    try:
        mask = harden_mask(estimate_subject_mask(image.convert("RGB")))
        return mask.binary
    except Exception as exc:  # noqa: BLE001
        logger.warning("rembg failed for auto-tag (%s); using warm heuristic", exc)
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        warm = _warm_fur_pixels(rgb)
        if not warm.any():
            return np.ones(rgb.shape[:2], dtype=bool)
        # Keep largest warm component.
        labelled, n = ndimage.label(warm)
        if n == 0:
            return warm
        sizes = np.bincount(labelled.ravel())
        sizes[0] = 0
        return labelled == int(sizes.argmax())


def _cool_bg_mask(plate_rgb: np.ndarray, subject: np.ndarray) -> np.ndarray:
    r = plate_rgb[:, :, 0].astype(np.int16)
    g = plate_rgb[:, :, 1].astype(np.int16)
    b = plate_rgb[:, :, 2].astype(np.int16)
    cool = (b > r + 8) & (b >= g) & (r < 160)
    return cool & ~subject


def detect_silhouette_notch(
    plate: Image.Image,
    *,
    subject: np.ndarray | None = None,
    min_area: int = 24,
) -> AutoTagHit | None:
    """Flag compact cool bites cupped into the top of the subject silhouette.

    Uses warm-fur geometry (not rembg alone) so teal notches that rembg still
    includes inside the matte remain detectable.
    """
    rgb = np.asarray(plate.convert("RGB"), dtype=np.uint8)
    warm = _warm_fur_pixels(rgb)
    if subject is not None and subject.any():
        # Prefer the warm core inside the matte; fall back to the matte.
        core = warm & subject
        body = core if core.any() else subject
    else:
        body = warm
        if not body.any():
            body = _estimate_subject_binary(plate)
    if not body.any():
        return None

    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    cool = (b > r + 8) & (b >= g - 5) & (r < 170) & ~warm

    ys, xs = np.where(body)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    top_band = np.zeros_like(body)
    top_y1 = y0 + max(6, int(0.32 * (y1 - y0 + 1)))
    top_band[max(0, y0 - 4) : top_y1, x0 : x1 + 1] = True

    # Pinch test: cool pixels that see warm body on opposite sides / 3 directions.
    # Do not require morphological "near" — deep crown bites sit in holes that
    # a short dilation from the warm core may not fill.
    left = np.zeros_like(body)
    right = np.zeros_like(body)
    up = np.zeros_like(body)
    down = np.zeros_like(body)
    for i in range(1, 18):
        left[:, i:] |= body[:, :-i]
        right[:, :-i] |= body[:, i:]
        up[i:, :] |= body[:-i, :]
        down[:-i, :] |= body[i:, :]
    pinch = (left & right & down) | ((left.astype(np.uint8) + right + up + down) >= 3)
    candidates = cool & ~body & top_band & pinch
    if not candidates.any():
        return None

    labelled, n = ndimage.label(candidates)
    best_area = 0
    best_touch = 0.0
    for cid in range(1, n + 1):
        comp = labelled == cid
        area = int(comp.sum())
        if area < min_area:
            continue
        # Reject wide top bars (true background); keep compact bites.
        ys_c, xs_c = np.where(comp)
        width = int(xs_c.max()) - int(xs_c.min()) + 1
        height = int(ys_c.max()) - int(ys_c.min()) + 1
        if width > max(36, int(0.40 * (x1 - x0 + 1))):
            continue
        if height > max(40, int(0.40 * (y1 - y0 + 1))):
            continue
        border = ndimage.binary_dilation(comp) & ~comp
        if not border.any():
            continue
        # Open crown bites touch exterior cool bg on top — score touch only
        # against warm body among non-exterior border pixels.
        interior_border = border & ndimage.binary_dilation(body, iterations=2)
        denom = float(interior_border.sum()) or float(border.sum())
        touch = float((border & body).sum()) / denom
        # Pinch already required left/right/down; accept modest touch.
        if touch >= 0.08 and area > best_area:
            best_area = area
            best_touch = touch
    if best_area <= 0:
        return None
    return AutoTagHit(
        tag="silhouette_notch",
        score=min(1.0, best_area / 300.0) * (0.55 + 0.45 * min(1.0, best_touch / 0.4)),
        detail=f"cool bite ≈{best_area}px into crown (touch={best_touch:.2f})",
    )


def detect_cream_background(
    plate: Image.Image,
    *,
    subject: np.ndarray | None = None,
    min_frac: float = 0.04,
) -> AutoTagHit | None:
    """Flag warm cream/tan fills that survived in the background."""
    rgb = np.asarray(plate.convert("RGB"), dtype=np.uint8)
    if subject is None:
        subject = _estimate_subject_binary(plate)
    bg = ~subject
    if not bg.any():
        return None
    cream = np.zeros(rgb.shape[:2], dtype=bool)
    # Vectorised warm-coat test on unique palette samples.
    flat = rgb.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    warm_u = np.array([_is_warm_coat_like_rgb(c) for c in uniq], dtype=bool)
    cream = warm_u[inv].reshape(rgb.shape[:2]) & bg
    frac = float(cream.mean())
    bg_frac = float(cream.sum()) / float(bg.sum())
    if bg_frac < min_frac and frac < 0.02:
        return None
    return AutoTagHit(
        tag="cream_background",
        score=min(1.0, bg_frac / 0.15),
        detail=f"warm coat-like bg covers {100 * bg_frac:.1f}% of background",
    )


def detect_busy_background(
    plate: Image.Image,
    *,
    subject: np.ndarray | None = None,
    max_regions: int = 20,
) -> AutoTagHit | None:
    """Flag fragmented figurative backgrounds (many bg components)."""
    rgb = np.asarray(plate.convert("RGB"), dtype=np.uint8)
    if subject is None:
        subject = _estimate_subject_binary(plate)
    bg = ~subject
    if not bg.any():
        return None
    # Quantize lightly by rounding to reduce noise, then count components.
    q = (rgb // 24) * 24
    codes = (
        q[:, :, 0].astype(np.int32) * 256 * 256
        + q[:, :, 1].astype(np.int32) * 256
        + q[:, :, 2].astype(np.int32)
    )
    labelled = np.zeros(bg.shape, dtype=np.int32)
    next_id = 1
    for code in np.unique(codes[bg]):
        mask = bg & (codes == code)
        part, n = ndimage.label(mask)
        for cid in range(1, n + 1):
            comp = part == cid
            if int(comp.sum()) < 40:
                continue
            labelled[comp] = next_id
            next_id += 1
    n_regions = int(next_id - 1)
    if n_regions <= max_regions:
        return None
    return AutoTagHit(
        tag="busy_background",
        score=min(1.0, (n_regions - max_regions) / 16.0),
        detail=f"{n_regions} background colour regions (max soft {max_regions})",
    )


def detect_muzzle_fleck(
    plate: Image.Image,
    *,
    subject: np.ndarray | None = None,
    category: str | None = "dogs",
) -> AutoTagHit | None:
    """Flag small high-contrast islands on/near the nose leather only."""
    if (category or "") not in {
        "dogs",
        "cats",
        "horses",
        "wildlife",
        "animals",
        "pets",
        "farm animals",
        "mammals",
        "people",
        "portraits",
    }:
        return None
    rgb = np.asarray(plate.convert("RGB"), dtype=np.uint8)
    if subject is None:
        subject = _estimate_subject_binary(plate)
    if not subject.any():
        return None
    lab = rgb_to_lab(rgb)
    luma = lab[:, :, 0]
    ys, xs = np.where(subject)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bh, bw = max(1, y1 - y0 + 1), max(1, x1 - x0 + 1)
    # Snout band (lower face) — avoid eye sockets in the upper face.
    snout = np.zeros_like(subject)
    snout[
        y0 + int(0.35 * bh) : y0 + int(0.75 * bh),
        x0 + int(0.25 * bw) : x1 - int(0.25 * bw) + 1,
    ] = True
    snout &= subject
    dark = snout & (luma <= 38.0)
    if int(dark.sum()) < 12:
        return None
    labelled, n = ndimage.label(dark)
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    nose_id = int(sizes.argmax())
    nose_area = int(sizes[nose_id])
    if nose_area < 20:
        return None
    nose = labelled == nose_id
    # Tight halo on the nose leather only (bridge flecks / specular dirt).
    halo = ndimage.binary_dilation(nose, iterations=3) & snout
    structure = np.ones((3, 3), dtype=bool)
    flecks = 0
    best = 0
    for colour in np.unique(rgb[halo].reshape(-1, 3), axis=0):
        band = (
            halo
            & (rgb[:, :, 0] == colour[0])
            & (rgb[:, :, 1] == colour[1])
            & (rgb[:, :, 2] == colour[2])
        )
        part, pn = ndimage.label(band, structure=structure)
        for cid in range(1, pn + 1):
            comp = part == cid
            area = int(comp.sum())
            if area < 4 or area > min(80, int(0.45 * nose_area)):
                continue
            if int((comp & nose).sum()) >= int(0.6 * nose_area):
                continue
            # Must touch the nose leather.
            if not (ndimage.binary_dilation(nose, iterations=2) & comp).any():
                continue
            dil = ndimage.binary_dilation(comp, iterations=2) & ~comp & snout
            if not dil.any():
                continue
            contrast = abs(float(luma[comp].mean()) - float(luma[dil].mean()))
            if contrast < 18.0:
                continue
            flecks += 1
            best = max(best, area)
    if flecks <= 0:
        return None
    return AutoTagHit(
        tag="muzzle_fleck",
        score=min(1.0, 0.45 * flecks + best / 60.0),
        detail=f"{flecks} high-contrast snout island(s), largest≈{best}px",
    )


def auto_tag_plate(
    plate: Image.Image,
    *,
    illustration: Image.Image | None = None,
    category: str | None = None,
    min_score: float = 0.35,
) -> list[AutoTagHit]:
    """Rules-based issue tags for one plate (optionally using the illustration)."""
    del illustration  # reserved for future img2img / absdiff cues
    subject = _estimate_subject_binary(plate)
    hits: list[AutoTagHit] = []
    for detector in (
        lambda: detect_silhouette_notch(plate, subject=subject),
        lambda: detect_cream_background(plate, subject=subject),
        lambda: detect_busy_background(plate, subject=subject),
        lambda: detect_muzzle_fleck(plate, subject=subject, category=category),
    ):
        hit = detector()
        if hit is not None and hit.score >= min_score and hit.tag in PLATE_ISSUE_TAGS:
            hits.append(hit)
    # Map busy/cream into the generic background tag for older review UI filters.
    tags = {h.tag for h in hits}
    if ("cream_background" in tags or "busy_background" in tags) and "background" not in tags:
        hits.append(
            AutoTagHit(
                tag="background",
                score=max(h.score for h in hits if h.tag.endswith("background")),
                detail="alias of cream/busy background findings",
            )
        )
    if "silhouette_notch" in tags and "outline" not in tags:
        notch = next(h for h in hits if h.tag == "silhouette_notch")
        hits.append(
            AutoTagHit(tag="outline", score=notch.score * 0.8, detail="alias of silhouette_notch")
        )
    if "muzzle_fleck" in tags and "nose_detail" not in tags:
        fleck = next(h for h in hits if h.tag == "muzzle_fleck")
        hits.append(
            AutoTagHit(
                tag="nose_detail",
                score=fleck.score * 0.8,
                detail="alias of muzzle_fleck",
            )
        )
    return hits


def assess_pair_dir(
    pair_dir: Path,
    *,
    set_id: str,
    category: str,
    subject: str,
    weak_tags: frozenset[str] = DEFAULT_WEAK_TAGS,
    min_score: float = 0.35,
) -> SlotAssessment:
    """Assess one library pair folder (expects plate.png, optional illustration)."""
    pair_dir = Path(pair_dir)
    slot = pair_dir.name
    plate_path = pair_dir / "plate.png"
    ill_path = pair_dir / "illustration.png"
    plate = Image.open(plate_path).convert("RGB")
    illustration = Image.open(ill_path).convert("RGB") if ill_path.is_file() else None
    hits = auto_tag_plate(
        plate, illustration=illustration, category=category, min_score=min_score
    )
    issues = [h.tag for h in hits if h.tag in AUTO_ISSUE_TAGS or h.tag in weak_tags]
    # Prefer primary tags in the critique issues list.
    primary = [
        t
        for t in (
            "silhouette_notch",
            "cream_background",
            "busy_background",
            "muzzle_fleck",
            "background",
            "outline",
            "nose_detail",
        )
        if t in {h.tag for h in hits}
    ]
    quality_passed = None
    quality_failures: list[str] = []
    try:
        # Lightweight quality on plate + outline if present.
        from .pipeline import create_colour_by_numbers
        from .style_presets import STYLE_VIBRANT

        # Re-run is expensive; only score palette/region heuristics from plate assets.
        outline = pair_dir / "outline.png"
        legend = pair_dir / "legend.png"
        if outline.is_file() and legend.is_file():
            # Build a minimal report via evaluate if we have a ColourByNumbersResult —
            # skip full pipeline; use plate-only soft signals.
            pass
        del create_colour_by_numbers, STYLE_VIBRANT
    except Exception:  # noqa: BLE001
        pass

    primary_weak = [t for t in primary if t in weak_tags]
    weak = bool(primary_weak)
    rating = "pass"
    if weak:
        rating = "needs_work"
    if any(h.score >= 0.75 for h in hits if h.tag in weak_tags):
        rating = "fail"

    notes = "; ".join(h.detail for h in hits if h.tag in primary) or "auto-tag clean"
    return SlotAssessment(
        plate_id=f"{set_id}/{slot}",
        slot=slot,
        category=category,
        subject=subject,
        rating=rating,
        issues=primary,
        hits=hits,
        quality_passed=quality_passed,
        quality_failures=quality_failures,
        notes=notes,
        weak=weak,
    )


def load_known_issues(path: Path | str | None) -> dict[str, dict]:
    """Optional human/known-issue seed: ``{slot: {rating, issues, notes}}``."""
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    slots = data.get("slots") if isinstance(data, dict) else None
    return dict(slots or {})


def assess_library_set(
    set_dir: Path,
    *,
    set_id: str | None = None,
    category: str = "dogs",
    subject: str = "golden retriever",
    weak_tags: frozenset[str] = DEFAULT_WEAK_TAGS,
    known_issues_path: Path | str | None = None,
    force_slots: set[str] | None = None,
) -> list[SlotAssessment]:
    """Assess all ``pairs/pNN`` folders under a library set directory."""
    set_dir = Path(set_dir)
    sid = set_id or set_dir.name
    pairs = set_dir / "pairs"
    if not pairs.is_dir():
        raise FileNotFoundError(f"No pairs/ under {set_dir}")
    known = load_known_issues(known_issues_path)
    force = force_slots or set()
    slots = sorted(p for p in pairs.iterdir() if p.is_dir() and p.name.startswith("p"))
    out: list[SlotAssessment] = []
    for slot_dir in slots:
        item = assess_pair_dir(
            slot_dir, set_id=sid, category=category, subject=subject, weak_tags=weak_tags
        )
        seed = known.get(item.slot) or known.get(item.slot.upper())
        if seed:
            seeded_issues = [
                str(t) for t in (seed.get("issues") or []) if str(t) in PLATE_ISSUE_TAGS
            ]
            merged = list(dict.fromkeys([*item.issues, *seeded_issues]))
            item.issues = merged
            item.notes = (
                f"{item.notes}; known: {seed.get('notes', '')}".strip("; ")
            )
            item.rating = str(seed.get("rating") or item.rating)
            item.weak = True
            item.hits = list(item.hits) + [
                AutoTagHit(
                    tag=t,
                    score=0.9,
                    detail=f"known issue seed: {seed.get('notes', t)}",
                )
                for t in seeded_issues
                if t not in {h.tag for h in item.hits}
            ]
        if item.slot in force:
            item.weak = True
            if item.rating == "pass":
                item.rating = "needs_work"
        out.append(item)
    return out


def assessments_to_critiques(
    assessments: list[SlotAssessment],
    *,
    reviewer: str = "auto-tagger",
) -> list[PlateCritique]:
    """Convert assessments into plate_critique rows."""
    rows: list[PlateCritique] = []
    stamp = _now_iso()
    for item in assessments:
        if item.rating == "pass" and not item.issues:
            continue
        hints = [
            prompt_hint_for_tag(tag, item.category)
            for tag in item.issues
            if prompt_hint_for_tag(tag, item.category)
        ]
        rows.append(
            PlateCritique(
                plate_id=item.plate_id,
                category=item.category,
                subject=item.subject,
                rating=item.rating,
                issues=tuple(item.issues),
                notes=item.notes,
                suggested_prompt="; ".join(hints),
                reviewer=reviewer,
                reviewed_at=stamp,
            )
        )
    return rows


def _tag_counts(assessments: list[SlotAssessment]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in assessments:
        for tag in item.issues:
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))


def _load_manifest_slots(manifest_path: Path) -> list[dict]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = data.get("plan") or data
    return list(plan.get("slots") or [])


def regenerate_weak_slots(
    *,
    set_dir: Path,
    assessments: list[SlotAssessment],
    manifest_path: Path | None,
    category: str,
    subject: str,
    style: str = "vibrant",
    seed_base: int = 200,
    output_dir: Path | None = None,
    max_slots: int = 3,
) -> list[str]:
    """Re-fal weak slots with lesson-seeded prompts; write into set_dir pairs."""
    from .discover import SubjectType
    from .generate import generate_colouring_page

    weak = [a for a in assessments if a.weak]
    # Prefer fail > needs_work, then more primary structural issues.
    rank = {"fail": 0, "needs_work": 1, "pass": 2}

    def _severity(item: SlotAssessment) -> tuple:
        structural = [
            t
            for t in item.issues
            if t in {"silhouette_notch", "cream_background", "busy_background"}
        ]
        return (rank.get(item.rating, 9), -len(structural), item.slot)

    weak = sorted(weak, key=_severity)[: max(0, int(max_slots))]
    if not weak:
        return []

    slots_meta: dict[int, dict] = {}
    if manifest_path is not None and Path(manifest_path).is_file():
        for row in _load_manifest_slots(Path(manifest_path)):
            slots_meta[int(row["index"])] = row

    subject_type = SubjectType(
        label=subject, category=category, search_query=subject
    )
    regenerated: list[str] = []
    out_root = Path(output_dir) if output_dir is not None else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    for item in weak:
        idx = int(item.slot.lstrip("p"))
        meta = slots_meta.get(idx, {})
        aspect = str(meta.get("aspect") or "side profile")
        scene = str(meta.get("scene") or "cool abstract colour panels")
        composition = str(
            meta.get("composition")
            or "FULL BODY clear side silhouette, all four legs visible"
        )
        tags = tuple(meta.get("tags") or ("side", "full_body", "single"))
        prompt = compose_slot_prompt(
            subject_type,
            aspect=aspect,
            scene=scene,
            composition=composition,
            style_preset=style,
            tags=tags,
        )
        prompt, applied = seed_prompt_with_plate_lessons(
            prompt, category=category, style_preset=style
        )
        logger.info(
            "Regenerating %s with %d lesson hint(s)", item.slot, len(applied)
        )
        page = generate_colouring_page(
            subject,
            subject_type=subject,
            discover_types=False,
            backend="fal",
            style=style,
            prompt_override=prompt,
            seed=seed_base + idx,
            check_quality=True,
            require_quality=False,
            subject_feedback=False,
        )
        dest = Path(set_dir) / "pairs" / item.slot
        dest.mkdir(parents=True, exist_ok=True)
        page.illustration.image.save(dest / "illustration.png")
        page.result.quantized.preview.save(dest / "plate.png")
        page.result.page.outline.save(dest / "outline.png")
        page.result.printable.save(dest / "page.png")
        page.result.page.legend.save(dest / "legend.png")
        if page.result.page.outline_svg:
            (dest / "outline.svg").write_text(
                page.result.page.outline_svg, encoding="utf-8"
            )
        if page.result.page.plate_svg:
            (dest / "plate.svg").write_text(
                page.result.page.plate_svg, encoding="utf-8"
            )
        if out_root is not None:
            slot_out = out_root / item.slot
            slot_out.mkdir(parents=True, exist_ok=True)
            page.illustration.image.save(slot_out / "illustration.png")
            page.result.quantized.preview.save(slot_out / "plate.png")
            page.result.page.outline.save(slot_out / "outline.png")
            (slot_out / "prompt.txt").write_text(prompt, encoding="utf-8")
            (slot_out / "lessons_applied.json").write_text(
                json.dumps(applied, indent=2), encoding="utf-8"
            )
        regenerated.append(item.slot)
    return regenerated


def run_batch_improve_slice(
    set_dir: Path,
    *,
    set_id: str | None = None,
    category: str = "dogs",
    subject: str = "golden retriever",
    critiques_path: Path | str | None = None,
    lessons_path: Path | str | None = None,
    manifest_path: Path | None = None,
    report_dir: Path | None = None,
    regenerate: bool = True,
    reviewer: str = "auto-tagger",
    max_regenerate: int = 3,
    known_issues_path: Path | str | None = None,
    force_slots: set[str] | None = None,
) -> BatchImproveReport:
    """Assess → record critiques → collate lessons → regen weak → reassess."""
    set_dir = Path(set_dir)
    sid = set_id or set_dir.name
    report_dir = Path(report_dir or Path("output") / "batch-improve" / sid)
    report_dir.mkdir(parents=True, exist_ok=True)

    before = assess_library_set(
        set_dir,
        set_id=sid,
        category=category,
        subject=subject,
        known_issues_path=known_issues_path,
        force_slots=force_slots,
    )
    (report_dir / "before.json").write_text(
        json.dumps([a.to_dict() for a in before], indent=2) + "\n", encoding="utf-8"
    )

    critiques = assessments_to_critiques(before, reviewer=reviewer)
    for row in critiques:
        record_plate_critique(row, path=critiques_path)

    all_rows = load_critiques(path=critiques_path)
    collation = collate_critiques(all_rows, min_count=1)
    lessons_file = write_lessons_json(collation, path=lessons_path)
    (report_dir / "collation.txt").write_text(
        json.dumps(
            {
                "total": collation.total,
                "by_tag": collation.by_tag,
                "by_category": collation.by_category,
                "global_hints": collation.global_hints,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    regenerated: list[str] = []
    if regenerate:
        regenerated = regenerate_weak_slots(
            set_dir=set_dir,
            assessments=before,
            manifest_path=manifest_path,
            category=category,
            subject=subject,
            output_dir=report_dir / "regenerated",
            max_slots=max_regenerate,
        )

    # Reassess with auto-tags only so known-issue seeds do not mask improvement.
    after = assess_library_set(
        set_dir, set_id=sid, category=category, subject=subject
    )
    (report_dir / "after.json").write_text(
        json.dumps([a.to_dict() for a in after], indent=2) + "\n", encoding="utf-8"
    )

    report = BatchImproveReport(
        set_id=sid,
        generated_at=_now_iso(),
        before=[a.to_dict() for a in before],
        after=[a.to_dict() for a in after],
        regenerated=regenerated,
        lessons_path=str(lessons_file),
        critiques_appended=len(critiques),
        tag_counts_before=_tag_counts(before),
        tag_counts_after=_tag_counts(after),
        weak_before=[a.slot for a in before if a.weak],
        weak_after=[a.slot for a in after if a.weak],
        notes=(
            "First-slice loop: rules auto-tag → critiques → collate lessons → "
            "regenerate weak slots with seeded prompts → reassess."
        ),
    )
    (report_dir / "report.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "report.md").write_text(format_batch_report_md(report), encoding="utf-8")
    return report


def format_batch_report_md(report: BatchImproveReport) -> str:
    """Human-readable before/after summary."""
    lines = [
        f"# Batch improve report — `{report.set_id}`",
        "",
        f"Generated: {report.generated_at}",
        "",
        "## Summary",
        "",
        f"- Critiques appended: **{report.critiques_appended}**",
        f"- Lessons file: `{report.lessons_path}`",
        f"- Regenerated slots: {', '.join(report.regenerated) or '(none)'}",
        f"- Weak before: {', '.join(report.weak_before) or '(none)'}",
        f"- Weak after: {', '.join(report.weak_after) or '(none)'}",
        "",
        "### Tag counts before",
        "",
        "```json",
        json.dumps(report.tag_counts_before, indent=2),
        "```",
        "",
        "### Tag counts after",
        "",
        "```json",
        json.dumps(report.tag_counts_after, indent=2),
        "```",
        "",
        "## Per-slot before → after",
        "",
    ]
    after_by = {row["slot"]: row for row in report.after}
    for row in report.before:
        slot = row["slot"]
        nxt = after_by.get(slot, {})
        lines.append(
            f"- **{slot}**: {row['rating']} {row['issues']} → "
            f"{nxt.get('rating', '?')} {nxt.get('issues', [])}"
        )
    lines.extend(["", report.notes, ""])
    return "\n".join(lines)
