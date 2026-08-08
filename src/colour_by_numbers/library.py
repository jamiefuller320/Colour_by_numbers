"""On-disk asset library for linked colour-plate / outline pairs and sets.

Design goals
------------
- **Pair** is the atomic unit: colour plate + numbered outline (+ editable
  label map / palette so colourways can be regenerated).
- **Set** collates pairs — single-category generation *or* mixed themes assembled
  from existing pairs.
- Outlines are the primary product; full-colour plates feed covers / guides and
  alternate colourways (natural, vivid, pop art, …).

Storage is JSON + files under ``data/library/`` (no SQL required yet).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .colourways import (
    COLOURWAY_NATURAL,
    list_colourways,
    render_colourway_plate,
    resolve_colourway,
)
from .generate import GeneratedPage
from .outline import build_legend
from .pipeline import ColourByNumbersResult
from .quantize import preview_from_labels

logger = logging.getLogger(__name__)

DEFAULT_LIBRARY_ROOT = Path("data/library")
PAIR_ROLE_INTERIOR = "interior"
PAIR_ROLE_COVER = "cover"
PAIR_ROLE_GUIDE = "guide"
SET_MODE_SINGLE = "single"
SET_MODE_MIXED = "mixed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(text: str, *, max_len: int = 48) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")[:max_len] or "item"


def make_set_id(
    *,
    mode: str,
    title: str,
    category: str | None = None,
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    short = uuid.uuid4().hex[:6]
    if mode == SET_MODE_MIXED:
        return f"mixed-{_slugify(title)}-{stamp}-{short}"
    cat = _slugify(category or "general", max_len=24)
    return f"{cat}-{_slugify(title)}-{stamp}-{short}"


def make_pair_id(set_id: str, index: int) -> str:
    return f"{set_id}/p{index:02d}"


@dataclass
class PairRecord:
    """Linked colour image + outline product unit."""

    pair_id: str
    set_id: str
    index: int
    category: str | None
    subject: str | None
    role: str = PAIR_ROLE_INTERIOR
    aspect: str | None = None
    scene: str | None = None
    style: str | None = None
    n_colours: int = 0
    paths: dict[str, str] = field(default_factory=dict)
    colourways: list[str] = field(default_factory=lambda: [COLOURWAY_NATURAL.id])
    created_at: str = field(default_factory=_utc_now)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PairRecord:
        return cls(
            pair_id=str(data["pair_id"]),
            set_id=str(data["set_id"]),
            index=int(data["index"]),
            category=data.get("category"),
            subject=data.get("subject"),
            role=str(data.get("role") or PAIR_ROLE_INTERIOR),
            aspect=data.get("aspect"),
            scene=data.get("scene"),
            style=data.get("style"),
            n_colours=int(data.get("n_colours") or 0),
            paths=dict(data.get("paths") or {}),
            colourways=list(data.get("colourways") or [COLOURWAY_NATURAL.id]),
            created_at=str(data.get("created_at") or _utc_now()),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class SetRecord:
    """Collated group of pairs (single-subject or mixed)."""

    set_id: str
    title: str
    mode: str  # single | mixed
    style: str | None = None
    query: str | None = None
    categories: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    pair_ids: list[str] = field(default_factory=list)
    colourways: list[str] = field(
        default_factory=lambda: [c.id for c in list_colourways()]
    )
    created_at: str = field(default_factory=_utc_now)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> SetRecord:
        return cls(
            set_id=str(data["set_id"]),
            title=str(data.get("title") or data["set_id"]),
            mode=str(data.get("mode") or SET_MODE_SINGLE),
            style=data.get("style"),
            query=data.get("query"),
            categories=list(data.get("categories") or []),
            subjects=list(data.get("subjects") or []),
            pair_ids=list(data.get("pair_ids") or []),
            colourways=list(
                data.get("colourways") or [c.id for c in list_colourways()]
            ),
            created_at=str(data.get("created_at") or _utc_now()),
            notes=str(data.get("notes") or ""),
        )


class AssetLibrary:
    """Filesystem-backed catalog of sets and linked plate/outline pairs."""

    def __init__(self, root: Path | str = DEFAULT_LIBRARY_ROOT) -> None:
        self.root = Path(root)
        self.sets_dir = self.root / "sets"
        self.index_path = self.root / "index.json"
        self.sets_dir.mkdir(parents=True, exist_ok=True)

    # --- paths -----------------------------------------------------------------

    def set_dir(self, set_id: str) -> Path:
        return self.sets_dir / set_id

    def pair_dir(self, pair_id: str) -> Path:
        set_id, _, leaf = pair_id.partition("/")
        if not set_id or not leaf:
            raise ValueError(f"Invalid pair_id {pair_id!r}")
        return self.set_dir(set_id) / "pairs" / leaf

    # --- index -----------------------------------------------------------------

    def _load_index(self) -> dict:
        if not self.index_path.exists():
            return {"version": 1, "sets": []}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save_index(self, payload: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    def _touch_index(self, record: SetRecord) -> None:
        payload = self._load_index()
        entries = [
            item
            for item in payload.get("sets", [])
            if item.get("set_id") != record.set_id
        ]
        entries.append(
            {
                "set_id": record.set_id,
                "title": record.title,
                "mode": record.mode,
                "categories": record.categories,
                "subjects": record.subjects,
                "n_pairs": len(record.pair_ids),
                "created_at": record.created_at,
            }
        )
        entries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        payload["sets"] = entries
        payload["version"] = 1
        self._save_index(payload)

    # --- set CRUD --------------------------------------------------------------

    def create_set(
        self,
        *,
        title: str,
        mode: str = SET_MODE_SINGLE,
        style: str | None = None,
        query: str | None = None,
        categories: list[str] | None = None,
        subjects: list[str] | None = None,
        set_id: str | None = None,
        notes: str = "",
        colourways: list[str] | None = None,
    ) -> SetRecord:
        mode_key = (mode or SET_MODE_SINGLE).lower().strip()
        if mode_key not in {SET_MODE_SINGLE, SET_MODE_MIXED}:
            raise ValueError("mode must be 'single' or 'mixed'")
        cats = list(categories or [])
        primary = cats[0] if cats else None
        sid = set_id or make_set_id(mode=mode_key, title=title, category=primary)
        record = SetRecord(
            set_id=sid,
            title=title,
            mode=mode_key,
            style=style,
            query=query,
            categories=cats,
            subjects=list(subjects or []),
            pair_ids=[],
            colourways=list(colourways or [c.id for c in list_colourways()]),
            notes=notes,
        )
        path = self.set_dir(sid)
        path.mkdir(parents=True, exist_ok=True)
        (path / "pairs").mkdir(exist_ok=True)
        self.save_set(record)
        return record

    def save_set(self, record: SetRecord) -> Path:
        path = self.set_dir(record.set_id)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "set.json"
        target.write_text(
            json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        self._touch_index(record)
        return target

    def load_set(self, set_id: str) -> SetRecord:
        path = self.set_dir(set_id) / "set.json"
        if not path.exists():
            raise FileNotFoundError(f"Unknown set {set_id!r}")
        return SetRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_sets(self) -> list[dict]:
        return list(self._load_index().get("sets") or [])

    # --- pair I/O --------------------------------------------------------------

    def save_pair(self, record: PairRecord) -> Path:
        path = self.pair_dir(record.pair_id)
        path.mkdir(parents=True, exist_ok=True)
        target = path / "meta.json"
        target.write_text(
            json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return target

    def load_pair(self, pair_id: str) -> PairRecord:
        path = self.pair_dir(pair_id) / "meta.json"
        if not path.exists():
            raise FileNotFoundError(f"Unknown pair {pair_id!r}")
        return PairRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_labels(self, pair_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (labels HxW int32, palette Nx3 uint8) for editable recolour."""
        pair_path = self.pair_dir(pair_id)
        labels_path = pair_path / "labels.npy"
        palette_path = pair_path / "palette.json"
        if not labels_path.exists() or not palette_path.exists():
            raise FileNotFoundError(
                f"Pair {pair_id!r} is missing editable labels/palette"
            )
        labels = np.load(labels_path)
        payload = json.loads(palette_path.read_text(encoding="utf-8"))
        colours = payload.get("colours") or []
        palette = np.array(
            [row["rgb"] for row in colours],
            dtype=np.uint8,
        )
        return labels.astype(np.int32), palette

    def _write_palette_json(self, path: Path, palette: np.ndarray) -> None:
        colours = [
            {
                "number": i + 1,
                "rgb": [int(c) for c in row],
                "hex": f"#{int(row[0]):02X}{int(row[1]):02X}{int(row[2]):02X}",
            }
            for i, row in enumerate(np.asarray(palette, dtype=np.uint8))
        ]
        path.write_text(
            json.dumps({"n_colours": len(colours), "colours": colours}, indent=2)
            + "\n",
            encoding="utf-8",
        )

    def add_pair_from_result(
        self,
        set_id: str,
        result: ColourByNumbersResult,
        *,
        illustration: Image.Image | None = None,
        index: int | None = None,
        category: str | None = None,
        subject: str | None = None,
        role: str = PAIR_ROLE_INTERIOR,
        aspect: str | None = None,
        scene: str | None = None,
        style: str | None = None,
        colourways: list[str] | None = None,
        notes: str = "",
    ) -> PairRecord:
        """Persist a pipeline result as a linked plate/outline pair."""
        record = self.load_set(set_id)
        next_index = index if index is not None else (len(record.pair_ids) + 1)
        pair_id = make_pair_id(set_id, next_index)
        pair_path = self.pair_dir(pair_id)
        pair_path.mkdir(parents=True, exist_ok=True)

        labels = np.asarray(result.page.labels, dtype=np.int32)
        palette = np.asarray(result.page.palette, dtype=np.uint8)
        np.save(pair_path / "labels.npy", labels)
        self._write_palette_json(pair_path / "palette.json", palette)

        plate = preview_from_labels(labels, palette)
        plate.save(pair_path / "plate.png")
        result.page.outline.save(pair_path / "outline.png")
        result.page.legend.save(pair_path / "legend.png")
        result.printable.save(pair_path / "page.png")
        if result.page.outline_svg:
            (pair_path / "outline.svg").write_text(
                result.page.outline_svg, encoding="utf-8"
            )
        if result.page.plate_svg:
            (pair_path / "plate.svg").write_text(
                result.page.plate_svg, encoding="utf-8"
            )
        if illustration is not None:
            illustration.convert("RGB").save(pair_path / "illustration.png")
        elif result.source is not None:
            result.source.convert("RGB").save(pair_path / "illustration.png")

        ways = list(colourways or record.colourways or [COLOURWAY_NATURAL.id])
        pair = PairRecord(
            pair_id=pair_id,
            set_id=set_id,
            index=next_index,
            category=category,
            subject=subject or result.subject_type_label,
            role=role,
            aspect=aspect,
            scene=scene,
            style=style or record.style,
            n_colours=int(palette.shape[0]),
            paths={
                "labels": "labels.npy",
                "palette": "palette.json",
                "plate": "plate.png",
                "outline": "outline.png",
                "legend": "legend.png",
                "page": "page.png",
                "illustration": "illustration.png",
                "outline_svg": "outline.svg",
                "plate_svg": "plate.svg",
            },
            colourways=ways,
            notes=notes,
        )
        self.save_pair(pair)

        # Render configured colourways (natural is the base plate).
        for way in ways:
            self.render_pair_colourway(pair_id, way)

        if pair_id not in record.pair_ids:
            record.pair_ids.append(pair_id)
        if category and category not in record.categories:
            record.categories.append(category)
        subj = pair.subject
        if subj and subj not in record.subjects:
            record.subjects.append(subj)
        self.save_set(record)
        return pair

    def add_pair_from_page(
        self,
        set_id: str,
        page: GeneratedPage,
        *,
        index: int | None = None,
        role: str = PAIR_ROLE_INTERIOR,
        aspect: str | None = None,
        scene: str | None = None,
        style: str | None = None,
        colourways: list[str] | None = None,
        notes: str = "",
    ) -> PairRecord:
        return self.add_pair_from_result(
            set_id,
            page.result,
            illustration=page.illustration.image,
            index=index,
            category=page.subject_type.category,
            subject=page.subject_type.label,
            role=role,
            aspect=aspect,
            scene=scene,
            style=style,
            colourways=colourways,
            notes=notes,
        )

    def render_pair_colourway(self, pair_id: str, colourway: str) -> Path:
        """Write ``colourways/<id>/plate.png`` (+ legend) for one colourway."""
        way = resolve_colourway(colourway)
        labels, palette = self.load_labels(pair_id)
        plate, remapped = render_colourway_plate(labels, palette, way)
        out_dir = self.pair_dir(pair_id) / "colourways" / way.id
        out_dir.mkdir(parents=True, exist_ok=True)
        plate_path = out_dir / "plate.png"
        plate.save(plate_path)
        legend = build_legend(remapped, list(range(1, len(remapped) + 1)))
        legend.save(out_dir / "legend.png")
        self._write_palette_json(out_dir / "palette.json", remapped)
        pair = self.load_pair(pair_id)
        if way.id not in pair.colourways:
            pair.colourways.append(way.id)
            self.save_pair(pair)
        return plate_path

    def open_pair_asset(self, pair_id: str, key: str) -> Path:
        pair = self.load_pair(pair_id)
        rel = pair.paths.get(key)
        if not rel:
            raise KeyError(f"Pair {pair_id!r} has no asset {key!r}")
        path = self.pair_dir(pair_id) / rel
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    # --- collation -------------------------------------------------------------

    def compose_mixed_set(
        self,
        *,
        title: str,
        pair_ids: list[str],
        style: str | None = None,
        notes: str = "",
        set_id: str | None = None,
    ) -> SetRecord:
        """Build a mixed set by referencing existing pairs (copied into new set)."""
        if not pair_ids:
            raise ValueError("pair_ids must be non-empty")
        sources = [self.load_pair(pid) for pid in pair_ids]
        cats = sorted({p.category for p in sources if p.category})
        subjects = []
        for pair in sources:
            if pair.subject and pair.subject not in subjects:
                subjects.append(pair.subject)
        record = self.create_set(
            title=title,
            mode=SET_MODE_MIXED,
            style=style,
            categories=cats,
            subjects=subjects,
            set_id=set_id,
            notes=notes,
        )
        for i, src in enumerate(sources, start=1):
            self._copy_pair_into_set(src, record.set_id, index=i)
        return self.load_set(record.set_id)

    def _copy_pair_into_set(
        self, source: PairRecord, dest_set_id: str, *, index: int
    ) -> PairRecord:
        import shutil

        new_id = make_pair_id(dest_set_id, index)
        src_dir = self.pair_dir(source.pair_id)
        dest_dir = self.pair_dir(new_id)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)
        pair = PairRecord(
            pair_id=new_id,
            set_id=dest_set_id,
            index=index,
            category=source.category,
            subject=source.subject,
            role=source.role,
            aspect=source.aspect,
            scene=source.scene,
            style=source.style,
            n_colours=source.n_colours,
            paths=dict(source.paths),
            colourways=list(source.colourways),
            notes=source.notes,
        )
        self.save_pair(pair)
        record = self.load_set(dest_set_id)
        if new_id not in record.pair_ids:
            record.pair_ids.append(new_id)
        if pair.category and pair.category not in record.categories:
            record.categories.append(pair.category)
        if pair.subject and pair.subject not in record.subjects:
            record.subjects.append(pair.subject)
        self.save_set(record)
        return pair


def ingest_generated_set(
    generated,
    *,
    library: AssetLibrary | None = None,
    title: str | None = None,
    style: str | None = None,
    colourways: list[str] | None = None,
) -> SetRecord:
    """Write an accepted ``GeneratedSet`` into the asset library."""
    lib = library or AssetLibrary()
    plan = generated.plan
    set_title = title or plan.subject_type.label
    record = lib.create_set(
        title=set_title,
        mode=SET_MODE_SINGLE,
        style=style,
        query=plan.original_query,
        categories=[plan.subject_type.category] if plan.subject_type.category else [],
        subjects=[plan.subject_type.label],
        colourways=colourways,
    )
    accepted = [
        item
        for item in generated.results
        if item.status == "accepted" and item.page is not None
    ]
    for item in accepted:
        lib.add_pair_from_page(
            record.set_id,
            item.page,
            index=item.slot.index,
            aspect=item.slot.aspect,
            scene=item.slot.scene,
            style=style,
            colourways=colourways,
        )
    # Refresh after adds
    final = lib.load_set(record.set_id)
    # Keep a pointer beside the classic set output if present later.
    return final


_SAFE_SET_ID = re.compile(r"^[a-z0-9][a-z0-9._/-]*$", re.I)


def validate_set_id(set_id: str) -> str:
    if not _SAFE_SET_ID.match(set_id) or ".." in set_id:
        raise ValueError(f"Unsafe set_id {set_id!r}")
    return set_id
