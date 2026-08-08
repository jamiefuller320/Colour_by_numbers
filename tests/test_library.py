"""Asset library: linked pairs, sets, and colourways."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from colour_by_numbers.colourways import (
    list_colourways,
    remap_palette,
    render_colourway_plate,
    resolve_colourway,
)
from colour_by_numbers.library import AssetLibrary, SET_MODE_MIXED, SET_MODE_SINGLE
from colour_by_numbers.pipeline import create_colour_by_numbers
from colour_by_numbers.set_plan import plan_mixed_colouring_set


def _tiny_result() -> object:
    image = Image.new("RGB", (96, 96), (240, 240, 245))
    draw = ImageDraw.Draw(image)
    draw.ellipse((18, 18, 78, 78), fill=(210, 90, 40))
    draw.rectangle((40, 40, 55, 55), fill=(40, 120, 160))
    return create_colour_by_numbers(
        image,
        n_colours=8,
        max_size=96,
        subject_mode="off",
        complexity="raw",
        palette_mode="exact",
        min_adjacent_delta_e=0.0,
        min_region_mm=None,
        silhouette_outline=False,
    )


def test_colourways_change_palette_but_keep_shape() -> None:
    palette = np.array(
        [[40, 40, 40], [200, 80, 40], [40, 120, 180], [240, 220, 80]],
        dtype=np.uint8,
    )
    vivid = remap_palette(palette, "vivid")
    pop = remap_palette(palette, "pop_art")
    pastel = remap_palette(palette, "pastel")
    assert vivid.shape == palette.shape
    assert not np.array_equal(vivid, palette)
    assert not np.array_equal(pop, palette)
    assert not np.array_equal(pastel, palette)
    assert resolve_colourway("POP-ART").id == "pop_art"
    assert len(list_colourways()) >= 4


def test_library_stores_linked_pair_and_colourways(tmp_path) -> None:
    lib = AssetLibrary(tmp_path / "library")
    result = _tiny_result()
    record = lib.create_set(
        title="demo dog",
        mode=SET_MODE_SINGLE,
        categories=["dogs"],
        subjects=["demo dog"],
        style="vibrant",
    )
    pair = lib.add_pair_from_result(
        record.set_id,
        result,
        illustration=result.source,
        category="dogs",
        subject="demo dog",
        aspect="portrait",
        scene="plain",
        colourways=["natural", "vivid", "pop_art"],
    )
    assert pair.pair_id.endswith("/p01")
    loaded = lib.load_pair(pair.pair_id)
    assert loaded.n_colours >= 2
    labels, palette = lib.load_labels(pair.pair_id)
    assert labels.ndim == 2
    assert palette.shape[1] == 3
    # Outline + plate both present and linked via meta paths.
    assert lib.open_pair_asset(pair.pair_id, "outline").exists()
    assert lib.open_pair_asset(pair.pair_id, "plate").exists()
    vivid_path = lib.pair_dir(pair.pair_id) / "colourways" / "vivid" / "plate.png"
    assert vivid_path.exists()
    plate, remapped = render_colourway_plate(labels, palette, "vivid")
    assert plate.size == (labels.shape[1], labels.shape[0])
    assert remapped.shape == palette.shape


def test_compose_mixed_set_from_pairs(tmp_path) -> None:
    lib = AssetLibrary(tmp_path / "library")
    result = _tiny_result()
    a = lib.create_set(title="cats", mode=SET_MODE_SINGLE, categories=["cats"])
    b = lib.create_set(title="planes", mode=SET_MODE_SINGLE, categories=["aircraft"])
    pa = lib.add_pair_from_result(
        a.set_id, result, category="cats", subject="tabby", index=1
    )
    pb = lib.add_pair_from_result(
        b.set_id, result, category="aircraft", subject="biplane", index=1
    )
    mixed = lib.compose_mixed_set(
        title="pets and planes",
        pair_ids=[pa.pair_id, pb.pair_id],
    )
    assert mixed.mode == SET_MODE_MIXED
    assert len(mixed.pair_ids) == 2
    assert set(mixed.categories) == {"cats", "aircraft"}
    assert "tabby" in mixed.subjects and "biplane" in mixed.subjects
    # Copies are independent paths under the new set.
    for pid in mixed.pair_ids:
        assert pid.startswith(mixed.set_id)
        assert lib.open_pair_asset(pid, "outline").exists()


def test_plan_mixed_colouring_set_offline() -> None:
    plan = plan_mixed_colouring_set(
        [("dogs", "pug"), ("aircraft", "biplane")],
        plates_per_subject=2,
        discover_types=False,
        style="vibrant",
    )
    assert plan.mode == "mixed"
    assert plan.n_plates == 4
    subjects = {slot.subject_label for slot in plan.slots}
    assert subjects == {"pug", "biplane"}
    assert all(slot.category for slot in plan.slots)


def test_browse_sets_and_pair_previews(tmp_path) -> None:
    from colour_by_numbers.library import seed_sample_sets

    lib = AssetLibrary(tmp_path / "library")
    plate = Image.new("RGB", (40, 40), (210, 90, 40))
    outline = Image.new("RGB", (40, 40), (255, 255, 255))
    record = lib.create_set(
        title="browse me",
        mode=SET_MODE_SINGLE,
        categories=["dogs"],
        subjects=["demo"],
        style="vibrant",
    )
    lib.add_pair_from_images(
        record.set_id,
        plate=plate,
        outline=outline,
        category="dogs",
        subject="demo",
        style="vibrant",
        n_colours=2,
    )
    rows = lib.browse_sets()
    assert len(rows) == 1
    assert rows[0]["title"] == "browse me"
    assert rows[0]["thumbnail"]
    assert rows[0]["n_pairs"] == 1
    assert rows[0]["thumbnail_colours"]
    previews = lib.list_pair_previews(record.set_id)
    assert len(previews) == 1
    assert previews[0]["assets"]["plate"]
    assert previews[0]["assets"]["outline"]


def test_seed_sample_sets_from_docs(tmp_path) -> None:
    from pathlib import Path

    from colour_by_numbers.library import seed_sample_sets

    samples = Path("docs/samples")
    if not (samples / "dogs" / "golden-retriever-vibrant" / "plate.png").exists():
        return
    lib = AssetLibrary(tmp_path / "library")
    created = seed_sample_sets(lib, samples_root=samples)
    assert len(created) >= 1
    browsed = lib.browse_sets()
    assert any(r["set_id"].startswith("sample-") for r in browsed)
    first = next(r for r in browsed if r["set_id"].startswith("sample-"))
    assert first["thumbnail"]
    assert Path(first["thumbnail"]).exists()
    gallery = lib.list_pair_previews(first["set_id"])
    assert gallery and "plate" in gallery[0]["assets"]


def test_publish_pages_library_manifest(tmp_path) -> None:
    import json
    import shutil
    from pathlib import Path

    from colour_by_numbers.library import publish_pages_library

    src = Path("docs/samples/dogs/golden-retriever-vibrant")
    if not (src / "plate.png").exists():
        return
    docs = tmp_path / "docs"
    dest = docs / "samples" / "dogs" / "golden-retriever-vibrant"
    dest.parent.mkdir(parents=True)
    shutil.copytree(src, dest)
    path = publish_pages_library(docs)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert any(
        s["set_id"] == "sample-golden-retriever-vibrant" for s in payload["sets"]
    )
    row = next(
        s for s in payload["sets"] if s["set_id"] == "sample-golden-retriever-vibrant"
    )
    assert row["thumbnail"]
    assert row["thumbnail_colours"]
    assert row["pairs"][0]["assets"]["plate"].endswith("plate.png")
