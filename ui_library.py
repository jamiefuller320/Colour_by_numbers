"""Shared Streamlit UI for browsing the on-disk asset library."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from colour_by_numbers.library import AssetLibrary, seed_sample_sets


def _swatch_html(hexes: list[str]) -> str:
    chips = "".join(
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'margin:0 2px 0 0;border-radius:2px;background:{h};'
        f'border:1px solid rgba(0,0,0,0.15);" title="{h}"></span>'
        for h in hexes
    )
    return f'<div style="margin-top:4px;line-height:0;">{chips}</div>'


def render_library_browser(
    *,
    library_root: str | Path = "data/library",
    auto_seed_samples: bool = True,
    key_prefix: str = "lib",
) -> None:
    """Browse library sets as colour thumbnails; open a set gallery on click."""
    st.subheader("Asset library")
    st.caption(
        "Each tile is a set (linked colour plates ↔ numbered outlines). "
        "Click a thumbnail to open its gallery."
    )

    root = Path(library_root)
    lib = AssetLibrary(root)

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        st.text_input(
            "Library root",
            value=str(root),
            key=f"{key_prefix}_root_display",
            disabled=True,
        )
    with col_b:
        if st.button("Refresh", key=f"{key_prefix}_refresh", use_container_width=True):
            st.rerun()
    with col_c:
        seed = st.button(
            "Seed samples",
            key=f"{key_prefix}_seed",
            use_container_width=True,
            help="Copy docs/samples vibrant packs into the library for browsing.",
        )

    if seed or (auto_seed_samples and not lib.list_sets()):
        with st.spinner("Seeding sample sets from docs/samples…"):
            created = seed_sample_sets(lib)
        if seed:
            st.success(f"Library has {len(created)} sample set(s) ready.")

    sets = lib.browse_sets()
    if not sets:
        st.info(
            "No sets in the library yet. Generate with `--library-ingest`, "
            "or click **Seed samples** to load the vibrant demo packs."
        )
        return

    open_key = f"{key_prefix}_open_set"
    if open_key not in st.session_state:
        st.session_state[open_key] = None

    open_id = st.session_state[open_key]
    if open_id:
        _render_set_gallery(lib, open_id, key_prefix=key_prefix, open_key=open_key)
        return

    st.markdown(f"**{len(sets)} set(s)**")
    cols = st.columns(3)
    for i, row in enumerate(sets):
        with cols[i % 3]:
            thumb = row.get("thumbnail")
            if thumb and Path(thumb).exists():
                st.image(thumb, use_container_width=True)
            else:
                st.markdown(
                    '<div style="aspect-ratio:1;background:#eee;'
                    'display:flex;align-items:center;justify-content:center;'
                    'color:#888;">No preview</div>',
                    unsafe_allow_html=True,
                )
            swatches = row.get("thumbnail_colours") or []
            if swatches:
                st.markdown(_swatch_html(swatches), unsafe_allow_html=True)
            cats = ", ".join(row.get("categories") or []) or "—"
            style = row.get("style") or "—"
            st.markdown(f"**{row['title']}**")
            st.caption(
                f"{row['n_pairs']} plate(s) · {style} · {cats}"
            )
            if st.button(
                "Open set",
                key=f"{key_prefix}_open_{row['set_id']}",
                use_container_width=True,
            ):
                st.session_state[open_key] = row["set_id"]
                st.rerun()


def _render_set_gallery(
    lib: AssetLibrary,
    set_id: str,
    *,
    key_prefix: str,
    open_key: str,
) -> None:
    try:
        record = lib.load_set(set_id)
    except FileNotFoundError:
        st.error(f"Unknown set {set_id!r}")
        st.session_state[open_key] = None
        return

    if st.button("← All sets", key=f"{key_prefix}_back"):
        st.session_state[open_key] = None
        st.rerun()

    st.markdown(f"### {record.title}")
    meta_bits = [
        f"`{record.set_id}`",
        record.mode,
        record.style or "no style",
        f"{len(record.pair_ids)} pair(s)",
    ]
    if record.categories:
        meta_bits.append(", ".join(record.categories))
    st.caption(" · ".join(meta_bits))

    pairs = lib.list_pair_previews(set_id)
    if not pairs:
        st.warning("This set has no pairs yet.")
        return

    for pair in pairs:
        label = (
            f"#{pair['index']:02d} — {pair.get('subject') or 'plate'}"
            f" ({pair.get('n_colours') or '?'} colours)"
        )
        with st.expander(label, expanded=len(pairs) <= 3):
            aspect = pair.get("aspect") or ""
            scene = pair.get("scene") or ""
            if aspect or scene:
                st.caption(f"{aspect} · {scene}".strip(" ·"))
            assets = pair.get("assets") or {}
            show = [
                ("plate", "Colour plate"),
                ("outline", "Numbered outline"),
                ("page", "Print page"),
                ("illustration", "Source illustration"),
            ]
            available = [(k, title) for k, title in show if k in assets]
            if not available:
                st.write("No preview assets on disk.")
                continue
            cols = st.columns(min(3, len(available)))
            for i, (key, title) in enumerate(available):
                with cols[i % len(cols)]:
                    path = assets[key]
                    st.markdown(f"**{title}**")
                    try:
                        st.image(Image.open(path), use_container_width=True)
                    except OSError as exc:
                        st.error(f"Could not open {key}: {exc}")
                    with open(path, "rb") as fh:
                        st.download_button(
                            f"Download {key}",
                            data=fh.read(),
                            file_name=Path(path).name,
                            mime="image/png",
                            key=f"{key_prefix}_dl_{pair['pair_id']}_{key}",
                            use_container_width=True,
                        )
