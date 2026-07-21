"""Streamlit UI for searching images and building colour-by-numbers pages."""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from colour_by_numbers.pipeline import create_colour_by_numbers
from colour_by_numbers.search import download_image, search_images


st.set_page_config(
    page_title="Colour by Numbers",
    page_icon="🎨",
    layout="wide",
)

st.title("Colour by Numbers")
st.write(
    "Search the web for a subject, reduce it to a limited palette, simplify "
    "regions into coherent shapes, and export a numbered outline ready for a "
    "colouring book."
)

with st.sidebar:
    st.header("Settings")
    n_colours = st.slider("Number of colours", min_value=4, max_value=32, value=16)
    complexity_options = ["raw", "fine", "light", "medium", "simple"]
    complexity = st.selectbox(
        "Complexity",
        options=complexity_options,
        index=complexity_options.index("fine"),
        help="fine is preferred after subject isolation.",
    )
    subject_mode = st.selectbox(
        "Subject engine",
        options=["isolate", "off"],
        index=0,
        help="isolate uses rembg/U²-Net to cut out the subject and crop it.",
    )
    max_size = st.slider(
        "Max image edge (px)", min_value=400, max_value=1400, value=900, step=50
    )
    source_mode = st.radio("Image source", ["Web search", "Upload file"], index=0)


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if source_mode == "Web search":
    query = st.text_input("Search for images", placeholder="aircraft, dogs, sailboat…")
    col_a, _col_b = st.columns([1, 3])
    with col_a:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    if "hits" not in st.session_state:
        st.session_state.hits = []
    if "query" not in st.session_state:
        st.session_state.query = ""

    if search_clicked and query.strip():
        with st.spinner("Searching the web…"):
            try:
                st.session_state.hits = search_images(query.strip(), max_results=8)
                st.session_state.query = query.strip()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Search failed: {exc}")
                st.session_state.hits = []

    hits = st.session_state.hits
    if hits:
        st.subheader(f"Results for “{st.session_state.query}”")
        labels = [f"{i + 1}. {hit.title[:80]}" for i, hit in enumerate(hits)]
        choice = st.selectbox(
            "Choose an image",
            options=list(range(len(hits))),
            format_func=lambda i: labels[i],
        )
        generate = st.button("Generate colour-by-numbers", type="primary")
        if generate:
            hit = hits[choice]
            with st.spinner("Downloading and converting…"):
                try:
                    image = download_image(hit.url)
                    result = create_colour_by_numbers(
                        image,
                        n_colours=n_colours,
                        max_size=max_size,
                        complexity=complexity,
                        subject_mode=subject_mode,
                        source_hit=hit,
                    )
                    st.session_state.result = result
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Conversion failed: {exc}")
    elif search_clicked:
        st.warning("No images found. Try a different search.")

else:
    uploaded = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp"]
    )
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        if st.button("Generate colour-by-numbers", type="primary"):
            with st.spinner("Converting…"):
                result = create_colour_by_numbers(
                    image,
                    n_colours=n_colours,
                    max_size=max_size,
                    complexity=complexity,
                    subject_mode=subject_mode,
                )
                st.session_state.result = result


result = st.session_state.get("result")
if result is not None:
    st.divider()
    st.subheader("Results")
    if result.page.simplification is not None:
        stats = result.page.simplification
        st.caption(
            f"Complexity “{result.complexity}” · subject “{result.subject_mode}”: "
            f"{stats.regions_before} regions → {stats.regions_after} "
            f"({result.quantized.n_colours} colours)"
        )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Source")
        st.image(result.source, use_container_width=True)
    with c2:
        label = "Subject plate" if result.prepared is not None else "Simplified palette"
        st.caption(f"{label} ({result.quantized.n_colours} colours)")
        st.image(
            result.prepared if result.prepared is not None else result.quantized.preview,
            use_container_width=True,
        )
    with c3:
        st.caption("Numbered outline")
        st.image(result.page.outline, use_container_width=True)

    if result.prepared is not None:
        st.caption(f"16-colour preview ({result.quantized.n_colours} colours)")
        st.image(result.quantized.preview, use_container_width=True)

    st.caption("Printable page (outline + colour key)")
    st.image(result.printable, use_container_width=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Download outline PNG",
            data=_image_to_png_bytes(result.page.outline),
            file_name="outline.png",
            mime="image/png",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download legend PNG",
            data=_image_to_png_bytes(result.page.legend),
            file_name="legend.png",
            mime="image/png",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "Download full page PNG",
            data=_image_to_png_bytes(result.printable),
            file_name="colour_by_numbers_page.png",
            mime="image/png",
            use_container_width=True,
        )

    if result.source_hit:
        hit = result.source_hit
        bits = [f"Source: {hit.url}"]
        if hit.provider:
            bits.append(f"via {hit.provider}")
        if hit.license:
            bits.append(f"licence: {hit.license}")
        st.caption(" · ".join(bits))
