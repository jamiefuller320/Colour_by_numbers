"""Streamlit UI for searching images and building colour-by-numbers pages."""

from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from colour_by_numbers.pipeline import COMPLEXITY_PRESETS, create_colour_by_numbers
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
    "colouring book. Colour plates are the focus for now — outline demos are "
    "paused while the palette work is refined."
)

PRESET_NAMES = ["raw", "fine", "light", "medium", "simple"]

with st.sidebar:
    st.header("Settings")

    st.subheader("Palette & frame")
    palette_mode = st.selectbox(
        "Palette",
        options=["standard", "free"],
        index=0,
        help="standard = fixed 32-colour colouring set; free = adaptive median-cut.",
    )
    n_colours = st.slider(
        "Colours used (from palette)",
        min_value=8,
        max_value=32,
        value=32,
        help="Standard mode draws from the fixed 32-colour set.",
    )
    min_adjacent_delta_e = st.slider(
        "Min adjacent colour ΔE",
        min_value=0.0,
        max_value=40.0,
        value=18.0,
        step=1.0,
        help="Merge touching sections whose paints are closer than this.",
    )
    colour_refine = st.checkbox(
        "Colour-refine subject mask",
        value=True,
        help="Snap silhouette using subject vs background colour contrast.",
    )
    enforce_contrast = st.checkbox(
        "Require subject/background colour contrast",
        value=False,
        help="Reject plates where subject and background colours are too similar.",
    )
    min_subject_bg_contrast = st.slider(
        "Min subject/bg ΔE",
        min_value=8.0,
        max_value=50.0,
        value=22.0,
        step=1.0,
        disabled=not enforce_contrast,
    )
    max_size = st.slider(
        "Processing max edge (px)",
        min_value=400,
        max_value=1600,
        value=900,
        step=50,
        help="Downscale after the native A4 resolution check.",
    )
    structure_size = st.number_input(
        "Structure size override (0 = preset)",
        min_value=0,
        max_value=1600,
        value=0,
        step=20,
    )
    line_width = st.number_input(
        "Outline line width (0 = preset)",
        min_value=0,
        max_value=6,
        value=0,
        step=1,
    )

    st.subheader("Subject")
    subject_mode = st.selectbox(
        "Subject engine",
        options=["dual", "isolate", "off"],
        index=0,
        help=(
            "dual: 80% fill crop + fine on subject / light on background. "
            "isolate: flat background. off: full frame."
        ),
    )
    subject_fill = st.slider(
        "Subject fill of frame",
        min_value=0.50,
        max_value=0.95,
        value=0.80,
        step=0.05,
    )
    firm_border = st.checkbox(
        "Firm subject borders (hard mask)",
        value=True,
        help="Binary mask from the original subject crop — no soft/blurred silhouette.",
    )
    subject_autocrop = st.checkbox("Autocrop to subject fill", value=True)
    subject_model = st.selectbox(
        "rembg model",
        options=["u2net", "u2netp", "u2net_human_seg", "isnet-general-use"],
        index=0,
    )

    st.subheader("A4 print filter")
    enforce_a4 = st.checkbox(
        "Require adequate A4 print resolution",
        value=True,
        help="Reject subject plates that would print softer than the DPI floor.",
    )
    min_a4_dpi = st.slider(
        "Minimum A4 DPI",
        min_value=72,
        max_value=300,
        value=150,
        step=6,
        disabled=not enforce_a4,
    )

    st.subheader("Complexity")
    complexity = st.selectbox(
        "Uniform complexity (off / isolate)",
        options=PRESET_NAMES,
        index=PRESET_NAMES.index("fine"),
    )
    subject_complexity = st.selectbox(
        "Subject complexity (dual)",
        options=PRESET_NAMES,
        index=PRESET_NAMES.index("fine"),
    )
    background_complexity = st.selectbox(
        "Background complexity (dual)",
        options=PRESET_NAMES,
        index=PRESET_NAMES.index("light"),
    )

    with st.expander("Blur / prefilter overrides", expanded=False):
        blur_radius = st.number_input(
            "Uniform blur radius (-1 = preset)",
            min_value=-1.0,
            max_value=5.0,
            value=-1.0,
            step=0.1,
        )
        subject_blur_radius = st.number_input(
            "Subject blur radius (-1 = preset)",
            min_value=-1.0,
            max_value=5.0,
            value=-1.0,
            step=0.1,
        )
        background_blur_radius = st.number_input(
            "Background blur radius (-1 = preset)",
            min_value=-1.0,
            max_value=5.0,
            value=-1.0,
            step=0.1,
        )

    with st.expander("Region simplify overrides", expanded=False):
        st.caption("0 / -1 means use the complexity preset.")
        min_region_area = st.number_input(
            "Min region area (0 = preset)", min_value=0, max_value=50_000, value=0
        )
        max_regions = st.number_input(
            "Max regions (0 = preset)", min_value=0, max_value=5000, value=0
        )
        smooth_radius = st.number_input(
            "Smooth radius (-1 = preset)", min_value=-1, max_value=8, value=-1
        )
        morph_radius = st.number_input(
            "Morph radius (-1 = preset)", min_value=-1, max_value=8, value=-1
        )
        boundary_sigma = st.number_input(
            "Boundary sigma (-1 = preset)",
            min_value=-1.0,
            max_value=5.0,
            value=-1.0,
            step=0.1,
        )
        st.markdown("**Subject zone**")
        subject_min_region_area = st.number_input(
            "Subject min region area (0 = preset)",
            min_value=0,
            max_value=50_000,
            value=0,
        )
        subject_max_regions = st.number_input(
            "Subject max regions (0 = preset)",
            min_value=0,
            max_value=5000,
            value=0,
        )
        subject_smooth_radius = st.number_input(
            "Subject smooth radius (-1 = preset)",
            min_value=-1,
            max_value=8,
            value=-1,
        )
        subject_morph_radius = st.number_input(
            "Subject morph radius (-1 = preset)",
            min_value=-1,
            max_value=8,
            value=-1,
        )
        subject_boundary_sigma = st.number_input(
            "Subject boundary sigma (-1 = preset)",
            min_value=-1.0,
            max_value=5.0,
            value=-1.0,
            step=0.1,
        )
        st.markdown("**Background zone**")
        background_min_region_area = st.number_input(
            "Background min region area (0 = preset)",
            min_value=0,
            max_value=50_000,
            value=0,
        )
        background_max_regions = st.number_input(
            "Background max regions (0 = preset)",
            min_value=0,
            max_value=5000,
            value=0,
        )
        background_smooth_radius = st.number_input(
            "Background smooth radius (-1 = preset)",
            min_value=-1,
            max_value=8,
            value=-1,
        )
        background_morph_radius = st.number_input(
            "Background morph radius (-1 = preset)",
            min_value=-1,
            max_value=8,
            value=-1,
        )
        background_boundary_sigma = st.number_input(
            "Background boundary sigma (-1 = preset)",
            min_value=-1.0,
            max_value=5.0,
            value=-1.0,
            step=0.1,
        )

    source_mode = st.radio("Image source", ["Web search", "Upload file"], index=0)


def _none_if(value, sentinel):
    return None if value == sentinel else value


def _pipeline_kwargs() -> dict:
    return dict(
        n_colours=n_colours,
        palette_mode=palette_mode,
        min_adjacent_delta_e=float(min_adjacent_delta_e),
        colour_refine=colour_refine,
        min_subject_bg_contrast=float(min_subject_bg_contrast) if enforce_contrast else None,
        max_size=max_size,
        complexity=complexity,
        subject_mode=subject_mode,
        subject_fill=subject_fill,
        subject_autocrop=subject_autocrop,
        subject_model=subject_model,
        subject_complexity=subject_complexity,
        background_complexity=background_complexity,
        firm_border=firm_border,
        min_a4_dpi=float(min_a4_dpi) if enforce_a4 else None,
        blur_radius=_none_if(float(blur_radius), -1.0),
        subject_blur_radius=_none_if(float(subject_blur_radius), -1.0),
        background_blur_radius=_none_if(float(background_blur_radius), -1.0),
        structure_size=_none_if(int(structure_size), 0),
        line_width=_none_if(int(line_width), 0),
        min_region_area=_none_if(int(min_region_area), 0),
        max_regions=_none_if(int(max_regions), 0),
        smooth_radius=_none_if(int(smooth_radius), -1),
        morph_radius=_none_if(int(morph_radius), -1),
        boundary_sigma=_none_if(float(boundary_sigma), -1.0),
        subject_min_region_area=_none_if(int(subject_min_region_area), 0),
        subject_max_regions=_none_if(int(subject_max_regions), 0),
        subject_smooth_radius=_none_if(int(subject_smooth_radius), -1),
        subject_morph_radius=_none_if(int(subject_morph_radius), -1),
        subject_boundary_sigma=_none_if(float(subject_boundary_sigma), -1.0),
        background_min_region_area=_none_if(int(background_min_region_area), 0),
        background_max_regions=_none_if(int(background_max_regions), 0),
        background_smooth_radius=_none_if(int(background_smooth_radius), -1),
        background_morph_radius=_none_if(int(background_morph_radius), -1),
        background_boundary_sigma=_none_if(float(background_boundary_sigma), -1.0),
    )


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if source_mode == "Web search":
    query = st.text_input("Search for images", placeholder="dogs, aircraft, sailboat…")
    discover_types = st.checkbox(
        "Discover specific types for broad categories",
        value=True,
        help="e.g. dogs → pug / golden retriever / … before searching photos.",
    )
    illustrate_first = st.checkbox(
        "Generate illustration first",
        value=False,
        help=(
            "Build a flat colouring-book illustration from a type-specific "
            "reference photo, then convert (local stylize backend)."
        ),
    )
    illustration_backend = st.selectbox(
        "Illustration backend",
        options=["local_stylize", "pollinations", "openai"],
        index=0,
        disabled=not illustrate_first,
        help=(
            "pollinations = free text-to-image (no subscription). "
            "openai requires OPENAI_API_KEY."
        ),
    )
    col_a, _col_b = st.columns([1, 3])
    with col_a:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    for key, default in (
        ("hits", []),
        ("query", ""),
        ("type_discovery", None),
        ("chosen_type", None),
    ):
        if key not in st.session_state:
            st.session_state[key] = default

    if search_clicked and query.strip():
        with st.spinner("Discovering types / searching…"):
            try:
                st.session_state.query = query.strip()
                st.session_state.hits = []
                st.session_state.chosen_type = None
                if discover_types:
                    from colour_by_numbers.discover import discover_subject_types

                    discovery = discover_subject_types(
                        query.strip(), probe_search=True, max_types=10
                    )
                    st.session_state.type_discovery = discovery
                    if discovery.skipped or (
                        len(discovery.types) == 1 and discovery.types[0].already_specific
                    ):
                        # Go straight to image search for the single/refined query.
                        chosen = discovery.types[0]
                        st.session_state.chosen_type = chosen
                        st.session_state.hits = search_images(
                            chosen.search_query,
                            max_results=8,
                            min_a4_dpi=float(min_a4_dpi) if enforce_a4 else None,
                            contrast_bias=True,
                        )
                else:
                    st.session_state.type_discovery = None
                    st.session_state.hits = search_images(
                        query.strip(),
                        max_results=8,
                        min_a4_dpi=float(min_a4_dpi) if enforce_a4 else None,
                        contrast_bias=True,
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Search failed: {exc}")
                st.session_state.hits = []
                st.session_state.type_discovery = None

    discovery = st.session_state.type_discovery
    if (
        discovery is not None
        and not discovery.skipped
        and discovery.needs_choice
        and st.session_state.chosen_type is None
    ):
        st.subheader(f"Choose a type of “{discovery.category}”")
        st.caption(
            "Broad categories are narrowed to a specific type so pages stay "
            "recognisable (e.g. pug vs golden retriever)."
        )
        type_labels = [
            f"{t.label}  (score {t.score:.1f})"
            + (f" — {t.evidence[0]}" if t.evidence else "")
            for t in discovery.types
        ]
        type_idx = st.selectbox(
            "Specific type",
            options=list(range(len(discovery.types))),
            format_func=lambda i: type_labels[i],
        )
        if st.button("Search photos for this type", type="primary"):
            chosen = discovery.types[type_idx]
            st.session_state.chosen_type = chosen
            if illustrate_first:
                with st.spinner(
                    f"Illustrating “{chosen.label}” via {illustration_backend}…"
                ):
                    try:
                        from colour_by_numbers.generate import generate_colouring_page

                        page = generate_colouring_page(
                            st.session_state.query,
                            subject_type=chosen.label,
                            discover_types=False,
                            backend=illustration_backend,
                            n_colours=n_colours,
                            illustration_colours=min(n_colours, 16),
                            complexity=complexity,
                            subject_mode="off",
                            palette_mode=palette_mode,
                            min_adjacent_delta_e=float(min_adjacent_delta_e),
                            firm_border=firm_border,
                            max_size=max_size,
                        )
                        st.session_state.illustration = page.illustration
                        st.session_state.result = page.result
                        st.session_state.hits = []
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Illustration failed: {exc}")
            else:
                with st.spinner(f"Searching for “{chosen.search_query}”…"):
                    try:
                        st.session_state.hits = search_images(
                            chosen.search_query,
                            max_results=8,
                            min_a4_dpi=float(min_a4_dpi) if enforce_a4 else None,
                            contrast_bias=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Search failed: {exc}")
                        st.session_state.hits = []

    chosen = st.session_state.chosen_type
    hits = st.session_state.hits
    if chosen is not None:
        st.info(
            f"Specific type: **{chosen.label}** · search “{chosen.search_query}”"
        )

    if hits:
        heading = st.session_state.query
        if chosen is not None:
            heading = chosen.label
        st.subheader(f"Results for “{heading}”")
        labels = []
        for i, hit in enumerate(hits):
            size_bit = ""
            if hit.width and hit.height:
                size_bit = f" · {hit.width}×{hit.height}"
            labels.append(f"{i + 1}. {hit.title[:70]}{size_bit}")
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
                        source_hit=hit,
                        **_pipeline_kwargs(),
                    )
                    if chosen is not None:
                        from colour_by_numbers.pipeline import ColourByNumbersResult

                        result = ColourByNumbersResult(
                            source=result.source,
                            quantized=result.quantized,
                            page=result.page,
                            printable=result.printable,
                            source_hit=result.source_hit,
                            complexity=result.complexity,
                            prepared=result.prepared,
                            subject_mask=result.subject_mask,
                            subject_mode=result.subject_mode,
                            subject_complexity=result.subject_complexity,
                            background_complexity=result.background_complexity,
                            print_dpi=result.print_dpi,
                            firm_border=result.firm_border,
                            palette_mode=result.palette_mode,
                            subject_bg_contrast=result.subject_bg_contrast,
                            min_adjacent_delta_e=result.min_adjacent_delta_e,
                            subject_type_label=chosen.label,
                            subject_type_query=chosen.search_query,
                        )
                    st.session_state.result = result
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Conversion failed: {exc}")
    elif search_clicked and st.session_state.chosen_type is None and (
        discovery is None or discovery.skipped or not discovery.needs_choice
    ):
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
                try:
                    result = create_colour_by_numbers(image, **_pipeline_kwargs())
                    st.session_state.result = result
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Conversion failed: {exc}")


result = st.session_state.get("result")
illustration = st.session_state.get("illustration")
if illustration is not None:
    st.divider()
    st.subheader("Generated illustration")
    st.caption(
        f"Backend “{illustration.backend}”"
        + (f" · type “{illustration.subject_type_label}”" if illustration.subject_type_label else "")
    )
    st.image(illustration.image, use_container_width=True)
    if illustration.reference_url:
        st.caption(f"Reference photo: {illustration.reference_url}")

if result is not None:
    st.divider()
    st.subheader("Colour plate results")
    bits = [
        f"Complexity “{result.complexity}”",
        f"subject “{result.subject_mode}”",
        f"{result.quantized.n_colours} colours",
    ]
    if result.print_dpi is not None:
        bits.append(f"~{result.print_dpi:.0f} DPI on A4")
    if getattr(result, "subject_type_label", None):
        bits.append(f"type “{result.subject_type_label}”")
    if result.firm_border:
        bits.append("firm borders")
    if result.page.simplification is not None:
        stats = result.page.simplification
        bits.append(f"{stats.regions_before} → {stats.regions_after} regions")
    st.caption(" · ".join(bits))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Source")
        st.image(result.source, use_container_width=True)
    with c2:
        if result.prepared is not None:
            st.caption("Subject plate (native crop)")
            st.image(result.prepared, use_container_width=True)
        else:
            st.caption("Source (no subject crop)")
            st.image(result.source, use_container_width=True)
    with c3:
        st.caption(f"Colour preview ({result.quantized.n_colours} colours, {result.palette_mode})")
        st.image(result.quantized.preview, use_container_width=True)

    with st.expander("Outline / printable page (preview only — demos paused)", expanded=False):
        st.image(result.page.outline, caption="Numbered outline", use_container_width=True)
        st.image(result.printable, caption="Printable page", use_container_width=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Download colour preview PNG",
            data=_image_to_png_bytes(result.quantized.preview),
            file_name="colour_preview.png",
            mime="image/png",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download outline PNG",
            data=_image_to_png_bytes(result.page.outline),
            file_name="outline.png",
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
        meta = [f"Source: {hit.url}"]
        if hit.provider:
            meta.append(f"via {hit.provider}")
        if hit.license:
            meta.append(f"licence: {hit.license}")
        if hit.width and hit.height:
            meta.append(f"{hit.width}×{hit.height}")
        st.caption(" · ".join(meta))

    with st.expander("Active complexity preset values", expanded=False):
        st.json(
            {
                "uniform": COMPLEXITY_PRESETS.get(complexity),
                "subject": COMPLEXITY_PRESETS.get(subject_complexity),
                "background": COMPLEXITY_PRESETS.get(background_complexity),
            }
        )
