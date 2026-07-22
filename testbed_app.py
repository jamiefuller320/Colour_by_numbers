"""Streamlit test bed for illustration backends (Pollinations / local / OpenAI).

Run with::

    streamlit run testbed_app.py --server.port 8502
"""

from __future__ import annotations

import io

import streamlit as st

from colour_by_numbers.discover import (
    CATEGORY_TYPES,
    SubjectType,
    build_type_search_query,
    discover_subject_types,
)
from colour_by_numbers.generate import generate_colouring_page
from colour_by_numbers.illustrate import (
    AVAILABLE_ILLUSTRATION_BACKENDS,
    illustration_prompt,
)


st.set_page_config(
    page_title="Colour by Numbers — Illustration Test Bed",
    page_icon="🧪",
    layout="wide",
)

st.title("Illustration test bed")
st.write(
    "Try illustration backends for colouring pages. "
    "**Pollinations** needs no paid subscription. "
    "**Local stylize** uses a real reference photo. "
    "**OpenAI** needs `OPENAI_API_KEY`."
)

with st.sidebar:
    st.header("Run settings")
    category = st.selectbox(
        "Category",
        options=sorted(CATEGORY_TYPES.keys()),
        index=sorted(CATEGORY_TYPES.keys()).index("dogs"),
    )
    discover = st.checkbox("Rank types from live search", value=True)
    backend = st.selectbox(
        "Illustration backend",
        options=[b for b in AVAILABLE_ILLUSTRATION_BACKENDS if b != "replicate"],
        index=1 if "pollinations" in AVAILABLE_ILLUSTRATION_BACKENDS else 0,
        help="pollinations = free text-to-image, no subscription.",
    )
    pollinations_model = st.selectbox(
        "Pollinations model",
        options=["flux", "turbo"],
        index=0,
        disabled=backend != "pollinations",
    )
    n_colours = st.slider("Colour-by-numbers colours (8–16)", 8, 16, 12)
    illustration_size = st.slider(
        "Illustration size (px)",
        min_value=512,
        max_value=1280,
        value=768,
        step=128,
        help="Smaller is faster on free Pollinations tier.",
    )
    min_region_mm = st.slider(
        "Min region size on A4 (mm)",
        min_value=3.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="Colouring regions smaller than this are absorbed when printed on A4.",
    )
    seed = st.number_input(
        "Seed (−1 = random)",
        min_value=-1,
        max_value=999_999,
        value=-1,
        step=1,
    )
    run_cbn = st.checkbox("Also build colour-by-numbers plate", value=True)


def _png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


if "types" not in st.session_state:
    st.session_state.types = []

col_disc, col_gen = st.columns([1, 2])
with col_disc:
    if st.button("Discover types", type="secondary", use_container_width=True):
        with st.spinner(f"Discovering types for “{category}”…"):
            try:
                discovery = discover_subject_types(
                    category,
                    max_types=10,
                    probe_search=discover,
                )
                st.session_state.types = discovery.types
            except Exception as exc:  # noqa: BLE001
                st.error(f"Discovery failed: {exc}")
                st.session_state.types = []

types = st.session_state.types
if not types:
    types = [
        SubjectType(
            label=label,
            category=category,
            search_query=build_type_search_query(label, category=category),
            score=0.0,
        )
        for label in CATEGORY_TYPES[category][:8]
    ]
    st.caption("Showing curated shortlist (click Discover types to rank from search).")

type_labels = [
    f"{t.label}" + (f"  · score {t.score:.1f}" if t.score else "") for t in types
]
choice = st.selectbox(
    "Specific type",
    options=list(range(len(types))),
    format_func=lambda i: type_labels[i],
)
chosen = types[choice]

default_prompt = illustration_prompt(chosen.label, category=chosen.category)
prompt = st.text_area("Illustration prompt", value=default_prompt, height=100)

with col_gen:
    generate = st.button(
        f"Generate with {backend}",
        type="primary",
        use_container_width=True,
    )

if generate:
    with st.spinner(f"Generating via {backend}… (Pollinations can take 20–60s)"):
        try:
            page = generate_colouring_page(
                category,
                subject_type=chosen.label,
                discover_types=False,
                backend=backend,
                n_colours=n_colours,
                illustration_colours=n_colours,
                illustration_size=illustration_size,
                complexity="fine",
                subject_mode="off",
                prompt_override=prompt,
                pollinations_model=pollinations_model,
                seed=None if seed < 0 else int(seed),
                min_region_mm=min_region_mm,
            )
            st.session_state.testbed_illustration = page.illustration
            st.session_state.testbed_result = page.result if run_cbn else None
            st.session_state.testbed_type = chosen.label
            st.success(
                f"Generated “{chosen.label}” via {page.illustration.backend}"
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Generation failed: {exc}")

illustration = st.session_state.get("testbed_illustration")
result = st.session_state.get("testbed_result")

if illustration is not None:
    st.divider()
    st.subheader("Results")
    st.caption(
        f"Backend “{illustration.backend}” · type “{illustration.subject_type_label}”"
        + (f" · {illustration.notes}" if illustration.notes else "")
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Illustration**")
        st.image(illustration.image, use_container_width=True)
        st.download_button(
            "Download illustration PNG",
            data=_png(illustration.image),
            file_name=f"{illustration.subject_type_label or 'illustration'}.png",
            mime="image/png",
            use_container_width=True,
        )
        if illustration.prompt:
            with st.expander("Prompt used"):
                st.code(illustration.prompt)
        if illustration.reference_url:
            st.caption(f"Reference: {illustration.reference_url}")
    with c2:
        if result is not None:
            st.markdown("**Colour-by-numbers preview**")
            st.image(result.quantized.preview, use_container_width=True)
            st.download_button(
                "Download colour preview PNG",
                data=_png(result.quantized.preview),
                file_name="colour_preview.png",
                mime="image/png",
                use_container_width=True,
            )
            with st.expander("Outline preview"):
                st.image(result.page.outline, use_container_width=True)
        else:
            st.info("Colour-by-numbers step skipped.")
