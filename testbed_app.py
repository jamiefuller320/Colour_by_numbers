"""Streamlit test bed for illustration backends (Pollinations / local / OpenAI).

Supports single-plate runs and Phase D set generation (aspect/scene slots).

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
from colour_by_numbers.quality import PHASE_B_MIN_REGION_MM, PHASE_B_PRIMARY_BACKEND
from colour_by_numbers.set_generate import generate_colouring_set
from colour_by_numbers.set_plan import plan_colouring_set


st.set_page_config(
    page_title="Colour by Numbers — Illustration Test Bed",
    page_icon="🧪",
    layout="wide",
)

st.title("Illustration test bed")
st.write(
    "Try illustration backends for colouring pages. "
    "**fal** (primary) needs `FAL_KEY`. "
    "**Local stylize** uses a reference photo. "
    "**OpenAI** / **Pollinations** are optional fallbacks. "
    "Use **Set mode** for Phase D aspect/scene variety. "
    "Browse ingested sets under **Library**. "
    "GitHub Pages is a plate/outline viewer only — generate here or via CLI."
)

ui_mode = st.radio(
    "Workspace",
    options=["Generate", "Library"],
    horizontal=True,
    help="Generate new plates, or browse the on-disk asset library.",
)

if ui_mode == "Library":
    from ui_library import render_library_browser

    render_library_browser(library_root="data/library", auto_seed_samples=True)
    st.stop()

with st.sidebar:
    st.header("Run settings")
    category = st.selectbox(
        "Category",
        options=sorted(CATEGORY_TYPES.keys()),
        index=sorted(CATEGORY_TYPES.keys()).index("dogs"),
    )
    discover = st.checkbox("Rank types from live search", value=True)
    backends = [b for b in AVAILABLE_ILLUSTRATION_BACKENDS if b != "replicate"]
    backend = st.selectbox(
        "Illustration backend",
        options=backends,
        index=(
            backends.index(PHASE_B_PRIMARY_BACKEND)
            if PHASE_B_PRIMARY_BACKEND in backends
            else 0
        ),
        help="Phase B primary = fal (Flux via fal.ai; needs FAL_KEY).",
    )
    fal_model = st.selectbox(
        "fal model",
        options=["fal-ai/flux/schnell", "fal-ai/flux/dev"],
        index=0,
        disabled=backend != "fal",
    )
    pollinations_model = st.selectbox(
        "Pollinations model (legacy)",
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
        "Min colourable block on A4 (mm wide & high)",
        min_value=3.0,
        max_value=12.0,
        value=float(PHASE_B_MIN_REGION_MM),
        step=0.5,
        help=(
            "Each colour fill must be at least this wide and high (fits a tip "
            "of that diameter). Phase B default is 8mm."
        ),
    )
    seed = st.number_input(
        "Seed (−1 = random / 0 base for sets)",
        min_value=-1,
        max_value=999_999,
        value=-1,
        step=1,
    )
    run_cbn = st.checkbox("Also build colour-by-numbers plate", value=True)
    require_quality = st.checkbox(
        "Require Phase B/C quality gate",
        value=False,
        help="Fail the run if the plate checklist does not pass.",
    )

    st.divider()
    st.subheader("Phase D — Set")
    set_mode = st.checkbox(
        "Set mode (varied plates)",
        value=False,
        help=(
            "Plan/generate N aspect/scene plates for the chosen type. "
            "Slot prompts replace the single prompt box."
        ),
        disabled=backend == "local_stylize",
    )
    set_size = st.slider(
        "Set size (plates)",
        min_value=2,
        max_value=8,
        value=4,
        disabled=not set_mode,
    )
    set_attempts = st.slider(
        "Attempts per slot",
        min_value=1,
        max_value=5,
        value=2,
        disabled=not set_mode,
        help="Retries when a plate fails quality or is a near-duplicate.",
    )

    st.divider()
    st.subheader("Subject feedback")
    subject_feedback = st.checkbox(
        "Subject recognition feedback loop",
        value=False,
        help=(
            "Ask whether the plate is recognisable as the requested subject; "
            "revise the prompt and retry. Stores lessons in data/subject_lessons.jsonl."
        ),
        disabled=backend == "local_stylize",
    )
    critique_mode = st.selectbox(
        "Critique mode",
        options=["rules", "openai"],
        index=0,
        disabled=not subject_feedback,
        help=(
            "rules = offline feature cues; openai = vision (needs OPENAI_API_KEY). "
            "Use CLI --critique-mode human for interactive review."
        ),
    )
    max_feedback_attempts = st.slider(
        "Feedback attempts",
        min_value=1,
        max_value=5,
        value=3,
        disabled=not subject_feedback,
    )


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

base_seed = 0 if seed < 0 else int(seed)
planned = None
if set_mode:
    planned = plan_colouring_set(
        category,
        subject_type=chosen.label,
        n_plates=set_size,
        base_seed=base_seed,
        discover_types=False,
    )
    st.markdown("**Set plan** (aspect / scene)")
    for slot in planned.slots:
        st.caption(f"{slot.index:02d}. **{slot.aspect}** — {slot.scene}")
    with st.expander("Slot prompts"):
        for slot in planned.slots:
            st.markdown(f"**{slot.slug}**")
            st.code(slot.prompt)
else:
    default_prompt = illustration_prompt(chosen.label, category=chosen.category)
    prompt = st.text_area("Illustration prompt", value=default_prompt, height=100)

with col_gen:
    if set_mode:
        generate = st.button(
            f"Generate set of {set_size} with {backend}",
            type="primary",
            use_container_width=True,
        )
    else:
        generate = st.button(
            f"Generate with {backend}",
            type="primary",
            use_container_width=True,
        )

if generate and set_mode and planned is not None:
    with st.spinner(
        f"Generating {set_size}-plate set via {backend}… "
        "(Pollinations is slow / rate-limited; several minutes is normal)"
    ):
        try:
            generated = generate_colouring_set(
                category,
                plan=planned,
                attempts_per_slot=set_attempts,
                require_plate_quality=require_quality,
                backend=backend,
                n_colours=n_colours,
                illustration_colours=n_colours,
                illustration_size=illustration_size,
                complexity="fine",
                subject_mode="off",
                fal_model=fal_model,
                pollinations_model=pollinations_model,
                min_region_mm=min_region_mm,
                subject_feedback=subject_feedback,
                critique_mode=critique_mode,
                max_feedback_attempts=max_feedback_attempts,
            )
            st.session_state.testbed_set = generated
            st.session_state.testbed_illustration = None
            st.session_state.testbed_result = None
            accepted = len(generated.accepted)
            st.success(
                f"Set for “{chosen.label}”: {accepted}/{planned.n_plates} accepted"
            )
            if generated.quality is not None:
                if generated.quality.passed:
                    st.info(generated.quality.summary())
                else:
                    st.warning(generated.quality.summary())
            for item in generated.results:
                st.caption(f"{item.slot.slug}: {item.status} — {item.reason}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Set generation failed: {exc}")

elif generate and not set_mode:
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
                fal_model=fal_model,
                pollinations_model=pollinations_model,
                seed=None if seed < 0 else int(seed),
                min_region_mm=min_region_mm,
                check_quality=True,
                require_quality=require_quality,
                subject_feedback=subject_feedback,
                critique_mode=critique_mode,
                max_feedback_attempts=max_feedback_attempts,
            )
            st.session_state.testbed_set = None
            st.session_state.testbed_illustration = page.illustration
            st.session_state.testbed_result = page.result if run_cbn else None
            st.session_state.testbed_type = chosen.label
            st.session_state.testbed_quality = page.quality
            st.session_state.testbed_feedback = page.feedback
            st.success(
                f"Generated “{chosen.label}” via {page.illustration.backend}"
            )
            if page.feedback is not None:
                st.info(page.feedback.notes)
                for i, attempt in enumerate(page.feedback.attempts, start=1):
                    c = attempt.critique
                    st.caption(
                        f"Attempt {i}: recognisable={c.recognisable} "
                        f"confidence={c.confidence:.2f} accepted={attempt.accepted}"
                    )
            if page.quality is not None:
                if page.quality.passed:
                    st.info(page.quality.summary())
                else:
                    st.warning(page.quality.summary())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Generation failed: {exc}")

generated_set = st.session_state.get("testbed_set")
illustration = st.session_state.get("testbed_illustration")
result = st.session_state.get("testbed_result")

if generated_set is not None:
    st.divider()
    st.subheader("Set results")
    st.caption(
        f"Subject “{generated_set.plan.subject_type.label}” · "
        f"{len(generated_set.accepted)}/{generated_set.plan.n_plates} accepted"
    )
    if generated_set.quality is not None:
        with st.expander("Set quality gate", expanded=not generated_set.quality.passed):
            st.text(generated_set.quality.summary())

    for item in generated_set.results:
        status_icon = {"accepted": "✅", "rejected": "⚠️", "error": "❌"}.get(
            item.status, "•"
        )
        with st.expander(
            f"{status_icon} {item.slot.index:02d}. {item.slot.aspect} / "
            f"{item.slot.scene} — {item.status}",
            expanded=item.status == "accepted",
        ):
            st.caption(
                f"{item.reason} · {item.attempts} attempt(s) · seed base {item.slot.seed}"
            )
            if item.page is None:
                st.info("No plate for this slot.")
                continue
            page = item.page
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Illustration**")
                st.image(page.illustration.image, use_container_width=True)
                st.download_button(
                    "Download illustration PNG",
                    data=_png(page.illustration.image),
                    file_name=f"{item.slot.slug}_illustration.png",
                    mime="image/png",
                    key=f"dl_illu_{item.slot.slug}",
                    use_container_width=True,
                )
                if page.illustration.prompt:
                    with st.expander("Prompt used"):
                        st.code(page.illustration.prompt)
            with c2:
                st.markdown("**Colour-by-numbers outline**")
                st.image(page.result.page.outline, use_container_width=True)
                st.caption(
                    f"{len(page.result.page.regions)} numbered blocks · "
                    f"{page.result.quantized.n_colours} colours"
                )
                st.download_button(
                    "Download outline PNG",
                    data=_png(page.result.page.outline),
                    file_name=f"{item.slot.slug}_outline.png",
                    mime="image/png",
                    key=f"dl_out_{item.slot.slug}",
                    use_container_width=True,
                )
                if page.quality is not None:
                    st.text(page.quality.summary())
                with st.expander("Colour preview + key"):
                    st.image(page.result.quantized.preview, use_container_width=True)
                    st.image(page.result.page.legend, use_container_width=True)

elif illustration is not None:
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
            st.markdown("**Colour-by-numbers outline**")
            st.image(result.page.outline, use_container_width=True)
            st.caption(
                f"{len(result.page.regions)} numbered regions · "
                f"{result.quantized.n_colours} colours"
            )
            st.download_button(
                "Download outline PNG",
                data=_png(result.page.outline),
                file_name="colour_by_numbers_outline.png",
                mime="image/png",
                use_container_width=True,
            )
            with st.expander("Colour preview + key"):
                st.image(result.quantized.preview, use_container_width=True)
                st.image(result.page.legend, use_container_width=True)
                st.download_button(
                    "Download printable page (outline + key)",
                    data=_png(result.printable),
                    file_name="colour_by_numbers_page.png",
                    mime="image/png",
                    use_container_width=True,
                )
        else:
            st.info("Colour-by-numbers step skipped.")
