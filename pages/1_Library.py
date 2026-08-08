"""Streamlit multipage: browse the asset library."""

from __future__ import annotations

import streamlit as st

from ui_library import render_library_browser

st.set_page_config(
    page_title="Asset library — Colour by Numbers",
    page_icon="📚",
    layout="wide",
)

render_library_browser(library_root="data/library", auto_seed_samples=True)
