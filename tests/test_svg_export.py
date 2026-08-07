"""Tests for SVG export."""

from __future__ import annotations

import numpy as np

from colour_by_numbers.outline import build_outline_page
from colour_by_numbers.svg_export import build_colour_plate_svg, build_outline_svg


def test_colour_plate_svg_contains_paths() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:, 20:] = 1
    palette = np.array([[30, 30, 30], [220, 200, 160]], dtype=np.uint8)
    svg = build_colour_plate_svg(labels, palette)
    assert "<svg" in svg
    assert "<path" in svg
    assert "#1E1E1E" in svg or "#1e1e1e" in svg.lower()


def test_outline_page_emits_svg() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[:, 20:] = 1
    palette = np.array([[30, 30, 30], [220, 200, 160]], dtype=np.uint8)
    page = build_outline_page(
        labels,
        palette,
        simplify=False,
        line_width=2,
        stroke_mm=0.6,
        export_svg=True,
    )
    assert page.outline_svg is not None
    assert page.plate_svg is not None
    assert "stroke-width" in page.outline_svg
    assert "<text" in page.outline_svg


def test_outline_svg_stroke_width() -> None:
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[:, 15:] = 1
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    svg = build_outline_svg(labels, [], line_width=3.5)
    assert 'stroke-width="3.50"' in svg
