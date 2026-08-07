"""Vector (SVG) export for flat colour plates and numbered outlines."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from scipy import ndimage
from skimage import measure


def _rgb_hex(rgb: np.ndarray | tuple[int, int, int]) -> str:
    r, g, b = (int(c) for c in rgb[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def _contour_paths(
    mask: np.ndarray,
    *,
    simplify_tolerance: float = 0.85,
) -> list[str]:
    contours = measure.find_contours(mask.astype(float), 0.5)
    paths: list[str] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        simplified = measure.approximate_polygon(contour, tolerance=simplify_tolerance)
        parts: list[str] = []
        for i, (y, x) in enumerate(simplified):
            cmd = "M" if i == 0 else "L"
            parts.append(f"{cmd}{float(x):.2f},{float(y):.2f}")
        parts.append("Z")
        paths.append(" ".join(parts))
    return paths


def iter_component_masks(labels: np.ndarray) -> list[tuple[int, np.ndarray]]:
    structure = np.ones((3, 3), dtype=bool)
    items: list[tuple[int, np.ndarray]] = []
    for colour in np.unique(labels):
        labelled, count = ndimage.label(labels == colour, structure=structure)
        for comp_id in range(1, count + 1):
            items.append((int(colour), labelled == comp_id))
    return items


def build_colour_plate_svg(
    labels: np.ndarray,
    palette: np.ndarray,
    *,
    simplify_tolerance: float = 0.85,
) -> str:
    """Smooth filled regions — avoids jagged raster colour boundaries."""
    height, width = labels.shape
    root = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
        },
    )
    bg = ET.SubElement(root, "rect", {"width": str(width), "height": str(height), "fill": "#FFFFFF"})
    group = ET.SubElement(root, "g", {"fill-rule": "evenodd"})
    for colour_idx, mask in iter_component_masks(labels):
        fill = _rgb_hex(palette[colour_idx])
        for path_d in _contour_paths(mask, simplify_tolerance=simplify_tolerance):
            ET.SubElement(group, "path", {"d": path_d, "fill": fill})
    return ET.tostring(root, encoding="unicode")


def build_outline_svg(
    labels: np.ndarray,
    regions: list[object],
    *,
    line_width: float = 2.0,
    simplify_tolerance: float = 0.85,
    number_font_scale: float = 1.0,
) -> str:
    """Numbered outline with round vector strokes (print-friendly)."""
    height, width = labels.shape
    root = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
        },
    )
    ET.SubElement(root, "rect", {"width": str(width), "height": str(height), "fill": "#FFFFFF"})
    stroke_group = ET.SubElement(
        root,
        "g",
        {
            "fill": "#FFFFFF",
            "stroke": "#000000",
            "stroke-width": f"{line_width:.2f}",
            "stroke-linejoin": "round",
            "stroke-linecap": "round",
        },
    )
    for _colour_idx, mask in iter_component_masks(labels):
        for path_d in _contour_paths(mask, simplify_tolerance=simplify_tolerance):
            ET.SubElement(stroke_group, "path", {"d": path_d})

    text_group = ET.SubElement(root, "g")
    base_font = max(12.0, min(width, height) * 0.035 * number_font_scale)
    for region in sorted(regions, key=lambda item: item.area, reverse=True):
        side = max(1.0, float(region.area) ** 0.5)
        font_size = max(7.0, min(base_font, side * 0.45))
        x, y = region.centroid
        text = str(region.number)
        halo = ET.SubElement(
            text_group,
            "text",
            {
                "x": f"{x:.2f}",
                "y": f"{y:.2f}",
                "text-anchor": "middle",
                "dominant-baseline": "middle",
                "font-family": "Source Sans 3, DejaVu Sans, Arial, sans-serif",
                "font-size": f"{font_size:.1f}",
                "font-weight": "700",
                "fill": "#FFFFFF",
                "stroke": "#FFFFFF",
                "stroke-width": f"{max(2.0, font_size / 5):.1f}",
            },
        )
        halo.text = text
        label = ET.SubElement(
            text_group,
            "text",
            {
                "x": f"{x:.2f}",
                "y": f"{y:.2f}",
                "text-anchor": "middle",
                "dominant-baseline": "middle",
                "font-family": "Source Sans 3, DejaVu Sans, Arial, sans-serif",
                "font-size": f"{font_size:.1f}",
                "font-weight": "700",
                "fill": "#000000",
            },
        )
        label.text = text
    return ET.tostring(root, encoding="unicode")
