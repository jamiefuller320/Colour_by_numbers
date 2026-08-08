"""Subject silhouette outline even when fills match the background."""

from __future__ import annotations

import numpy as np

from colour_by_numbers.outline import build_outline_page, silhouette_edge_mask


def test_silhouette_edge_mask_traces_subject_boundary() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    edge = silhouette_edge_mask(mask, width=1)
    assert edge.any()
    assert not edge[20, 20]  # interior
    assert edge[10, 20] or edge[11, 20]


def test_outline_inks_same_colour_subject_background_seam() -> None:
    """A subject square that shares its fill with the background still gets a stroke."""
    colour = (40, 140, 160)
    image_labels = np.zeros((80, 80), dtype=np.int32)
    # Whole canvas one palette index — no label edges at all.
    palette = np.array([colour], dtype=np.uint8)
    silhouette = np.zeros((80, 80), dtype=bool)
    silhouette[20:60, 20:60] = True

    without = build_outline_page(
        image_labels,
        palette,
        simplify=False,
        export_svg=False,
        force_silhouette_outline=False,
        silhouette_mask=silhouette,
        line_width=1,
    )
    with_sil = build_outline_page(
        image_labels,
        palette,
        simplify=False,
        export_svg=True,
        force_silhouette_outline=True,
        silhouette_mask=silhouette,
        line_width=1,
    )

    without_ink = np.asarray(without.outline).mean(axis=2) < 40
    with_ink = np.asarray(with_sil.outline).mean(axis=2) < 40
    assert without_ink.sum() < 10
    assert with_ink.sum() > 100
    # Seam pixels near the square boundary should be inked.
    assert with_ink[20, 40] or with_ink[21, 40]
    assert with_sil.outline_svg is not None
    assert 'fill="none"' in with_sil.outline_svg


def test_force_silhouette_adds_ink_vs_label_edges_only() -> None:
    """Same-colour lobe against matching background: silhouette adds the missing seam."""
    labels = np.zeros((100, 100), dtype=np.int32)
    # Teal subject body + orange lobe; orange also fills the right background.
    labels[:, :] = 1  # orange background
    labels[20:80, 15:55] = 0  # teal body
    labels[35:65, 55:85] = 1  # orange lobe (same label as background — no seam)
    palette = np.array([[30, 120, 140], [200, 80, 40]], dtype=np.uint8)
    silhouette = np.zeros((100, 100), dtype=bool)
    silhouette[20:80, 15:55] = True
    silhouette[35:65, 55:85] = True

    without = build_outline_page(
        labels,
        palette,
        simplify=False,
        export_svg=False,
        force_silhouette_outline=False,
        line_width=1,
    )
    with_sil = build_outline_page(
        labels,
        palette,
        simplify=False,
        export_svg=False,
        force_silhouette_outline=True,
        silhouette_mask=silhouette,
        line_width=1,
    )
    without_ink = np.asarray(without.outline).mean(axis=2) < 40
    with_ink = np.asarray(with_sil.outline).mean(axis=2) < 40
    # The orange-on-orange vertical seam around x=85 should only appear with silhouette.
    assert not without_ink[50, 84:86].any()
    assert with_ink[50, 84] or with_ink[50, 85]
