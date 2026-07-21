"""Build labeled comparison spreads for colour-by-numbers demos."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .pipeline import create_colour_by_numbers
from .quantize import resize_for_processing
from .simplify import count_regions


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    width, height = box
    fitted = image.copy()
    fitted.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(
        fitted,
        ((width - fitted.width) // 2, (height - fitted.height) // 2),
    )
    return canvas


def _captioned_tile(
    image: Image.Image,
    *,
    title: str,
    subtitle: str,
    tile_size: tuple[int, int],
    caption_height: int = 56,
) -> Image.Image:
    width, height = tile_size
    tile = Image.new("RGB", (width, height + caption_height), "white")
    tile.paste(_fit(image, tile_size), (0, caption_height))
    draw = ImageDraw.Draw(tile)
    draw.rectangle([0, 0, width, caption_height], fill=(245, 245, 245))
    draw.text((10, 8), title, fill="black", font=_font(16))
    draw.text((10, 32), subtitle, fill=(60, 60, 60), font=_font(12))
    return tile


def _row(tiles: list[Image.Image]) -> Image.Image:
    gap = 12
    pad = 16
    width = pad * 2 + sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    height = pad * 2 + max(t.height for t in tiles)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    x = pad
    for tile in tiles:
        sheet.paste(tile, (x, pad))
        x += tile.width + gap
    return sheet


def build_demo_spread(
    source: Image.Image,
    *,
    n_colours: int = 32,
    max_size: int = 700,
    complexity: str = "fine",
    subject_modes: tuple[str, ...] = ("off", "dual"),
    tile_width: int = 300,
    firm_border: bool = True,
    min_a4_dpi: float | None = None,
    include_outlines: bool = False,
) -> tuple[Image.Image, Image.Image | None, dict[str, dict[str, object]]]:
    """Return (colour_spread, outline_spread|None, stats).

    Outline spreads are omitted by default while colour plates are refined.
    ``min_a4_dpi`` defaults to None for demos so existing sources still render;
    production UI/CLI keep the A4 filter on.
    """
    source_fit = resize_for_processing(source.convert("RGB"), max_size=max_size)
    aspect = source_fit.height / max(source_fit.width, 1)
    tile_height = max(180, int(tile_width * aspect))
    tile_size = (tile_width, tile_height)

    colour_tiles: list[Image.Image] = [
        _captioned_tile(
            source_fit,
            title="Original",
            subtitle="source photograph",
            tile_size=tile_size,
        )
    ]
    outline_tiles: list[Image.Image] = []
    if include_outlines:
        outline_tiles.append(
            _captioned_tile(
                source_fit,
                title="Original",
                subtitle="source photograph",
                tile_size=tile_size,
            )
        )
    stats: dict[str, dict[str, object]] = {}

    labels = {
        "off": "fine (full frame)",
        "isolate": "fine + flat isolate",
        "dual": "dual fine/light · 80% fill",
    }

    for mode in subject_modes:
        result = create_colour_by_numbers(
            source,
            n_colours=n_colours,
            max_size=max_size,
            complexity=complexity,
            subject_mode=mode,
            subject_fill=0.80,
            subject_complexity="fine",
            background_complexity="light",
            firm_border=firm_border,
            min_a4_dpi=min_a4_dpi,
        )
        regions = (
            result.page.simplification.regions_after
            if result.page.simplification is not None
            else count_regions(result.page.labels)
        )
        colours = result.quantized.n_colours
        fg = (
            round(100 * result.subject_mask.foreground_fraction, 1)
            if result.subject_mask is not None
            else None
        )
        stats[mode] = {
            "regions": int(regions),
            "colours": int(colours),
            "foreground_pct": fg,
            "complexity": result.complexity,
            "mode": mode,
            "print_dpi": result.print_dpi,
            "firm_border": result.firm_border,
        }
        subtitle = f"{colours}-colour · {regions} regions"
        if result.print_dpi is not None:
            subtitle += f" · ~{result.print_dpi:.0f} DPI"
        if fg is not None:
            subtitle += f" · fg {fg}%"
        if result.prepared is not None and mode == "dual":
            colour_tiles.append(
                _captioned_tile(
                    result.prepared,
                    title="80% fill crop",
                    subtitle="subject-centred frame",
                    tile_size=tile_size,
                )
            )
        colour_tiles.append(
            _captioned_tile(
                result.quantized.preview,
                title=labels.get(mode, mode),
                subtitle=subtitle,
                tile_size=tile_size,
            )
        )
        if include_outlines:
            outline_tiles.append(
                _captioned_tile(
                    result.page.outline,
                    title=labels.get(mode, mode),
                    subtitle=subtitle,
                    tile_size=tile_size,
                )
            )

    outlines = _row(outline_tiles) if include_outlines and outline_tiles else None
    return _row(colour_tiles), outlines, stats


def write_demo_spreads(
    sources: dict[str, Path | str],
    output_dir: Path | str,
    *,
    n_colours: int = 32,
    max_size: int = 700,
    complexity: str = "fine",
    include_outlines: bool = False,
    firm_border: bool = True,
    min_a4_dpi: float | None = None,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for name, path in sources.items():
        image = Image.open(path).convert("RGB")
        colours, outlines, stats = build_demo_spread(
            image,
            n_colours=n_colours,
            max_size=max_size,
            complexity=complexity,
            subject_modes=("off", "dual"),
            include_outlines=include_outlines,
            firm_border=firm_border,
            min_a4_dpi=min_a4_dpi,
        )
        colour_path = out / f"{name}_spread_colours.png"
        colours.save(colour_path)
        written[f"{name}_colours"] = colour_path
        if outlines is not None:
            outline_path = out / f"{name}_spread_outlines.png"
            outlines.save(outline_path)
            written[f"{name}_outlines"] = outline_path
        source_path = out / f"{name}_original.png"
        resize_for_processing(image, max_size=max_size).save(source_path)
        written[f"{name}_original"] = source_path
        print(f"{name}: {stats}")

    return written
