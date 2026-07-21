"""Build labeled comparison spreads: original + 16-colour simplification settings."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .pipeline import DEMO_SPREAD_SETTINGS, create_colour_by_numbers
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
    """Letterbox an image into ``box`` (width, height) on white."""
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
    caption_height: int = 52,
) -> Image.Image:
    width, height = tile_size
    tile = Image.new("RGB", (width, height + caption_height), "white")
    tile.paste(_fit(image, tile_size), (0, caption_height))
    draw = ImageDraw.Draw(tile)
    draw.rectangle([0, 0, width, caption_height], fill=(245, 245, 245))
    draw.text((10, 8), title, fill="black", font=_font(18))
    draw.text((10, 30), subtitle, fill=(60, 60, 60), font=_font(13))
    return tile


def build_demo_spread(
    source: Image.Image,
    *,
    n_colours: int = 16,
    max_size: int = 700,
    settings: tuple[str, ...] = DEMO_SPREAD_SETTINGS,
    tile_width: int = 320,
) -> tuple[Image.Image, Image.Image, dict[str, dict[str, int]]]:
    """Return (colour_spread, outline_spread, stats) for one source image.

    The colour spread shows the original plus the 16-colour preview at each
    simplification setting. The outline spread shows the matching numbered pages.
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
    outline_tiles: list[Image.Image] = [
        _captioned_tile(
            source_fit,
            title="Original",
            subtitle="source photograph",
            tile_size=tile_size,
        )
    ]
    stats: dict[str, dict[str, int]] = {}

    for setting in settings:
        result = create_colour_by_numbers(
            source_fit,
            n_colours=n_colours,
            max_size=max_size,
            complexity=setting,
        )
        regions = (
            result.page.simplification.regions_after
            if result.page.simplification is not None
            else count_regions(result.page.labels)
        )
        colours = result.quantized.n_colours
        stats[setting] = {
            "regions": int(regions),
            "colours": int(colours),
            "regions_before": int(
                result.page.simplification.regions_before
                if result.page.simplification is not None
                else regions
            ),
        }
        subtitle = f"{n_colours}-colour · {regions} regions · {colours} used"
        colour_tiles.append(
            _captioned_tile(
                result.quantized.preview,
                title=setting,
                subtitle=subtitle,
                tile_size=tile_size,
            )
        )
        outline_tiles.append(
            _captioned_tile(
                result.page.outline,
                title=setting,
                subtitle=subtitle,
                tile_size=tile_size,
            )
        )

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

    return _row(colour_tiles), _row(outline_tiles), stats


def write_demo_spreads(
    sources: dict[str, Path | str],
    output_dir: Path | str,
    *,
    n_colours: int = 16,
    max_size: int = 700,
) -> dict[str, Path]:
    """Generate colour + outline spreads for each named source image."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for name, path in sources.items():
        image = Image.open(path).convert("RGB")
        colours, outlines, stats = build_demo_spread(
            image, n_colours=n_colours, max_size=max_size
        )
        colour_path = out / f"{name}_spread_colours.png"
        outline_path = out / f"{name}_spread_outlines.png"
        colours.save(colour_path)
        outlines.save(outline_path)
        written[f"{name}_colours"] = colour_path
        written[f"{name}_outlines"] = outline_path
        # Also keep the original copy next to the spreads.
        source_path = out / f"{name}_original.png"
        resize_for_processing(image, max_size=max_size).save(source_path)
        written[f"{name}_original"] = source_path
        print(f"{name}: {stats}")

    return written
