"""Web image search and download helpers."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

import requests
from PIL import Image

from .print_resolution import evaluate_print_resolution

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "ColourByNumbers/0.1 (+https://github.com/colour-by-numbers; educational use)"
)


@dataclass(frozen=True)
class ImageHit:
    """A single search result that can be downloaded."""

    title: str
    url: str
    thumbnail: str | None = None
    source: str | None = None
    provider: str | None = None
    license: str | None = None
    width: int | None = None
    height: int | None = None


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    return session


def hit_a4_dpi(hit: ImageHit) -> float | None:
    """Effective A4 DPI from hit metadata, or None if size unknown."""
    if not hit.width or not hit.height or hit.width <= 0 or hit.height <= 0:
        return None
    return evaluate_print_resolution(hit.width, hit.height, min_dpi=1.0).effective_dpi


def hit_meets_min_a4_dpi(hit: ImageHit, min_dpi: float) -> bool | None:
    """True/False when metadata is known; None when size is unknown."""
    dpi = hit_a4_dpi(hit)
    if dpi is None:
        return None
    return dpi + 1e-6 >= min_dpi


def filter_hits_for_a4(
    hits: list[ImageHit],
    *,
    min_dpi: float,
    prefer_known: bool = True,
) -> list[ImageHit]:
    """Prefer (or require) hits that meet an A4 print-resolution floor.

    Hits with known dimensions below ``min_dpi`` are dropped. Hits with
    unknown size are kept at the end so providers without metadata still work.
    """
    if min_dpi <= 0:
        return list(hits)

    adequate: list[ImageHit] = []
    unknown: list[ImageHit] = []
    for hit in hits:
        verdict = hit_meets_min_a4_dpi(hit, min_dpi)
        if verdict is True:
            adequate.append(hit)
        elif verdict is None:
            unknown.append(hit)
        else:
            logger.info(
                "Skipping low-res hit %sx%s (~%.0f DPI A4): %s",
                hit.width,
                hit.height,
                hit_a4_dpi(hit) or 0.0,
                hit.url,
            )

    if adequate:
        # Largest first among known-adequate results.
        adequate.sort(
            key=lambda h: (h.width or 0) * (h.height or 0),
            reverse=True,
        )
        return adequate + (unknown if not prefer_known else unknown)
    return unknown


def _search_openverse(query: str, *, max_results: int) -> list[ImageHit]:
    """Search Openverse for openly licensed images."""
    session = _session()
    response = session.get(
        "https://api.openverse.org/v1/images/",
        params={
            "q": query,
            "page_size": min(max(max_results * 3, 20), 40),
            "license_type": "commercial,modification",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    hits: list[ImageHit] = []
    for item in payload.get("results") or []:
        image_url = item.get("url")
        if not image_url:
            continue
        license_info = item.get("license") or ""
        if item.get("license_version"):
            license_info = f"{license_info} {item['license_version']}".strip()
        width = item.get("width")
        height = item.get("height")
        hits.append(
            ImageHit(
                title=item.get("title") or query,
                url=image_url,
                thumbnail=item.get("thumbnail"),
                source=item.get("foreign_landing_url") or item.get("source"),
                provider="openverse",
                license=license_info or None,
                width=int(width) if width else None,
                height=int(height) if height else None,
            )
        )
    return hits


def _search_wikimedia(query: str, *, max_results: int) -> list[ImageHit]:
    """Search Wikimedia Commons file titles/descriptions."""
    session = _session()
    response = session.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": min(max(max_results * 3, 20), 40),
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
            "format": "json",
        },
        timeout=30,
    )
    response.raise_for_status()
    pages = (response.json().get("query") or {}).get("pages") or {}
    ordered = sorted(pages.values(), key=lambda page: page.get("index", 0))
    hits: list[ImageHit] = []
    for page in ordered:
        info_list = page.get("imageinfo") or []
        if not info_list:
            continue
        info = info_list[0]
        mime = (info.get("mime") or "").lower()
        if mime and not mime.startswith("image/"):
            continue
        if mime in {"image/svg+xml", "image/tiff"}:
            continue
        image_url = info.get("url")
        if not image_url:
            continue
        title = page.get("title") or query
        if title.startswith("File:"):
            title = title[5:]
        width = info.get("width")
        height = info.get("height")
        hits.append(
            ImageHit(
                title=title,
                url=image_url,
                thumbnail=info.get("thumburl"),
                source="https://commons.wikimedia.org/",
                provider="wikimedia",
                license="Wikimedia Commons (check file page)",
                width=int(width) if width else None,
                height=int(height) if height else None,
            )
        )
    return hits


def _search_duckduckgo(query: str, *, max_results: int, safesearch: str) -> list[ImageHit]:
    """Optional DuckDuckGo fallback (may rate-limit)."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install ddgs for DuckDuckGo search fallback.") from exc

    hits: list[ImageHit] = []
    with DDGS() as ddgs:
        results = ddgs.images(
            query,
            safesearch=safesearch,
            max_results=max(max_results * 3, max_results),
        )
        for item in results or []:
            image_url = item.get("image") or item.get("url")
            if not image_url:
                continue
            width = item.get("width")
            height = item.get("height")
            try:
                width_i = int(width) if width else None
            except (TypeError, ValueError):
                width_i = None
            try:
                height_i = int(height) if height else None
            except (TypeError, ValueError):
                height_i = None
            hits.append(
                ImageHit(
                    title=item.get("title") or query,
                    url=image_url,
                    thumbnail=item.get("thumbnail"),
                    source=item.get("source"),
                    provider="duckduckgo",
                    license=None,
                    width=width_i,
                    height=height_i,
                )
            )
    return hits


def search_images(
    query: str,
    *,
    max_results: int = 8,
    safesearch: str = "moderate",
    providers: Iterable[str] | None = None,
    min_a4_dpi: float | None = None,
) -> list[ImageHit]:
    """Search the web for images matching a descriptive query.

    Tries Openverse first (open licences), then Wikimedia Commons, then
    DuckDuckGo if needed. No API keys are required.

    When ``min_a4_dpi`` is set, results known to be below that print
    resolution are filtered out and larger images are preferred.
    """
    query = query.strip()
    if not query:
        raise ValueError("Search query must not be empty.")

    order = list(providers or ("openverse", "wikimedia", "duckduckgo"))
    errors: list[str] = []
    fetch_n = max_results * 3 if min_a4_dpi else max_results

    for provider in order:
        try:
            if provider == "openverse":
                hits = _search_openverse(query, max_results=fetch_n)
            elif provider == "wikimedia":
                hits = _search_wikimedia(query, max_results=fetch_n)
            elif provider == "duckduckgo":
                hits = _search_duckduckgo(
                    query, max_results=fetch_n, safesearch=safesearch
                )
            else:
                raise ValueError(f"Unknown search provider: {provider}")
            if hits:
                if min_a4_dpi is not None and min_a4_dpi > 0:
                    hits = filter_hits_for_a4(hits, min_dpi=min_a4_dpi)
                if hits:
                    logger.info(
                        "Search provider %s returned %d hits", provider, len(hits)
                    )
                    return hits[:max_results]
                errors.append(f"{provider}: no results meeting A4 DPI filter")
                continue
            errors.append(f"{provider}: no results")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Search provider %s failed: %s", provider, exc)
            errors.append(f"{provider}: {exc}")

    raise RuntimeError(
        f"No images found for query {query!r}. Attempts:\n" + "\n".join(errors)
    )


def download_image(
    url: str,
    *,
    timeout: float = 30.0,
    max_bytes: int = 15_000_000,
) -> Image.Image:
    """Download an image URL and return a Pillow RGB image."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")

    session = _session()
    with session.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "").lower()
        if content_type and not content_type.startswith("image/"):
            logger.debug("Unexpected content type %s for %s", content_type, url)

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Image exceeds size limit of {max_bytes} bytes.")
            chunks.append(chunk)

    data = b"".join(chunks)
    image = Image.open(io.BytesIO(data))
    image.load()
    return image.convert("RGB")


def load_local_image(path: str) -> Image.Image:
    """Load an image from disk as RGB."""
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def search_and_download(
    query: str,
    *,
    max_results: int = 8,
    pick: int = 0,
    min_a4_dpi: float | None = None,
) -> tuple[Image.Image, ImageHit]:
    """Search for images and download one result (default: first).

    When ``min_a4_dpi`` is set, prefer hits that meet the A4 floor and skip
    downloads whose decoded pixel size falls short.
    """
    hits = search_images(
        query, max_results=max_results, min_a4_dpi=min_a4_dpi
    )
    if not hits:
        raise RuntimeError(f"No images found for query: {query!r}")
    if pick < 0 or pick >= len(hits):
        raise IndexError(f"pick={pick} out of range for {len(hits)} results")

    errors: list[str] = []
    order: Iterable[int] = (pick, *range(len(hits)))
    tried: set[int] = set()
    for index in order:
        if index in tried:
            continue
        tried.add(index)
        hit = hits[index]
        try:
            image = download_image(hit.url)
            if min_a4_dpi is not None and min_a4_dpi > 0:
                report = evaluate_print_resolution(
                    image.width, image.height, min_dpi=min_a4_dpi
                )
                if not report.adequate:
                    errors.append(
                        f"{hit.url}: downloaded {image.width}×{image.height} "
                        f"(~{report.effective_dpi:.0f} DPI A4) below {min_a4_dpi:.0f}"
                    )
                    logger.warning(
                        "Downloaded image below A4 DPI filter: %s (~%.0f DPI)",
                        hit.url,
                        report.effective_dpi,
                    )
                    continue
            return image, hit
        except Exception as exc:  # noqa: BLE001 - collect and try next
            errors.append(f"{hit.url}: {exc}")
            logger.warning("Failed to download %s: %s", hit.url, exc)

    raise RuntimeError(
        "Could not download any search result. Errors:\n" + "\n".join(errors)
    )
