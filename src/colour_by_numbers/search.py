"""Web image search and download helpers."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

import requests
from PIL import Image

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


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    return session


def _search_openverse(query: str, *, max_results: int) -> list[ImageHit]:
    """Search Openverse for openly licensed images."""
    session = _session()
    response = session.get(
        "https://api.openverse.org/v1/images/",
        params={
            "q": query,
            "page_size": min(max_results, 20),
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
        hits.append(
            ImageHit(
                title=item.get("title") or query,
                url=image_url,
                thumbnail=item.get("thumbnail"),
                source=item.get("foreign_landing_url") or item.get("source"),
                provider="openverse",
                license=license_info or None,
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
            "gsrlimit": min(max_results, 20),
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
        hits.append(
            ImageHit(
                title=title,
                url=image_url,
                thumbnail=info.get("thumburl"),
                source="https://commons.wikimedia.org/",
                provider="wikimedia",
                license="Wikimedia Commons (check file page)",
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
            max_results=max_results,
        )
        for item in results or []:
            image_url = item.get("image") or item.get("url")
            if not image_url:
                continue
            hits.append(
                ImageHit(
                    title=item.get("title") or query,
                    url=image_url,
                    thumbnail=item.get("thumbnail"),
                    source=item.get("source"),
                    provider="duckduckgo",
                    license=None,
                )
            )
    return hits


def search_images(
    query: str,
    *,
    max_results: int = 8,
    safesearch: str = "moderate",
    providers: Iterable[str] | None = None,
) -> list[ImageHit]:
    """Search the web for images matching a descriptive query.

    Tries Openverse first (open licences), then Wikimedia Commons, then
    DuckDuckGo if needed. No API keys are required.
    """
    query = query.strip()
    if not query:
        raise ValueError("Search query must not be empty.")

    order = list(providers or ("openverse", "wikimedia", "duckduckgo"))
    errors: list[str] = []

    for provider in order:
        try:
            if provider == "openverse":
                hits = _search_openverse(query, max_results=max_results)
            elif provider == "wikimedia":
                hits = _search_wikimedia(query, max_results=max_results)
            elif provider == "duckduckgo":
                hits = _search_duckduckgo(
                    query, max_results=max_results, safesearch=safesearch
                )
            else:
                raise ValueError(f"Unknown search provider: {provider}")
            if hits:
                logger.info("Search provider %s returned %d hits", provider, len(hits))
                return hits[:max_results]
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
) -> tuple[Image.Image, ImageHit]:
    """Search for images and download one result (default: first)."""
    hits = search_images(query, max_results=max_results)
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
            return download_image(hit.url), hit
        except Exception as exc:  # noqa: BLE001 - collect and try next
            errors.append(f"{hit.url}: {exc}")
            logger.warning("Failed to download %s: %s", hit.url, exc)

    raise RuntimeError(
        "Could not download any search result. Errors:\n" + "\n".join(errors)
    )
