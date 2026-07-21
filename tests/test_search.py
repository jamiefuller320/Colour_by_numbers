"""Tests for search helpers with mocked HTTP."""

from __future__ import annotations

from colour_by_numbers.search import ImageHit, search_images


def test_search_openverse_mocked(monkeypatch) -> None:
    def fake_openverse(query: str, *, max_results: int):
        assert query == "dogs"
        return [
            ImageHit(
                title="Dog",
                url="https://example.com/dog.jpg",
                provider="openverse",
                license="cc0",
            )
        ]

    monkeypatch.setattr(
        "colour_by_numbers.search._search_openverse",
        fake_openverse,
    )
    monkeypatch.setattr(
        "colour_by_numbers.search._search_wikimedia",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not fall back")),
    )
    hits = search_images("dogs", max_results=3, providers=("openverse",))
    assert len(hits) == 1
    assert hits[0].provider == "openverse"


def test_search_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(
        "colour_by_numbers.search._search_openverse",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")),
    )
    monkeypatch.setattr(
        "colour_by_numbers.search._search_wikimedia",
        lambda *a, **k: [
            ImageHit(
                title="Plane",
                url="https://example.com/plane.jpg",
                provider="wikimedia",
            )
        ],
    )
    hits = search_images(
        "aircraft",
        max_results=2,
        providers=("openverse", "wikimedia"),
    )
    assert hits[0].provider == "wikimedia"
