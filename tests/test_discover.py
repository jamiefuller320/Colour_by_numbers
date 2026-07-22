"""Tests for subject-type discovery."""

from __future__ import annotations

from colour_by_numbers.discover import (
    build_type_search_query,
    discover_subject_types,
    find_matching_type,
    is_broad_category_query,
    pick_subject_type,
    resolve_category,
)
from colour_by_numbers.search import ImageHit


def test_resolve_category_aliases() -> None:
    assert resolve_category("dogs") == "dogs"
    assert resolve_category("Dog") == "dogs"
    assert resolve_category("aircraft") == "aircraft"
    assert resolve_category("golden retriever") is None


def test_broad_vs_specific_queries() -> None:
    assert is_broad_category_query("dogs")
    assert is_broad_category_query("dog photos")
    assert not is_broad_category_query("pug")
    assert not is_broad_category_query("golden retriever")
    assert not is_broad_category_query("spitfire")


def test_find_matching_type_prefers_longest_label() -> None:
    assert find_matching_type("golden retriever") == ("dogs", "golden retriever")
    assert find_matching_type("pug") == ("dogs", "pug")


def test_discover_broad_dogs_without_network(monkeypatch) -> None:
    def fake_search(query: str, **kwargs):
        assert query == "dogs"
        return [
            ImageHit("Golden Retriever in park", "https://ex/a.jpg"),
            ImageHit("Cute pug portrait", "https://ex/b.jpg"),
            ImageHit("Pug puppy", "https://ex/c.jpg"),
            ImageHit("Random dog", "https://ex/d.jpg"),
        ]

    monkeypatch.setattr("colour_by_numbers.discover.search_images", fake_search)
    discovery = discover_subject_types("dogs", max_types=5, probe_search=True)
    assert discovery.category == "dogs"
    assert discovery.needs_choice
    assert discovery.types[0].label in {"pug", "golden retriever"}
    # Pug mentioned twice should rank highly.
    labels = [t.label for t in discovery.types]
    assert "pug" in labels
    assert build_type_search_query("pug", category="dogs") == "pug portrait"


def test_discover_specific_skips_shortlist() -> None:
    discovery = discover_subject_types("pug", probe_search=False)
    assert len(discovery.types) == 1
    assert discovery.types[0].label == "pug"
    assert discovery.types[0].already_specific
    assert not discovery.needs_choice


def test_pick_subject_type_by_name() -> None:
    discovery = discover_subject_types("dogs", probe_search=False, max_types=8)
    chosen = pick_subject_type(discovery, type_name="border collie")
    assert chosen.label == "border collie"
    assert "border collie" in chosen.search_query


def test_custom_query_skipped() -> None:
    discovery = discover_subject_types("red vintage tractor", probe_search=False)
    assert discovery.skipped
    assert discovery.types[0].search_query == "red vintage tractor"
