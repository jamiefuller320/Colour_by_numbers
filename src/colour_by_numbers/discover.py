"""Discover specific subject types from broad category queries.

A request like ``dogs`` should not jump straight to a generic dog photo.
Instead we shortlist concrete types (e.g. pug, golden retriever), then run
image search / colour-by-numbers against the chosen type for recognisable
specificity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .search import ImageHit, search_images

# Broad category → concrete types used for colouring-page specificity.
# Keep lists short and visually distinct; order is a soft prior.
CATEGORY_TYPES: dict[str, tuple[str, ...]] = {
    "dogs": (
        "golden retriever",
        "labrador retriever",
        "pug",
        "french bulldog",
        "german shepherd",
        "border collie",
        "beagle",
        "dachshund",
        "corgi",
        "husky",
        "poodle",
        "boxer",
        "dalmatian",
        "rottweiler",
        "chihuahua",
        "yorkshire terrier",
    ),
    "cats": (
        "tabby cat",
        "siamese cat",
        "persian cat",
        "maine coon",
        "british shorthair",
        "bengal cat",
        "ragdoll cat",
        "scottish fold",
    ),
    "aircraft": (
        "biplane",
        "fighter jet",
        "airliner",
        "helicopter",
        "seaplane",
        "glider",
        "spitfire",
        "cessna",
        "concorde",
        "hot air balloon",
    ),
    "birds": (
        "robin",
        "eagle",
        "owl",
        "parrot",
        "swan",
        "flamingo",
        "penguin",
        "hummingbird",
        "peacock",
        "toucan",
    ),
    "horses": (
        "thoroughbred horse",
        "pony",
        "foal",
        "clydesdale",
        "arabian horse",
    ),
    "flowers": (
        "sunflower",
        "rose",
        "tulip",
        "daisy",
        "poppy",
        "orchid",
        "lavender",
        "lily",
    ),
    "cars": (
        "vintage car",
        "sports car",
        "volkswagen beetle",
        "pickup truck",
        "formula one car",
    ),
    "boats": (
        "sailboat",
        "yacht",
        "rowboat",
        "canoe",
        "fishing boat",
        "tall ship",
    ),
}

# Singular / plural / synonym → canonical category key.
CATEGORY_ALIASES: dict[str, str] = {
    "dog": "dogs",
    "dogs": "dogs",
    "puppy": "dogs",
    "puppies": "dogs",
    "canine": "dogs",
    "cat": "cats",
    "cats": "cats",
    "kitten": "cats",
    "kittens": "cats",
    "feline": "cats",
    "aircraft": "aircraft",
    "airplane": "aircraft",
    "aeroplane": "aircraft",
    "plane": "aircraft",
    "planes": "aircraft",
    "jet": "aircraft",
    "bird": "birds",
    "birds": "birds",
    "horse": "horses",
    "horses": "horses",
    "pony": "horses",
    "flower": "flowers",
    "flowers": "flowers",
    "car": "cars",
    "cars": "cars",
    "automobile": "cars",
    "boat": "boats",
    "boats": "boats",
    "ship": "boats",
    "sailboat": "boats",
}


@dataclass(frozen=True)
class SubjectType:
    """A concrete subject type chosen for one colouring-page run."""

    label: str
    category: str
    search_query: str
    score: float = 0.0
    evidence: tuple[str, ...] = ()
    already_specific: bool = False


@dataclass
class TypeDiscoveryResult:
    """Outcome of type discovery for a user query."""

    original_query: str
    category: str | None
    types: list[SubjectType] = field(default_factory=list)
    skipped: bool = False
    reason: str = ""

    @property
    def needs_choice(self) -> bool:
        return not self.skipped and len(self.types) > 1


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def resolve_category(query: str) -> str | None:
    """Return canonical category key if ``query`` names a broad category."""
    q = _norm(query)
    if q in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[q]
    # Single leading token: "dogs photos" → dogs
    first = q.split(" ")[0]
    if first in CATEGORY_ALIASES and len(q.split()) <= 2:
        # Avoid treating "golden retriever" as category via first token.
        if first in {"dog", "dogs", "cat", "cats", "bird", "birds", "horse", "horses",
                     "flower", "flowers", "car", "cars", "boat", "boats",
                     "aircraft", "airplane", "aeroplane", "plane", "planes",
                     "jet", "puppy", "puppies", "kitten", "kittens"}:
            return CATEGORY_ALIASES[first]
    return None


def find_matching_type(query: str) -> tuple[str, str] | None:
    """If query already names a known type, return ``(category, label)``."""
    q = _norm(query)
    candidates: list[tuple[int, str, str]] = []
    for category, types in CATEGORY_TYPES.items():
        for label in types:
            if label == q or label in q:
                candidates.append((len(label), category, label))
                continue
            # All label tokens present as whole words in the query.
            label_tokens = label.split()
            query_tokens = set(q.replace("-", " ").split())
            if label_tokens and all(tok in query_tokens for tok in label_tokens):
                candidates.append((len(label), category, label))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, category, label = candidates[0]
    return category, label


def is_broad_category_query(query: str) -> bool:
    """True when the query is a broad category needing type discovery."""
    category = resolve_category(query)
    if category is None:
        return False
    # "golden retriever" resolves no category; "dogs" does and is not a type label.
    matched = find_matching_type(query)
    if matched is not None and _norm(matched[1]) == _norm(query):
        return False
    if matched is not None and category is None:
        return False
    # Broad if the query is essentially just the category word(s).
    q = _norm(query)
    alias_forms = {k for k, v in CATEGORY_ALIASES.items() if v == category}
    if q in alias_forms:
        return True
    tokens = q.split()
    if len(tokens) <= 2 and tokens[0] in alias_forms and matched is None:
        return True
    return matched is None and q.split()[0] in alias_forms


def build_type_search_query(label: str, *, category: str | None = None) -> str:
    """Build a specific, colouring-friendly search string for one type."""
    label = label.strip()
    if category == "aircraft":
        return f"{label} clear sky"
    if category == "flowers":
        return f"{label} close up"
    if category in {"cars", "boats"}:
        return f"{label} side view"
    return f"{label} portrait"


def _title_match_score(label: str, titles: list[str]) -> tuple[float, list[str]]:
    """Score how often ``label`` appears in hit titles."""
    needle = _norm(label)
    tokens = [t for t in needle.split() if len(t) > 2]
    if not tokens:
        return 0.0, []
    evidence: list[str] = []
    hits = 0.0
    for title in titles:
        nt = _norm(title)
        if needle in nt:
            hits += 1.0
            if len(evidence) < 3:
                evidence.append(title.strip()[:80])
        elif all(tok in nt for tok in tokens):
            hits += 0.75
            if len(evidence) < 3:
                evidence.append(title.strip()[:80])
        elif any(tok in nt for tok in tokens):
            hits += 0.2
    return hits, evidence


def discover_subject_types(
    query: str,
    *,
    max_types: int = 8,
    probe_search: bool = True,
    max_probe_results: int = 24,
) -> TypeDiscoveryResult:
    """Shortlist concrete subject types for a broad (or already-specific) query.

    - Already-specific queries (e.g. ``pug``) return a single type.
    - Broad categories (e.g. ``dogs``) return curated types ranked by how often
      they appear in a category image search.
    - Unknown free-text queries skip discovery and keep the original string.
    """
    original = query.strip()
    if not original:
        raise ValueError("Query must not be empty.")

    if is_broad_category_query(original):
        category = resolve_category(original)
        assert category is not None
        catalog = list(CATEGORY_TYPES[category])
        titles: list[str] = []
        if probe_search:
            try:
                hits = search_images(
                    category,
                    max_results=max_probe_results,
                    contrast_bias=False,
                )
                titles = [h.title for h in hits if h.title]
            except Exception:  # noqa: BLE001 - discovery should degrade gracefully
                titles = []

        scored: list[SubjectType] = []
        for index, label in enumerate(catalog):
            mention_score, evidence = _title_match_score(label, titles)
            prior = max(0.0, (len(catalog) - index) / max(len(catalog), 1)) * 0.15
            score = mention_score + prior
            scored.append(
                SubjectType(
                    label=label,
                    category=category,
                    search_query=build_type_search_query(label, category=category),
                    score=score,
                    evidence=tuple(evidence),
                    already_specific=False,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        top = scored[: max(1, max_types)]
        return TypeDiscoveryResult(
            original_query=original,
            category=category,
            types=top,
            skipped=False,
            reason=f"ranked {len(catalog)} curated types for category {category!r}",
        )

    specific = find_matching_type(original)
    if specific is not None:
        category, label = specific
        st = SubjectType(
            label=label,
            category=category,
            search_query=build_type_search_query(label, category=category),
            score=1.0,
            evidence=(original,),
            already_specific=True,
        )
        return TypeDiscoveryResult(
            original_query=original,
            category=category,
            types=[st],
            skipped=False,
            reason="query already names a specific type",
        )

    return TypeDiscoveryResult(
        original_query=original,
        category=None,
        types=[
            SubjectType(
                label=original,
                category="custom",
                search_query=original,
                score=1.0,
                already_specific=True,
            )
        ],
        skipped=True,
        reason="not a known broad category; using query as-is",
    )


def pick_subject_type(
    discovery: TypeDiscoveryResult,
    *,
    type_name: str | None = None,
    pick: int = 0,
) -> SubjectType:
    """Select one type from a discovery result by name or index."""
    if not discovery.types:
        raise RuntimeError(f"No subject types found for {discovery.original_query!r}")

    if type_name:
        want = _norm(type_name)
        for item in discovery.types:
            if _norm(item.label) == want or want in _norm(item.label):
                return item
        # Allow choosing a catalog type even if it fell outside the top-N.
        if discovery.category and discovery.category in CATEGORY_TYPES:
            for label in CATEGORY_TYPES[discovery.category]:
                if _norm(label) == want or want in _norm(label):
                    return SubjectType(
                        label=label,
                        category=discovery.category,
                        search_query=build_type_search_query(
                            label, category=discovery.category
                        ),
                        score=0.0,
                        already_specific=True,
                    )
        raise ValueError(
            f"Unknown type {type_name!r} for query {discovery.original_query!r}. "
            f"Try one of: {', '.join(t.label for t in discovery.types)}"
        )

    if pick < 0 or pick >= len(discovery.types):
        raise IndexError(f"pick={pick} out of range for {len(discovery.types)} types")
    return discovery.types[pick]


def search_images_for_type(
    subject_type: SubjectType,
    *,
    max_results: int = 8,
    min_a4_dpi: float | None = None,
    contrast_bias: bool = True,
) -> list[ImageHit]:
    """Search images using a discovered type's refined query."""
    return search_images(
        subject_type.search_query,
        max_results=max_results,
        min_a4_dpi=min_a4_dpi,
        contrast_bias=contrast_bias,
    )
