"""Colour-by-numbers: search images and turn them into numbered outline pages."""

from .discover import SubjectType, discover_subject_types
from .generate import GeneratedPage, generate_colouring_page
from .illustrate import IllustrationResult, generate_illustration
from .pipeline import ColourByNumbersResult, create_colour_by_numbers

__all__ = [
    "ColourByNumbersResult",
    "GeneratedPage",
    "IllustrationResult",
    "SubjectType",
    "create_colour_by_numbers",
    "discover_subject_types",
    "generate_colouring_page",
    "generate_illustration",
]
__version__ = "0.1.0"
