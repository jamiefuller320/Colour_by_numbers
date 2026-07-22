"""Colour-by-numbers: search images and turn them into numbered outline pages."""

from .discover import SubjectType, discover_subject_types
from .pipeline import ColourByNumbersResult, create_colour_by_numbers

__all__ = [
    "ColourByNumbersResult",
    "SubjectType",
    "create_colour_by_numbers",
    "discover_subject_types",
]
__version__ = "0.1.0"
