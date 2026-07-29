"""Colour-by-numbers: search images and turn them into numbered outline pages."""

from .discover import SubjectType, discover_subject_types
from .generate import GeneratedPage, generate_colouring_page
from .illustrate import IllustrationResult, generate_illustration
from .pipeline import ColourByNumbersResult, create_colour_by_numbers
from .quality import PlateQualityReport, evaluate_plate_quality

__all__ = [
    "ColourByNumbersResult",
    "GeneratedPage",
    "IllustrationResult",
    "PlateQualityReport",
    "SubjectType",
    "create_colour_by_numbers",
    "discover_subject_types",
    "evaluate_plate_quality",
    "generate_colouring_page",
    "generate_illustration",
]
__version__ = "0.1.0"
