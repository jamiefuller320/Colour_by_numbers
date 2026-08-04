"""Colour-by-numbers: search images and turn them into numbered outline pages."""

from .discover import SubjectType, discover_subject_types
from .feedback import FeedbackLoopResult, run_subject_feedback_loop
from .generate import GeneratedPage, generate_colouring_page
from .illustrate import IllustrationResult, generate_illustration
from .pipeline import ColourByNumbersResult, create_colour_by_numbers
from .quality import PlateQualityReport, evaluate_plate_quality
from .set_generate import GeneratedSet, generate_colouring_set
from .set_plan import SetPlan, plan_colouring_set

__all__ = [
    "ColourByNumbersResult",
    "FeedbackLoopResult",
    "GeneratedPage",
    "GeneratedSet",
    "IllustrationResult",
    "PlateQualityReport",
    "SetPlan",
    "SubjectType",
    "create_colour_by_numbers",
    "discover_subject_types",
    "evaluate_plate_quality",
    "generate_colouring_page",
    "generate_colouring_set",
    "generate_illustration",
    "plan_colouring_set",
    "run_subject_feedback_loop",
]
__version__ = "0.1.0"
