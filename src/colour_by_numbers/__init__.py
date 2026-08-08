"""Colour-by-numbers: search images and turn them into numbered outline pages."""

from .colourways import list_colourways, remap_palette, render_colourway_plate
from .discover import SubjectType, discover_subject_types
from .feedback import FeedbackLoopResult, run_subject_feedback_loop
from .generate import GeneratedPage, generate_colouring_page
from .illustrate import IllustrationResult, generate_illustration
from .library import (
    AssetLibrary,
    PairRecord,
    SetRecord,
    ingest_generated_set,
    seed_sample_sets,
)
from .pipeline import ColourByNumbersResult, create_colour_by_numbers
from .quality import PlateQualityReport, evaluate_plate_quality
from .set_generate import GeneratedSet, generate_colouring_set
from .set_plan import SetPlan, plan_colouring_set, plan_mixed_colouring_set

__all__ = [
    "AssetLibrary",
    "ColourByNumbersResult",
    "FeedbackLoopResult",
    "GeneratedPage",
    "GeneratedSet",
    "IllustrationResult",
    "PairRecord",
    "PlateQualityReport",
    "SetPlan",
    "SetRecord",
    "SubjectType",
    "create_colour_by_numbers",
    "discover_subject_types",
    "evaluate_plate_quality",
    "generate_colouring_page",
    "generate_colouring_set",
    "generate_illustration",
    "ingest_generated_set",
    "list_colourways",
    "plan_colouring_set",
    "plan_mixed_colouring_set",
    "remap_palette",
    "render_colourway_plate",
    "run_subject_feedback_loop",
    "seed_sample_sets",
]
__version__ = "0.1.0"
