"""Observation contracts and lean preprocessing for PLaiGROUND models.

The game emits a 188-value lean packet. The pipeline decodes it and one-hots
the categorical fields for the model.

Example:

    from plaiground.observations import build_lean_pipeline

    pre = build_lean_pipeline(tile_names=game_tile_names)
    model_x = pre.fit_transform(lean_batch)
"""

from . import spec
from .base import StateTransformer
from .defaults import build_lean_pipeline, build_lean_stick_pipeline
from .lean import (
    DecodeLeanState,
    LeanFeatures,
    LeanState,
    LeanStateBatch,
)
from .lean_stick import LeanStickPreprocessor
from .pipeline import StatePipeline

__all__ = [
    "DecodeLeanState",
    "LeanFeatures",
    "LeanState",
    "LeanStateBatch",
    "LeanStickPreprocessor",
    "StatePipeline",
    "StateTransformer",
    "build_lean_pipeline",
    "build_lean_stick_pipeline",
    "spec",
]
