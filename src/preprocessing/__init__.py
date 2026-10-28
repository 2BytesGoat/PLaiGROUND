"""Observation preprocessing for PLaiGROUND models.

The package is deliberately independent of the socket and training drivers.
A run should build one pipeline, fit it on the training observations, save it
with the model, and reuse the exact same pipeline for evaluation and play.

Example:

    from preprocessing import build_raw_state_pipeline

    pre = build_raw_state_pipeline("raw")
    pre.fit(raw_batch)
    model_input = pre.transform(raw_observation)
"""

from .base import Clip, Identity, Standardize, StateTransformer
from .defaults import build_raw_state_pipeline
from .features import (
    CollisionFeatures,
    FlowFeatures,
    PlayerFeatures,
    TileFeatures,
)
from .pipeline import FeatureUnion, Pipeline, StatePipeline
from .raw import DecodeRawState, RawState, RawStateBatch

__all__ = [
    "Clip",
    "CollisionFeatures",
    "DecodeRawState",
    "FeatureUnion",
    "FlowFeatures",
    "Identity",
    "Pipeline",
    "PlayerFeatures",
    "RawState",
    "RawStateBatch",
    "Standardize",
    "StatePipeline",
    "StateTransformer",
    "TileFeatures",
    "build_raw_state_pipeline",
]
