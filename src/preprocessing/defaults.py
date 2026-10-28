"""Ready-made preprocessing pipelines for the raw observation."""

from __future__ import annotations

from typing import Iterable

from .base import Identity
from .features import (
    CollisionFeatures,
    FlowFeatures,
    PlayerFeatures,
    TileFeatures,
)
from .pipeline import FeatureUnion, StatePipeline
from .raw import DecodeRawState


def build_raw_state_pipeline(
    output: str = "raw",
    *,
    tile_names: Iterable[str] | None = None,
    tile_one_hot: bool = False,
    flow_scale: float = 1.0,
) -> StatePipeline:
    """Build a model-facing observation pipeline.

    ``output="raw"``
        Identity on the current 342-value raw packet. This is the safe default
        and preserves the existing model input contract.

    ``output="features"``
        Decode the raw packet into named arrays and rebuild a flat feature
        vector from tile IDs/states, player state, collisions, and flow. With
        ``tile_one_hot=False`` this is still 342 values, but the assembly is
        explicit and independently testable.

    ``output="lean"``
        Reserved for the not-yet-defined lean encoder layout. It currently
        raises a clear error rather than silently inventing a model contract.
    """
    key = output.lower()
    if key == "raw":
        return StatePipeline([("identity", Identity())])
    if key == "lean":
        raise NotImplementedError(
            "the lean observation layout has not been defined; use output='raw' "
            "or output='features' and add an explicit encoder when the layout "
            "is agreed"
        )
    if key != "features":
        raise ValueError(f"unknown output {output!r}; expected raw, features, or lean")

    n_tiles = None
    if tile_names is not None:
        n_tiles = len(list(tile_names))
    return StatePipeline([
        ("decode", DecodeRawState()),
        ("features", FeatureUnion([
            (
                "tiles",
                TileFeatures(
                    one_hot_ids=tile_one_hot,
                    n_tiles=n_tiles,
                    include_states=True,
                ),
            ),
            ("player", PlayerFeatures()),
            ("collisions", CollisionFeatures()),
            ("flow", FlowFeatures(scale=flow_scale)),
        ])),
    ])
