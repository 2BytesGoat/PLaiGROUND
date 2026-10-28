"""Ready-made lean observation pipeline."""

from __future__ import annotations

from typing import Iterable

from .lean import DecodeLeanState, LeanFeatures
from .pipeline import StatePipeline


def build_lean_pipeline(
    *,
    tile_names: Iterable[str] | None = None,
    flow_scale: float = 1.0,
) -> StatePipeline:
    """Build the one-hot model pipeline from the game's 188-value lean packet."""
    return StatePipeline([
        ("decode", DecodeLeanState()),
        (
            "features",
            LeanFeatures(
                tile_names=tile_names,
                flow_scale=flow_scale,
            ),
        ),
    ])
