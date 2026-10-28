"""Feature branches over a decoded raw state."""

from __future__ import annotations

from typing import Any

import numpy as np

import obs_spec
from .base import StateTransformer
from .raw import RawStateBatch


class TileFeatures(StateTransformer):
    """Flatten the local tile window and its dynamic cell states.

    ``one_hot_ids=False`` keeps the 7x7 tile IDs and the five dynamic state
    values as a 294-value vector. With one-hot IDs, configure ``n_tiles`` (from
    the game's ``tile_names``) or let ``fit`` infer it from the training data.
    """

    def __init__(
        self,
        one_hot_ids: bool = False,
        n_tiles: int | None = None,
        include_states: bool = True,
    ) -> None:
        self.one_hot_ids = bool(one_hot_ids)
        self.n_tiles = None if n_tiles is None else int(n_tiles)
        self.include_states = bool(include_states)

    def fit(self, X: RawStateBatch, y: Any = None) -> "TileFeatures":
        self._validate(X)
        if self.one_hot_ids:
            if self.n_tiles is None:
                self.n_tiles_ = int(np.max(X.tile_ids)) + 1
            else:
                self.n_tiles_ = int(self.n_tiles)
            if self.n_tiles_ <= 0:
                raise ValueError("n_tiles must be positive")
        else:
            self.n_tiles_ = 1
        self.n_features_out_ = (
            obs_spec.GRID_CELLS
            * (self.n_tiles_ if self.one_hot_ids else 1)
            + (
                obs_spec.GRID_CELLS * obs_spec.RAW_TILE_STATE_FLOATS
                if self.include_states
                else 0
            )
        )
        return self

    def transform(self, X: RawStateBatch) -> np.ndarray:
        self._validate(X)
        count = len(X)
        if self.one_hot_ids:
            n_tiles = int(self.n_tiles_)
            ids = X.tile_ids.astype(np.int64)
            if np.any(ids < 0) or np.any(ids >= n_tiles):
                raise ValueError(
                    f"tile id outside [0, {n_tiles}); got "
                    f"min={int(ids.min())}, max={int(ids.max())}"
                )
            tile_part = np.eye(n_tiles, dtype=np.float32)[ids].reshape(count, -1)
        else:
            tile_part = X.tile_ids.astype(np.float32).reshape(count, -1)
        if not self.include_states:
            return tile_part
        states = X.tile_states.astype(np.float32).reshape(count, -1)
        return np.concatenate([tile_part, states], axis=1)

    @property
    def output_size(self) -> int:
        if not hasattr(self, "n_features_out_"):
            raise RuntimeError("TileFeatures must be fitted first")
        return int(self.n_features_out_)

    @staticmethod
    def _validate(X: Any) -> None:
        if not isinstance(X, RawStateBatch):
            raise TypeError(
                "TileFeatures expects RawStateBatch; put DecodeRawState first"
            )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "one_hot_ids": self.one_hot_ids,
            "n_tiles": self.n_tiles,
            "include_states": self.include_states,
        }

    def set_params(self, **params: Any) -> "TileFeatures":
        for key, value in params.items():
            if key == "one_hot_ids":
                self.one_hot_ids = bool(value)
            elif key == "n_tiles":
                self.n_tiles = None if value is None else int(value)
            elif key == "include_states":
                self.include_states = bool(value)
            else:
                raise ValueError(f"unknown TileFeatures parameter: {key!r}")
        return self


class PlayerFeatures(StateTransformer):
    """The 16 raw player fields, in the shared field order."""

    fields = obs_spec.RAW_PLAYER_FIELDS

    def fit(self, X: RawStateBatch, y: Any = None) -> "PlayerFeatures":
        if not isinstance(X, RawStateBatch):
            raise TypeError(
                "PlayerFeatures expects RawStateBatch; put DecodeRawState first"
            )
        self.n_features_out_ = len(self.fields)
        return self

    def transform(self, X: RawStateBatch) -> np.ndarray:
        if not isinstance(X, RawStateBatch):
            raise TypeError(
                "PlayerFeatures expects RawStateBatch; put DecodeRawState first"
            )
        return np.stack([X.player[name] for name in self.fields], axis=1).astype(
            np.float32, copy=False
        )

    @property
    def output_size(self) -> int:
        return len(self.fields)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}


class CollisionFeatures(StateTransformer):
    """Flatten the eight directional collision triples."""

    def fit(self, X: RawStateBatch, y: Any = None) -> "CollisionFeatures":
        if not isinstance(X, RawStateBatch):
            raise TypeError(
                "CollisionFeatures expects RawStateBatch; put DecodeRawState first"
            )
        self.n_features_out_ = (
            obs_spec.RAW_DIRECTIONS * obs_spec.RAW_COLLISION_FLOATS
        )
        return self

    def transform(self, X: RawStateBatch) -> np.ndarray:
        if not isinstance(X, RawStateBatch):
            raise TypeError(
                "CollisionFeatures expects RawStateBatch; put DecodeRawState first"
            )
        return X.collisions.reshape(len(X), -1).astype(np.float32, copy=False)

    @property
    def output_size(self) -> int:
        return obs_spec.RAW_DIRECTIONS * obs_spec.RAW_COLLISION_FLOATS

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}


class FlowFeatures(StateTransformer):
    """Eight flow-field deltas, optionally scaled."""

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = float(scale)

    def fit(self, X: RawStateBatch, y: Any = None) -> "FlowFeatures":
        if not isinstance(X, RawStateBatch):
            raise TypeError(
                "FlowFeatures expects RawStateBatch; put DecodeRawState first"
            )
        self.n_features_out_ = obs_spec.RAW_FLOW_COUNT
        return self

    def transform(self, X: RawStateBatch) -> np.ndarray:
        if not isinstance(X, RawStateBatch):
            raise TypeError(
                "FlowFeatures expects RawStateBatch; put DecodeRawState first"
            )
        return (X.flow.astype(np.float32, copy=False) * self.scale)

    @property
    def output_size(self) -> int:
        return obs_spec.RAW_FLOW_COUNT

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"scale": self.scale}

    def set_params(self, **params: Any) -> "FlowFeatures":
        if "scale" in params:
            self.scale = float(params["scale"])
        return self
