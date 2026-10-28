"""Decode the game's flat raw observation into named arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import obs_spec
from .base import StateTransformer


@dataclass(frozen=True)
class RawState:
    """One decoded raw observation."""

    flat: np.ndarray
    tile_ids: np.ndarray
    tile_states: np.ndarray
    player: dict[str, float]
    collisions: np.ndarray
    flow: np.ndarray


@dataclass(frozen=True)
class RawStateBatch:
    """A decoded batch of raw observations.

    All fields keep a leading batch axis. ``first()`` returns one
    ``RawState`` for debugging and direct use.
    """

    flat: np.ndarray
    tile_ids: np.ndarray
    tile_states: np.ndarray
    player: dict[str, np.ndarray]
    collisions: np.ndarray
    flow: np.ndarray

    def __len__(self) -> int:
        return int(self.flat.shape[0])

    def first(self) -> RawState:
        return RawState(
            flat=self.flat[0],
            tile_ids=self.tile_ids[0],
            tile_states=self.tile_states[0],
            player={name: float(values[0]) for name, values in self.player.items()},
            collisions=self.collisions[0],
            flow=self.flow[0],
        )


class DecodeRawState(StateTransformer):
    """Turn ``(342,)`` or ``(N, 342)`` into ``RawStateBatch``.

    The flat input is validated with the shared ``obs_spec`` contract. The
    output is always batched; ``StatePipeline`` removes the batch axis again
    for a single input.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: Any, y: Any = None) -> "DecodeRawState":
        array = self._batch(X)
        self.n_features_in_ = int(array.shape[1])
        return self

    def transform(self, X: Any) -> RawStateBatch:
        array = self._batch(X)
        count = int(array.shape[0])
        tile_ids = array[:, obs_spec.RAW_TILE_ID_OFFSET:obs_spec.RAW_TILE_STATE_OFFSET]
        tile_ids = tile_ids.astype(np.int64).reshape(
            count, obs_spec.GRID_SIZE, obs_spec.GRID_SIZE
        )
        tile_states = array[:, obs_spec.RAW_TILE_STATE_OFFSET:obs_spec.RAW_PLAYER_OFFSET]
        tile_states = tile_states.reshape(
            count, obs_spec.GRID_CELLS, obs_spec.RAW_TILE_STATE_FLOATS
        )
        player_values = array[
            :, obs_spec.RAW_PLAYER_OFFSET:obs_spec.RAW_COLLISION_OFFSET
        ]
        player = {
            name: player_values[:, index]
            for index, name in enumerate(obs_spec.RAW_PLAYER_FIELDS)
        }
        collisions = array[
            :, obs_spec.RAW_COLLISION_OFFSET:obs_spec.RAW_FLOW_OFFSET
        ].reshape(count, obs_spec.RAW_DIRECTIONS, obs_spec.RAW_COLLISION_FLOATS)
        flow = array[
            :, obs_spec.RAW_FLOW_OFFSET:obs_spec.RAW_OBSERVATION_SIZE
        ]
        return RawStateBatch(
            flat=array,
            tile_ids=tile_ids,
            tile_states=tile_states,
            player=player,
            collisions=collisions,
            flow=flow,
        )

    @staticmethod
    def _batch(X: Any) -> np.ndarray:
        array = np.asarray(X, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != obs_spec.RAW_OBSERVATION_SIZE:
            raise ValueError(
                f"Expected a raw observation of shape ({obs_spec.RAW_OBSERVATION_SIZE},) "
                f"or (N, {obs_spec.RAW_OBSERVATION_SIZE}), got {array.shape}"
            )
        return obs_spec.validate_obs_batch(array, obs_spec.ENCODING_RAW)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    def set_params(self, **params: Any) -> "DecodeRawState":
        if params:
            raise ValueError(f"DecodeRawState has no parameters: {sorted(params)}")
        return self
