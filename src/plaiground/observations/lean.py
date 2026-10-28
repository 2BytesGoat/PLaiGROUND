"""Decode and encode the game's lean 188-value observation.

Godot now owns the lean state:

    49 primary tile IDs
    49 rotations (radians)
    49 signed phases
     9 player values
    24 collision values
     8 flow deltas
    = 188

Python owns representation: this module decodes the packet and one-hots the
categorical fields (tile IDs, player state, powerup stack, collision types).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from . import spec as obs_spec
from .base import StateTransformer


@dataclass(frozen=True)
class LeanState:
    """One decoded lean observation."""

    flat: np.ndarray
    tile_ids: np.ndarray
    rotations: np.ndarray
    phases: np.ndarray
    player: dict[str, float]
    collisions: np.ndarray
    flow: np.ndarray


@dataclass(frozen=True)
class LeanStateBatch:
    """A decoded batch of lean observations."""

    flat: np.ndarray
    tile_ids: np.ndarray
    rotations: np.ndarray
    phases: np.ndarray
    player: dict[str, np.ndarray]
    collisions: np.ndarray
    flow: np.ndarray

    def __len__(self) -> int:
        return int(self.flat.shape[0])

    def first(self) -> LeanState:
        return LeanState(
            flat=self.flat[0],
            tile_ids=self.tile_ids[0],
            rotations=self.rotations[0],
            phases=self.phases[0],
            player={name: float(values[0]) for name, values in self.player.items()},
            collisions=self.collisions[0],
            flow=self.flow[0],
        )


class DecodeLeanState(StateTransformer):
    """Turn ``(188,)`` or ``(N, 188)`` into ``LeanStateBatch``."""

    def fit(self, X: Any, y: Any = None) -> "DecodeLeanState":
        array = self._batch(X)
        self.n_features_in_ = int(array.shape[1])
        return self

    def transform(self, X: Any) -> LeanStateBatch:
        array = self._batch(X)
        count = int(array.shape[0])
        tile_ids = array[
            :, obs_spec.LEAN_TILE_ID_OFFSET:obs_spec.LEAN_ROTATION_OFFSET
        ].astype(np.int64)
        tile_ids = tile_ids.reshape(count, obs_spec.GRID_SIZE, obs_spec.GRID_SIZE)
        rotations = array[
            :, obs_spec.LEAN_ROTATION_OFFSET:obs_spec.LEAN_PHASE_OFFSET
        ].reshape(count, obs_spec.GRID_CELLS)
        phases = array[
            :, obs_spec.LEAN_PHASE_OFFSET:obs_spec.LEAN_PLAYER_OFFSET
        ].reshape(count, obs_spec.GRID_CELLS)
        player_values = array[
            :, obs_spec.LEAN_PLAYER_OFFSET:obs_spec.LEAN_COLLISION_OFFSET
        ]
        player = {
            name: player_values[:, index]
            for index, name in enumerate(obs_spec.LEAN_PLAYER_FIELDS)
        }
        collisions = array[
            :, obs_spec.LEAN_COLLISION_OFFSET:obs_spec.LEAN_FLOW_OFFSET
        ].reshape(count, obs_spec.RAW_DIRECTIONS, obs_spec.RAW_COLLISION_FLOATS)
        flow = array[
            :, obs_spec.LEAN_FLOW_OFFSET:obs_spec.LEAN_OBSERVATION_SIZE
        ]
        return LeanStateBatch(
            flat=array,
            tile_ids=tile_ids,
            rotations=rotations,
            phases=phases,
            player=player,
            collisions=collisions,
            flow=flow,
        )

    @staticmethod
    def _batch(X: Any) -> np.ndarray:
        array = np.asarray(X, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != obs_spec.LEAN_OBSERVATION_SIZE:
            raise ValueError(
                f"Expected a lean observation of shape "
                f"({obs_spec.LEAN_OBSERVATION_SIZE},) or "
                f"(N, {obs_spec.LEAN_OBSERVATION_SIZE}), got {array.shape}"
            )
        return obs_spec.validate_obs_batch(array, obs_spec.ENCODING_LEAN)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    def set_params(self, **params: Any) -> "DecodeLeanState":
        if params:
            raise ValueError(f"DecodeLeanState has no parameters: {sorted(params)}")
        return self


def _one_hot(
    values: np.ndarray,
    categories: Sequence[int],
    *,
    name: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    index = np.full(values.shape, len(categories) - 1, dtype=np.int64)
    for category_index, category in enumerate(categories):
        index[values == int(category)] = category_index
    if np.any(values < min(categories)) or np.any(values > max(categories)):
        raise ValueError(
            f"{name} value outside categories {min(categories)}..{max(categories)}: "
            f"min={int(values.min())}, max={int(values.max())}"
        )
    encoded = np.eye(len(categories), dtype=np.float32)[index]
    return encoded.reshape(values.shape[0], -1)


class LeanFeatures(StateTransformer):
    """One-hot/encode the lean packet for a model.

    Categorical fields:

    - 49 tile IDs over ``tile_names`` (or inferred IDs)
    - player ``state_id`` over the player state names plus unknown
    - three powerup stack slots over ``[-1, 0, 1, 2, 3]``
    - 8 collision object-type IDs over the fixed eight type categories
  (or an explicit `collision_types` vocabulary)

    Numeric fields stay numeric:
    - rotations, phases
    - player velocity x/y, facing, jump phase, gold ratio
    - collision distance/hit
    - flow
    """

    def __init__(
        self,
        tile_names: Iterable[str] | None = None,
        state_names: Iterable[str] | None = None,
        collision_types: Iterable[int] | None = None,
        flow_scale: float = 1.0,
    ) -> None:
        self.tile_names = None if tile_names is None else list(tile_names)
        self.state_names = (
            list(obs_spec.PLAYER_STATE_NAMES)
            if state_names is None
            else list(state_names)
        )
        self.collision_types = (
            None if collision_types is None else [int(item) for item in collision_types]
        )
        self.flow_scale = float(flow_scale)

    def fit(self, X: LeanStateBatch, y: Any = None) -> "LeanFeatures":
        self._validate(X)
        tile_ids = X.tile_ids
        if tile_ids.size and int(tile_ids.min()) < 0:
            raise ValueError("lean tile IDs must be non-negative")
        self.n_tiles_ = (
            len(self.tile_names)
            if self.tile_names is not None
            else max(1, int(tile_ids.max()) + 1 if tile_ids.size else 1)
        )
        if self.n_tiles_ <= 0:
            raise ValueError("tile vocabulary must be non-empty")
        max_tile = int(tile_ids.max()) if tile_ids.size else -1
        if max_tile >= self.n_tiles_:
            raise ValueError(
                f"tile ID {max_tile} outside tile vocabulary of size {self.n_tiles_}"
            )

        self.n_states_ = len(self.state_names) + 1
        max_state = int(X.player["state_id"].max()) if X.player["state_id"].size else 0
        if max_state >= self.n_states_:
            self.n_states_ = max_state + 1

        self.collision_categories_ = (
            list(self.collision_types)
            if self.collision_types is not None
            else list(range(obs_spec.TYPE_BITS))
        )
        max_collision = (
            int(X.collisions[:, :, 2].max()) if X.collisions.size else 0
        )
        if max_collision >= len(self.collision_categories_):
            raise ValueError(
                f"collision type {max_collision} outside vocabulary "
                f"of size {len(self.collision_categories_)}"
            )

        self.n_features_out_ = (
            obs_spec.GRID_CELLS * self.n_tiles_
            + obs_spec.LEAN_ROTATION_COUNT
            + obs_spec.LEAN_PHASE_COUNT
            + 5  # velocity x/y, facing, jump phase, gold ratio
            + self.n_states_
            + 3 * 5  # three powerup slots, -1 + four kinds
            + obs_spec.RAW_DIRECTIONS * 2  # collision distance + hit
            + obs_spec.RAW_DIRECTIONS * len(self.collision_categories_)
            + obs_spec.RAW_FLOW_COUNT
        )
        return self

    def transform(self, X: LeanStateBatch) -> np.ndarray:
        self._validate(X)
        count = len(X)
        tile_ids = X.tile_ids.reshape(count, -1)
        tile_one_hot = np.eye(self.n_tiles_, dtype=np.float32)[tile_ids].reshape(
            count, -1
        )

        numeric_player = np.stack(
            [
                X.player["velocity_x"],
                X.player["velocity_y"],
                X.player["facing"],
                X.player["jump_phase"],
                X.player["gold_ratio"],
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        state_one_hot = _one_hot(
            X.player["state_id"],
            list(range(self.n_states_)),
            name="state_id",
        )
        powerup_one_hot = np.concatenate(
            [
                _one_hot(
                    X.player[f"powerup_{index}"],
                    [-1, 0, 1, 2, 3],
                    name=f"powerup_{index}",
                )
                for index in range(3)
            ],
            axis=1,
        )

        collision_numeric = X.collisions[:, :, :2].reshape(count, -1)
        collision_types = X.collisions[:, :, 2].reshape(count, -1)
        collision_one_hot = _one_hot(
            collision_types,
            self.collision_categories_,
            name="collision_type",
        )

        return np.concatenate(
            [
                tile_one_hot,
                X.rotations,
                X.phases,
                numeric_player,
                state_one_hot,
                powerup_one_hot,
                collision_numeric,
                collision_one_hot,
                X.flow * self.flow_scale,
            ],
            axis=1,
        ).astype(np.float32, copy=False)

    @property
    def output_size(self) -> int:
        if not hasattr(self, "n_features_out_"):
            raise RuntimeError("LeanFeatures must be fitted first")
        return int(self.n_features_out_)

    @staticmethod
    def _validate(X: Any) -> None:
        if not isinstance(X, LeanStateBatch):
            raise TypeError(
                "LeanFeatures expects LeanStateBatch; put DecodeLeanState first"
            )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "tile_names": self.tile_names,
            "state_names": self.state_names,
            "collision_types": self.collision_types,
            "flow_scale": self.flow_scale,
        }

    def set_params(self, **params: Any) -> "LeanFeatures":
        for key, value in params.items():
            if key == "tile_names":
                self.tile_names = None if value is None else list(value)
            elif key == "state_names":
                self.state_names = list(value)
            elif key == "collision_types":
                self.collision_types = (
                    None if value is None else [int(item) for item in value]
                )
            elif key == "flow_scale":
                self.flow_scale = float(value)
            else:
                raise ValueError(f"unknown LeanFeatures parameter: {key!r}")
        return self
