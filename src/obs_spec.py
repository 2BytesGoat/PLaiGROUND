"""Shared AI observation contract for Dragon Jump.

The game emits a flat float vector. This module mirrors the game's
``AIObservationSpec`` so the Python training/export path can validate the
width, label the layout, and decode it for debugging without guessing.

Encodings
---------
legacy
    The shipped 58-value vector. Do not use for new multi-hot runs.
multi_hot
    A 7x7 semantic grid flattened row-major, followed by player/route
    features. Each cell has 15 values:

    type multi-hot (8) + powerup kind (2) + dissolve state (2) +
    dissolve timer (1) + bounce orientation (2)

    Empty cells are all zeros. A cell can set multiple type bits (for
    example wall + secret, or wall + powerup).
"""

from __future__ import annotations

import numpy as np

ENCODING_LEGACY = "legacy"
ENCODING_MULTI_HOT = "multi_hot"
ENCODING_MULTI_HOT_SENSOR = "multi_hot_sensor"
SUPPORTED_ENCODINGS = (
    ENCODING_LEGACY,
    ENCODING_MULTI_HOT,
    ENCODING_MULTI_HOT_SENSOR,
)

SPEC_VERSION_LEGACY = 1
SPEC_VERSION_MULTI_HOT = 2
SPEC_VERSION_MULTI_HOT_SENSOR = 3

LEGACY_OBSERVATION_SIZE = 58

GRID_RADIUS = 3
GRID_SIZE = 7
GRID_CELLS = GRID_SIZE * GRID_SIZE

TYPE_SOLID = 0
TYPE_HAZARD = 1
TYPE_EXIT = 2
TYPE_BOUNCE = 3
TYPE_DISSOLVE = 4
TYPE_POWERUP = 5
TYPE_PICKUP = 6
TYPE_SECRET = 7
TYPE_BITS = 8
TYPE_NAMES = (
    "solid",
    "hazard",
    "exit",
    "bounce",
    "dissolve",
    "powerup",
    "pickup",
    "secret",
)

POWERUP_KIND_BITS = 2
POWERUP_KIND_NAMES = ("DoubleJump", "Stomp", "Dash", "Grapple")

DISSOLVE_STATE_SOLID = 0
DISSOLVE_STATE_DISSOLVING = 1
DISSOLVE_STATE_DISABLED = 2
DISSOLVE_STATE_REPAIRING = 3
DISSOLVE_STATE_BITS = 2
DISSOLVE_STATE_NAMES = ("solid", "dissolving", "disabled", "repairing")
DISSOLVE_TIMER_FLOATS = 1
DISSOLVE_TIMER_SCALE = 3.0

BOUNCE_ORIENTATION_FLOATS = 2

CELL_TYPE_OFFSET = 0
CELL_POWERUP_KIND_OFFSET = CELL_TYPE_OFFSET + TYPE_BITS
CELL_DISSOLVE_STATE_OFFSET = CELL_POWERUP_KIND_OFFSET + POWERUP_KIND_BITS
CELL_DISSOLVE_TIMER_OFFSET = CELL_DISSOLVE_STATE_OFFSET + DISSOLVE_STATE_BITS
CELL_BOUNCE_OFFSET = CELL_DISSOLVE_TIMER_OFFSET + DISSOLVE_TIMER_FLOATS
CELL_FEATURE_COUNT = CELL_BOUNCE_OFFSET + BOUNCE_ORIENTATION_FLOATS

GRID_FEATURE_COUNT = GRID_CELLS * CELL_FEATURE_COUNT

PLAYER_EXIT_DIRECTION_OFFSET = GRID_FEATURE_COUNT
PLAYER_TIME_TO_EXIT_OFFSET = PLAYER_EXIT_DIRECTION_OFFSET + 2
PLAYER_ROUTE_GRADIENT_OFFSET = PLAYER_TIME_TO_EXIT_OFFSET + 1
PLAYER_VELOCITY_OFFSET = PLAYER_ROUTE_GRADIENT_OFFSET + 2
PLAYER_ON_FLOOR_OFFSET = PLAYER_VELOCITY_OFFSET + 2
PLAYER_ON_WALL_OFFSET = PLAYER_ON_FLOOR_OFFSET + 1
PLAYER_JUMP_PHASE_OFFSET = PLAYER_ON_WALL_OFFSET + 1
PLAYER_POWERUP_COUNTS_OFFSET = PLAYER_JUMP_PHASE_OFFSET + 1
PLAYER_POWERUP_TOP_OFFSET = PLAYER_POWERUP_COUNTS_OFFSET + 4
PLAYER_POWERUP_TOTAL_OFFSET = PLAYER_POWERUP_TOP_OFFSET + 4
PLAYER_FEATURE_COUNT = (
    PLAYER_POWERUP_TOTAL_OFFSET + 1 - PLAYER_EXIT_DIRECTION_OFFSET
)

MULTI_HOT_OBSERVATION_SIZE = GRID_FEATURE_COUNT + PLAYER_FEATURE_COUNT
ROUTE_TIME_SCALE = 10.0

# Optional head-mounted ray sensor appended after the base multi-hot vector.
SENSOR_RAYS = 20
SENSOR_VALUES_PER_RAY = 2
SENSOR_FEATURE_COUNT = SENSOR_RAYS * SENSOR_VALUES_PER_RAY
MULTI_HOT_SENSOR_OBSERVATION_SIZE = (
    MULTI_HOT_OBSERVATION_SIZE + SENSOR_FEATURE_COUNT
)


def expected_obs_size(encoding: str) -> int:
    if encoding == ENCODING_LEGACY:
        return LEGACY_OBSERVATION_SIZE
    if encoding == ENCODING_MULTI_HOT:
        return MULTI_HOT_OBSERVATION_SIZE
    if encoding == ENCODING_MULTI_HOT_SENSOR:
        return MULTI_HOT_SENSOR_OBSERVATION_SIZE
    raise ValueError(f"Unknown observation encoding: {encoding!r}")


def spec_version(encoding: str) -> int:
    if encoding == ENCODING_LEGACY:
        return SPEC_VERSION_LEGACY
    if encoding == ENCODING_MULTI_HOT:
        return SPEC_VERSION_MULTI_HOT
    if encoding == ENCODING_MULTI_HOT_SENSOR:
        return SPEC_VERSION_MULTI_HOT_SENSOR
    raise ValueError(f"Unknown observation encoding: {encoding!r}")


def validate_obs(obs: np.ndarray, encoding: str) -> np.ndarray:
    array = np.asarray(obs, dtype=np.float32)
    expected = expected_obs_size(encoding)
    if array.ndim != 1 or array.shape[0] != expected:
        raise ValueError(
            f"Expected a flat {expected}-value {encoding} observation, "
            f"got shape {array.shape}"
        )
    return array


def validate_obs_batch(obs: np.ndarray, encoding: str) -> np.ndarray:
    array = np.asarray(obs, dtype=np.float32)
    expected = expected_obs_size(encoding)
    if array.ndim != 2 or array.shape[1] != expected:
        raise ValueError(
            f"Expected an (N, {expected}) {encoding} observation batch, "
            f"got shape {array.shape}"
        )
    return array


def decode_multi_hot(obs: np.ndarray) -> dict:
    """Decode one multi-hot observation for diagnostics/tests.

    Returns a dict with a ``(7, 7, 15)`` grid, the player/route fields, and
    named slices. This is not used in the training hot path.
    """
    array = validate_obs(obs, ENCODING_MULTI_HOT)
    grid = array[:GRID_FEATURE_COUNT].reshape(
        GRID_SIZE, GRID_SIZE, CELL_FEATURE_COUNT
    )
    player = array[GRID_FEATURE_COUNT:]
    return {
        "grid": grid,
        "type_bits": grid[:, :, CELL_TYPE_OFFSET:CELL_POWERUP_KIND_OFFSET],
        "powerup_kind_bits": grid[
            :, :, CELL_POWERUP_KIND_OFFSET:CELL_DISSOLVE_STATE_OFFSET
        ],
        "dissolve_state_bits": grid[
            :, :, CELL_DISSOLVE_STATE_OFFSET:CELL_DISSOLVE_TIMER_OFFSET
        ],
        "dissolve_timer": grid[:, :, CELL_DISSOLVE_TIMER_OFFSET],
        "bounce_orientation": grid[
            :, :, CELL_BOUNCE_OFFSET:CELL_FEATURE_COUNT
        ],
        "exit_direction": player[
            PLAYER_EXIT_DIRECTION_OFFSET - GRID_FEATURE_COUNT:
            PLAYER_TIME_TO_EXIT_OFFSET - GRID_FEATURE_COUNT
        ],
        "time_to_exit": player[
            PLAYER_TIME_TO_EXIT_OFFSET - GRID_FEATURE_COUNT
        ],
        "route_gradient": player[
            PLAYER_ROUTE_GRADIENT_OFFSET - GRID_FEATURE_COUNT:
            PLAYER_VELOCITY_OFFSET - GRID_FEATURE_COUNT
        ],
        "velocity": player[
            PLAYER_VELOCITY_OFFSET - GRID_FEATURE_COUNT:
            PLAYER_ON_FLOOR_OFFSET - GRID_FEATURE_COUNT
        ],
        "on_floor": player[
            PLAYER_ON_FLOOR_OFFSET - GRID_FEATURE_COUNT
        ],
        "on_wall": player[
            PLAYER_ON_WALL_OFFSET - GRID_FEATURE_COUNT
        ],
        "jump_phase": player[
            PLAYER_JUMP_PHASE_OFFSET - GRID_FEATURE_COUNT
        ],
        "powerup_counts": player[
            PLAYER_POWERUP_COUNTS_OFFSET - GRID_FEATURE_COUNT:
            PLAYER_POWERUP_TOP_OFFSET - GRID_FEATURE_COUNT
        ],
        "powerup_top": player[
            PLAYER_POWERUP_TOP_OFFSET - GRID_FEATURE_COUNT:
            PLAYER_POWERUP_TOTAL_OFFSET - GRID_FEATURE_COUNT
        ],
        "powerup_total": player[
            PLAYER_POWERUP_TOTAL_OFFSET - GRID_FEATURE_COUNT
        ],
    }
