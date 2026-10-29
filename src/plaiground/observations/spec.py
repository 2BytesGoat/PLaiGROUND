"""Shared AI observation contract for Dragon Jump.

The game emits a flat float vector. This module mirrors the game's
``AIObservationSpec`` so the Python training/export path can validate the
width, label the layout, and decode it for debugging without guessing.

Encodings
---------
legacy
    The shipped 58-value vector. Do not use for new multi-hot runs.
legacy_stick
    The legacy 58 values plus three hold/ceiling flags: on_ceiling,
    jump_held, and stick_active. 61 values; version 6. Use for new
    hold-sensitive searches without changing the frozen legacy contract.
multi_hot
    A 7x7 semantic grid flattened row-major, followed by player/route
    features. Each cell has 15 values:

    type multi-hot (8) + powerup kind (2) + dissolve state (2) +
    dissolve timer (1) + bounce orientation (2)

    Empty cells are all zeros. A cell can set multiple type bits (for
    example wall + secret, or wall + powerup).

raw
    Primitive local state for Python-side encoding: 7x7 tile IDs, 7x7 tile
    states, player state (including clamped ``gold_ratio``), 8-way collisions,
    and 8-way flow deltas. 342 values.
lean
    Godot-emitted compact state: 49 primary tile IDs, 49 rotations, 49 signed
    phases, 9 player values, 24 collisions, and 8 flow deltas. 188 values.
    Python one-hots the categoricals for the model.
lean_stick
    The 188 lean values plus on_ceiling, jump_held, and stick_active. 191
    values; version 7. The first 188 values remain byte-for-byte lean.
"""

from __future__ import annotations

import numpy as np

ENCODING_LEGACY = "legacy"
# Versioned extension of the shipped 58-value vector. The first 58 values are
# byte-for-byte legacy; the appended flags expose held-jump / ceiling-stick
# state for levels such as 1-4. The 58-value contract stays frozen.
ENCODING_LEGACY_STICK = "legacy_stick"
ENCODING_MULTI_HOT = "multi_hot"
ENCODING_MULTI_HOT_SENSOR = "multi_hot_sensor"
ENCODING_RAW = "raw"
ENCODING_LEAN = "lean"
# Lean 188 plus on_ceiling, jump_held, and stick_active. The first 188 values
# stay byte-for-byte identical to lean.
ENCODING_LEAN_STICK = "lean_stick"
SUPPORTED_ENCODINGS = (
    ENCODING_LEGACY,
    ENCODING_LEGACY_STICK,
    ENCODING_MULTI_HOT,
    ENCODING_MULTI_HOT_SENSOR,
    ENCODING_RAW,
    ENCODING_LEAN,
    ENCODING_LEAN_STICK,
)

SPEC_VERSION_LEGACY = 1
SPEC_VERSION_LEGACY_STICK = 6
SPEC_VERSION_MULTI_HOT = 2
SPEC_VERSION_MULTI_HOT_SENSOR = 3
SPEC_VERSION_RAW = 4
SPEC_VERSION_LEAN = 5
SPEC_VERSION_LEAN_STICK = 7

LEGACY_OBSERVATION_SIZE = 58
LEGACY_ON_CEILING_OFFSET = LEGACY_OBSERVATION_SIZE
LEGACY_JUMP_HELD_OFFSET = LEGACY_ON_CEILING_OFFSET + 1
LEGACY_STICK_ACTIVE_OFFSET = LEGACY_JUMP_HELD_OFFSET + 1
LEGACY_STICK_OBSERVATION_SIZE = LEGACY_STICK_ACTIVE_OFFSET + 1

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
# Game-side collision lifecycle timings. The raw timer is normalized by
# DISSOLVE_TIMER_SCALE; the lean encoder converts it to a phase.
DISSOLVE_COLLISION_DISABLE_TIME = 0.7
DISSOLVE_REPAIR_TIME = 0.5

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

# Raw local-state encoding. The game sends primitive state; Python builds the
# learning features. Keep in sync with AIObservationSpec.RAW_*.
RAW_TILE_ID_COUNT = GRID_CELLS
RAW_TILE_STATE_FLOATS = 5
RAW_TILE_STATE_COUNT = GRID_CELLS * RAW_TILE_STATE_FLOATS
RAW_PLAYER_FLOATS = 16
RAW_GOLD_RATIO_CAP = 3.0
RAW_DIRECTIONS = 8
RAW_COLLISION_FLOATS = 3
RAW_COLLISION_COUNT = RAW_DIRECTIONS * RAW_COLLISION_FLOATS
RAW_FLOW_COUNT = RAW_DIRECTIONS
RAW_FLOW_DELTA_SCALE = 4.0
RAW_TILE_ID_OFFSET = 0
RAW_TILE_STATE_OFFSET = RAW_TILE_ID_OFFSET + RAW_TILE_ID_COUNT
RAW_PLAYER_OFFSET = RAW_TILE_STATE_OFFSET + RAW_TILE_STATE_COUNT
RAW_COLLISION_OFFSET = RAW_PLAYER_OFFSET + RAW_PLAYER_FLOATS
RAW_FLOW_OFFSET = RAW_COLLISION_OFFSET + RAW_COLLISION_COUNT
RAW_OBSERVATION_SIZE = RAW_FLOW_OFFSET + RAW_FLOW_COUNT

# Lean model-facing layout: one primary symbol ID, one rotation, and one
# signed phase per cell, followed by the player/collision/flow blocks.
LEAN_TILE_ID_COUNT = GRID_CELLS
LEAN_ROTATION_COUNT = GRID_CELLS
LEAN_PHASE_COUNT = GRID_CELLS
LEAN_CELL_FEATURE_COUNT = (
    LEAN_TILE_ID_COUNT + LEAN_ROTATION_COUNT + LEAN_PHASE_COUNT
)
LEAN_PLAYER_FLOATS = 9
LEAN_TILE_ID_OFFSET = 0
LEAN_ROTATION_OFFSET = LEAN_TILE_ID_OFFSET + LEAN_TILE_ID_COUNT
LEAN_PHASE_OFFSET = LEAN_ROTATION_OFFSET + LEAN_ROTATION_COUNT
LEAN_PLAYER_OFFSET = LEAN_PHASE_OFFSET + LEAN_PHASE_COUNT
LEAN_COLLISION_OFFSET = LEAN_PLAYER_OFFSET + LEAN_PLAYER_FLOATS
LEAN_FLOW_OFFSET = LEAN_COLLISION_OFFSET + RAW_COLLISION_COUNT
LEAN_OBSERVATION_SIZE = LEAN_FLOW_OFFSET + RAW_FLOW_COUNT
LEAN_STICK_ON_CEILING_OFFSET = LEAN_OBSERVATION_SIZE
LEAN_STICK_JUMP_HELD_OFFSET = LEAN_STICK_ON_CEILING_OFFSET + 1
LEAN_STICK_STICK_ACTIVE_OFFSET = LEAN_STICK_JUMP_HELD_OFFSET + 1
LEAN_STICK_OBSERVATION_SIZE = LEAN_STICK_STICK_ACTIVE_OFFSET + 1
LEAN_PLAYER_FIELDS = (
    "velocity_x",
    "velocity_y",
    "facing",
    "state_id",
    "jump_phase",
    "powerup_0",
    "powerup_1",
    "powerup_2",
    "gold_ratio",
)

RAW_TILE_STATE_FIELDS = (
    "powerup_kind",
    "dissolve_state",
    "dissolve_timer",
    "bounce_x",
    "bounce_y",
)
RAW_PLAYER_FIELDS = (
    "cell_x",
    "cell_y",
    "velocity_x",
    "velocity_y",
    "facing",
    "state_id",
    "on_floor",
    "on_wall",
    "jump_phase",
    "powerup_0",
    "powerup_1",
    "powerup_2",
    "gold_ratio",
    "run_active",
    "started_walking",
    "finished",
)
RAW_DIRECTION_NAMES = ("E", "SE", "S", "SW", "W", "NW", "N", "NE")
PLAYER_STATE_NAMES = (
    "Idle",
    "Move",
    "Jump",
    "Fall",
    "Walled",
    "Bounce",
    "Dash",
    "DoubleJump",
    "Grapple",
    "Stomp",
    "Celebrate",
    "Die",
)


def expected_obs_size(encoding: str) -> int:
    if encoding == ENCODING_LEGACY:
        return LEGACY_OBSERVATION_SIZE
    if encoding == ENCODING_LEGACY_STICK:
        return LEGACY_STICK_OBSERVATION_SIZE
    if encoding == ENCODING_MULTI_HOT:
        return MULTI_HOT_OBSERVATION_SIZE
    if encoding == ENCODING_MULTI_HOT_SENSOR:
        return MULTI_HOT_SENSOR_OBSERVATION_SIZE
    if encoding == ENCODING_RAW:
        return RAW_OBSERVATION_SIZE
    if encoding == ENCODING_LEAN:
        return LEAN_OBSERVATION_SIZE
    if encoding == ENCODING_LEAN_STICK:
        return LEAN_STICK_OBSERVATION_SIZE
    raise ValueError(f"Unknown observation encoding: {encoding!r}")


def spec_version(encoding: str) -> int:
    if encoding == ENCODING_LEGACY:
        return SPEC_VERSION_LEGACY
    if encoding == ENCODING_LEGACY_STICK:
        return SPEC_VERSION_LEGACY_STICK
    if encoding == ENCODING_MULTI_HOT:
        return SPEC_VERSION_MULTI_HOT
    if encoding == ENCODING_MULTI_HOT_SENSOR:
        return SPEC_VERSION_MULTI_HOT_SENSOR
    if encoding == ENCODING_RAW:
        return SPEC_VERSION_RAW
    if encoding == ENCODING_LEAN:
        return SPEC_VERSION_LEAN
    if encoding == ENCODING_LEAN_STICK:
        return SPEC_VERSION_LEAN_STICK
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


def decode_legacy_stick(obs: np.ndarray) -> dict:
    """Decode the versioned 61-value legacy+hold observation."""
    array = validate_obs(obs, ENCODING_LEGACY_STICK)
    return {
        "legacy": array[:LEGACY_OBSERVATION_SIZE],
        "on_ceiling": float(array[LEGACY_ON_CEILING_OFFSET]),
        "jump_held": float(array[LEGACY_JUMP_HELD_OFFSET]),
        "stick_active": float(array[LEGACY_STICK_ACTIVE_OFFSET]),
    }


def decode_lean(obs: np.ndarray) -> dict:
    """Decode one lean 188-value observation for diagnostics/tests."""
    array = validate_obs(obs, ENCODING_LEAN)
    player_values = array[LEAN_PLAYER_OFFSET:LEAN_COLLISION_OFFSET]
    return {
        "flat": array,
        "tile_ids": array[LEAN_TILE_ID_OFFSET:LEAN_ROTATION_OFFSET]
        .astype(np.int64)
        .reshape(GRID_SIZE, GRID_SIZE),
        "rotations": array[LEAN_ROTATION_OFFSET:LEAN_PHASE_OFFSET].reshape(
            GRID_SIZE, GRID_SIZE
        ),
        "phases": array[LEAN_PHASE_OFFSET:LEAN_PLAYER_OFFSET].reshape(
            GRID_SIZE, GRID_SIZE
        ),
        "player": {
            name: float(value)
            for name, value in zip(LEAN_PLAYER_FIELDS, player_values)
        },
        "collisions": array[LEAN_COLLISION_OFFSET:LEAN_FLOW_OFFSET].reshape(
            RAW_DIRECTIONS, RAW_COLLISION_FLOATS
        ),
        "flow": array[LEAN_FLOW_OFFSET:LEAN_OBSERVATION_SIZE],
    }


def decode_lean_stick(obs: np.ndarray) -> dict:
    """Decode the versioned 191-value lean+hold observation."""
    array = validate_obs(obs, ENCODING_LEAN_STICK)
    base = decode_lean(array[:LEAN_OBSERVATION_SIZE])
    return {
        **base,
        "on_ceiling": float(array[LEAN_STICK_ON_CEILING_OFFSET]),
        "jump_held": float(array[LEAN_STICK_JUMP_HELD_OFFSET]),
        "stick_active": float(array[LEAN_STICK_STICK_ACTIVE_OFFSET]),
    }


def decode_raw(obs: np.ndarray) -> dict:
    """Decode one raw local-state observation for diagnostics/tests.

    Returns named slices; the flat array is also kept. This is not used in the
    training hot path.
    """
    array = validate_obs(obs, ENCODING_RAW)
    tile_ids = array[
        RAW_TILE_ID_OFFSET:RAW_TILE_STATE_OFFSET
    ].astype(np.int64).reshape(GRID_SIZE, GRID_SIZE)
    tile_states = array[
        RAW_TILE_STATE_OFFSET:RAW_PLAYER_OFFSET
    ].reshape(GRID_CELLS, RAW_TILE_STATE_FLOATS)
    player_values = array[RAW_PLAYER_OFFSET:RAW_COLLISION_OFFSET]
    player = {
        name: float(value)
        for name, value in zip(RAW_PLAYER_FIELDS, player_values)
    }
    collisions = array[
        RAW_COLLISION_OFFSET:RAW_FLOW_OFFSET
    ].reshape(RAW_DIRECTIONS, RAW_COLLISION_FLOATS)
    flow = array[RAW_FLOW_OFFSET:RAW_OBSERVATION_SIZE]
    return {
        "flat": array,
        "tile_ids": tile_ids,
        "tile_states": tile_states,
        "player": player,
        "collisions": collisions,
        "flow": flow,
    }
