"""Tests for the lean observation preprocessing pipeline."""

from __future__ import annotations

import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import obs_spec  # noqa: E402
from preprocessing import build_lean_pipeline  # noqa: E402


def _lean() -> np.ndarray:
    lean = np.zeros(obs_spec.LEAN_OBSERVATION_SIZE, dtype=np.float32)
    lean[obs_spec.LEAN_TILE_ID_OFFSET:obs_spec.LEAN_ROTATION_OFFSET] = 1.0
    lean[obs_spec.LEAN_ROTATION_OFFSET] = np.pi / 2.0
    lean[obs_spec.LEAN_PHASE_OFFSET] = -0.5

    player = obs_spec.LEAN_PLAYER_OFFSET
    lean[player + 0] = 0.5       # velocity_x
    lean[player + 1] = -0.25     # velocity_y
    lean[player + 2] = -1.0      # facing
    lean[player + 3] = 4.0       # state_id
    lean[player + 4] = 0.4       # jump_phase
    lean[player + 5] = -1.0      # powerup_0
    lean[player + 6] = 0.0       # powerup_1
    lean[player + 7] = 3.0       # powerup_2
    lean[player + 8] = 1.5       # gold_ratio

    collision = obs_spec.LEAN_COLLISION_OFFSET
    for direction in range(obs_spec.RAW_DIRECTIONS):
        base = collision + direction * obs_spec.RAW_COLLISION_FLOATS
        lean[base + 0] = 0.25
        lean[base + 1] = 1.0
        lean[base + 2] = 3.0

    lean[obs_spec.LEAN_FLOW_OFFSET:] = 0.5
    return lean


class LeanPreprocessingTest(unittest.TestCase):
    def test_lean_pipeline_one_hot_shape_and_values(self) -> None:
        lean = _lean()
        tile_names = ["E", "W", "Y", "B", "O", "J"]
        pre = build_lean_pipeline(tile_names=tile_names)
        out = pre.fit_transform(lean[None, :])
        expected = (
            obs_spec.GRID_CELLS * len(tile_names)
            + obs_spec.LEAN_ROTATION_COUNT
            + obs_spec.LEAN_PHASE_COUNT
            + 5
            + len(obs_spec.PLAYER_STATE_NAMES) + 1
            + 3 * 5
            + obs_spec.RAW_DIRECTIONS * 2
            + obs_spec.RAW_DIRECTIONS * obs_spec.TYPE_BITS
            + obs_spec.RAW_FLOW_COUNT
        )
        self.assertEqual(out.shape, (1, expected))
        self.assertEqual(pre.output_size, expected)
        self.assertEqual(pre.transform(lean).shape, (expected,))

        tile_offset = obs_spec.GRID_CELLS * len(tile_names)
        rotation_offset = tile_offset
        phase_offset = rotation_offset + obs_spec.LEAN_ROTATION_COUNT
        numeric_player_offset = phase_offset + obs_spec.LEAN_PHASE_COUNT
        state_offset = numeric_player_offset + 5
        powerup_offset = state_offset + len(obs_spec.PLAYER_STATE_NAMES) + 1
        collision_numeric_offset = powerup_offset + 3 * 5
        collision_type_offset = collision_numeric_offset + obs_spec.RAW_DIRECTIONS * 2
        flow_offset = (
            collision_type_offset + obs_spec.RAW_DIRECTIONS * obs_spec.TYPE_BITS
        )

        np.testing.assert_allclose(out[0, 0:len(tile_names)], [0, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(float(out[0, rotation_offset]), np.pi / 2.0)
        self.assertAlmostEqual(float(out[0, phase_offset]), -0.5)
        np.testing.assert_allclose(
            out[0, numeric_player_offset:numeric_player_offset + 5],
            [0.5, -0.25, -1.0, 0.4, 1.5],
        )
        self.assertAlmostEqual(float(out[0, state_offset + 4]), 1.0)
        self.assertAlmostEqual(float(out[0, powerup_offset + 0]), 1.0)
        self.assertAlmostEqual(float(out[0, powerup_offset + 5 + 1]), 1.0)
        self.assertAlmostEqual(float(out[0, powerup_offset + 10 + 4]), 1.0)
        self.assertAlmostEqual(float(out[0, collision_type_offset + 3]), 1.0)
        np.testing.assert_allclose(
            out[0, flow_offset:flow_offset + obs_spec.RAW_FLOW_COUNT],
            0.5,
        )

    def test_single_and_batch_transform_match(self) -> None:
        lean = _lean()
        batch = np.stack([lean, lean + 0.1])
        pre = build_lean_pipeline(tile_names=["E", "W", "Y", "B", "O", "J"])
        batch_out = pre.fit_transform(batch)
        single_out = pre.transform(lean)
        self.assertEqual(single_out.shape, batch_out.shape[1:])
        np.testing.assert_allclose(single_out, batch_out[0])

    def test_pickle_round_trip(self) -> None:
        lean = _lean()
        pre = build_lean_pipeline(tile_names=["E", "W", "Y", "B", "O", "J"])
        expected = pre.fit_transform(lean[None, :])
        restored = pickle.loads(pickle.dumps(pre))
        np.testing.assert_allclose(restored.transform(lean[None, :]), expected)

    def test_rejects_non_lean_input(self) -> None:
        pre = build_lean_pipeline(tile_names=["E", "W"])
        with self.assertRaises(ValueError):
            pre.fit_transform(np.zeros((1, obs_spec.RAW_OBSERVATION_SIZE)))


if __name__ == "__main__":
    unittest.main()
