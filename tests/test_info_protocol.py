"""Unit checks for Godot's static + columnar info wire format."""

from __future__ import annotations

import json
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wrappers.godot_env import GodotEnv  # noqa: E402


def _env() -> GodotEnv:
    env = object.__new__(GodotEnv)
    env.num_envs = 2
    env.info_schema = 2
    env.static_info = {
        "obs_encoding": "multi_hot",
        "obs_size": 754,
        "tile_names": ["Wall", "Floor"],
    }
    env._spawn_cells = [[10, 14], [-1, -1]]
    return env


def _columns() -> dict:
    return {
        "schema": 2,
        "global_position": [[1.0, 2.0], [3.0, 4.0]],
        "facing_direction": [1, -1],
        "state": ["Run", "Idle"],
        "distinct_cells_touched": [4, 0],
        "run_time": [0.5, 0.0],
        "best_run_time": [float("inf"), float("inf")],
        "finished": [False, True],
        "deaths": [0, 1],
        "efficiency": [1.0, 0.75],
        "flow": [0.75, 0.5],
    }


class InfoProtocolTest(unittest.TestCase):
    def test_static_and_dynamic_info_are_reconstructed(self) -> None:
        infos = _env()._decode_info(_columns())
        self.assertEqual(len(infos), 2)
        first, second = infos
        self.assertEqual(first["obs_encoding"], "multi_hot")
        self.assertEqual(first["obs_size"], 754)
        self.assertEqual(first["tile_names"], ["Wall", "Floor"])
        self.assertEqual(first["spawn_cell"], [10, 14])
        self.assertEqual(first["global_position"], [1.0, 2.0])
        self.assertEqual(first["state"], "Run")
        self.assertEqual(first["distinct_cells_touched"], 4)
        self.assertAlmostEqual(first["run_time"], 0.5)
        self.assertEqual(first["facing_direction"], 1)
        self.assertFalse(first["finished"])
        self.assertEqual(first["deaths"], 0)
        self.assertAlmostEqual(first["efficiency"], 1.0)
        self.assertAlmostEqual(first["flow"], 0.75)
        self.assertEqual(second["spawn_cell"], [-1, -1])
        self.assertTrue(second["finished"])
        self.assertEqual(second["deaths"], 1)

    def test_reset_info_updates_only_reset_spawn_cells(self) -> None:
        env = _env()
        env._apply_reset_info(
            {
                "reset_info": {
                    "agents": [1],
                    "spawn_cell": [[20, 13]],
                }
            }
        )
        self.assertEqual(env._spawn_cells, [[10, 14], [20, 13]])

    def test_binary_observation_frame_decodes(self) -> None:
        env = _env()
        header = {
            "type": "step",
            "schema": 2,
            "obs_count": 2,
            "obs_size": 3,
            "reward": [1.0, 2.0],
            "done": [False, True],
            "info": _columns(),
        }
        header_bytes = json.dumps(header).encode()
        raw = np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="<f4"
        ).tobytes()
        payload = (
            b"B"
            + struct.pack("<I", len(header_bytes))
            + header_bytes
            + raw
        )
        response = env._decode_binary_response(payload)
        self.assertEqual(response["type"], "step")
        self.assertEqual(len(response["obs"]), 2)
        np.testing.assert_allclose(
            response["obs"][0]["obs"], np.asarray([1.0, 2.0, 3.0])
        )
        np.testing.assert_allclose(
            response["obs"][1]["obs"], np.asarray([4.0, 5.0, 6.0])
        )
        self.assertEqual(response["info"]["schema"], 2)

    def test_legacy_list_info_is_untouched(self) -> None:
        legacy = [{"flow": 1.0}, {"flow": 0.5}]
        self.assertIs(_env()._decode_info(legacy), legacy)


if __name__ == "__main__":
    unittest.main()
