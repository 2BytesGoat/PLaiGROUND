"""Unit checks for Godot's static + columnar info wire format."""

from __future__ import annotations

import json
import struct
import unittest
import numpy as np

from plaiground.envs.godot import GodotEnv, _normalise_spawn
from plaiground.envs.stable_baselines_wrapper import StableBaselinesGodotEnv


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

    def test_reward_telemetry_fields_decode(self) -> None:
        columns = _columns()
        columns.update(
            {
                "gold_ratio": [3.0, 0.5],
                "flow_cost": [12.0, 0.0],
                "flow_in_bounds": [True, False],
                "wasted_air_frames": [4, 0],
            }
        )
        infos = _env()._decode_info(columns)
        self.assertAlmostEqual(infos[0]["gold_ratio"], 3.0)
        self.assertAlmostEqual(infos[0]["flow_cost"], 12.0)
        self.assertTrue(infos[0]["flow_in_bounds"])
        self.assertEqual(infos[0]["wasted_air_frames"], 4)
        self.assertAlmostEqual(infos[1]["gold_ratio"], 0.5)
        self.assertFalse(infos[1]["flow_in_bounds"])

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

    def test_normalise_spawn_flattens_clean_start_state(self) -> None:
        entry = {
            "agent": 0,
            "cell": [4, 7],
            "facing": -1,
            "position": [64.0, 112.0],
            "velocity": [0.0, 0.0],
            "state_name": "Move",
            "powerup_stack": [1],
            "run_time": 1.25,
            "best_run_time": 9.0,
            "run_active": True,
            "started_walking": True,
            "world_state": {"version": 1, "objects": {}},
        }
        result = _normalise_spawn(entry)
        self.assertEqual(result["agent"], 0)
        self.assertEqual(result["facing"], -1)
        self.assertEqual(result["position"], [64.0, 112.0])
        self.assertEqual(result["powerup_stack"], [1])
        self.assertAlmostEqual(result["run_time"], 1.25)
        self.assertEqual(result["world_state"], {"version": 1, "objects": {}})
        self.assertNotIn("state", result)

    def test_normalise_spawn_reads_legacy_nested_state(self) -> None:
        result = _normalise_spawn(
            {
                "agent": 0,
                "cell": [4, 7],
                "facing": 1,
                "state": {
                    "velocity": [0.0, 0.0],
                    "state_name": "Move",
                    "run_time": 0.5,
                },
            }
        )
        self.assertAlmostEqual(result["run_time"], 0.5)
        self.assertEqual(result["state_name"], "Move")
        self.assertNotIn("state", result)

    def test_powerup_stack_decodes_from_columnar_info(self) -> None:
        columns = _columns()
        columns["powerup_stack"] = [[1.0, -1.0, -1.0], [0.0, 2.0, -1.0]]
        infos = _env()._decode_info(columns)
        self.assertEqual(infos[0]["powerup_stack"], [1.0, -1.0, -1.0])
        self.assertEqual(infos[1]["powerup_stack"], [0.0, 2.0, -1.0])

    def test_request_info_round_trip(self) -> None:
        env = object.__new__(GodotEnv)
        env.supports_info_request = True
        sent = {}

        def fake_send(message):
            sent.update(message)

        env._send_as_json = fake_send
        env._get_json_dict = lambda: {
            "type": "info",
            "request": ["world_state"],
            "world_state": {"0": {"version": 1, "objects": {}}},
        }
        response = env.request_info(["world_state"], agents=[0])
        self.assertEqual(sent["type"], "info_request")
        self.assertEqual(sent["agents"], [0])
        self.assertEqual(
            response["world_state"]["0"]["version"],
            1,
        )

    def test_vec_request_info_maps_global_agent_indices(self) -> None:
        class FakeEnv:
            def __init__(self, count: int, offset: int) -> None:
                self.num_envs = count
                self.offset = offset
                self.calls: list[list[int]] = []

            def request_info(self, keys, agents=None):
                self.calls.append(list(agents))
                return {
                    "type": "info",
                    "request": list(keys),
                    "world_state": {
                        str(index): {"version": 1, "offset": self.offset}
                        for index in agents
                    },
                }

        wrapper = object.__new__(StableBaselinesGodotEnv)
        wrapper.envs = [FakeEnv(2, 0), FakeEnv(2, 2)]
        response = wrapper.request_info(["world_state"], agents=[1, 3])
        self.assertEqual(wrapper.envs[0].calls, [[1]])
        self.assertEqual(wrapper.envs[1].calls, [[1]])
        self.assertEqual(
            sorted(response["world_state"].keys()),
            ["1", "3"],
        )
        self.assertEqual(response["world_state"]["1"]["offset"], 0)
        self.assertEqual(response["world_state"]["3"]["offset"], 2)


if __name__ == "__main__":
    unittest.main()
