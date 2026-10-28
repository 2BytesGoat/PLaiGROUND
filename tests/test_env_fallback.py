"""Regression tests for training-env info keys used by VecMonitor."""

from __future__ import annotations

import unittest

import numpy as np

from plaiground.envs.single_agent import SingleAgentGodotEnv


class FakeGame:
    def reset(self, spawns=None):
        return [{"obs": np.zeros(4, dtype=np.float32)}], [{}]

    def step(self, actions):
        return (
            [{"obs": np.zeros(4, dtype=np.float32)}],
            np.asarray([0.0], dtype=np.float32),
            np.asarray([True]),
            np.asarray([False]),
            [{}],
        )

    def close(self):
        pass


class FallbackInfoTest(unittest.TestCase):
    def test_canonical_fallback_has_every_monitor_key(self) -> None:
        env = object.__new__(SingleAgentGodotEnv)
        env.game = FakeGame()
        env.level = "1-1"
        env.max_decisions = 150
        env.action_repeat = 4
        env.force_start_jump = False
        env.obs_encoding = "raw"
        env.preprocessor = None
        env.start_pool = [{"prefix_actions": [1]}]
        env.start_weights = np.asarray([1.0], dtype=np.float64)
        env.start_rng = np.random.default_rng(0)
        env.last_start = {}
        env._start_info = {}
        env._steps = 0
        env.grid_dropout = 0.0
        env.sensor_dropout = 0.0
        env.dropout_rng = np.random.default_rng(0)
        env.trajectory_dir = None
        env.trajectory_max_time = 0.0
        env._trajectory_obs = []
        env._trajectory_actions = []
        env._trajectory_id = 0
        env._last_obs = None

        _, info = env.reset()
        self.assertEqual(info["start_source"], "canonical_fallback")
        for key in (
            "start_cell",
            "start_facing",
            "start_distance",
            "prefix_len",
        ):
            self.assertIn(key, info)


if __name__ == "__main__":
    unittest.main()
