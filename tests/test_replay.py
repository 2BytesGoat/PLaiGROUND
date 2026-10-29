from __future__ import annotations

import unittest

import numpy as np
from gymnasium import spaces

from plaiground.algorithms.reinforcement.replay import (
    ProtectedReplayBuffer,
    RareReplayBuffer,
    state_bucket_key,
)


class ReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)
        self.n_envs = 2

    def _add(self, buffer, value: float = 0.0) -> None:
        obs = np.full((self.n_envs, 4), value, dtype=np.float32)
        next_obs = np.full((self.n_envs, 4), value + 0.1, dtype=np.float32)
        actions = np.zeros((self.n_envs, 1), dtype=np.int64)
        rewards = np.ones(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=np.float32)
        infos = [
            {
                "level": "1-1",
                "start_kind": "canonical" if index == 0 else "state",
                "cell": [index, 2],
                "facing_direction": 1,
                "run_time": 0.25 * index,
                "powerup_stack": [-1, -1, -1],
            }
            for index in range(self.n_envs)
        ]
        buffer.add(obs, next_obs, actions, rewards, dones, infos)

    def test_bucket_key(self) -> None:
        self.assertEqual(
            state_bucket_key(
                {
                    "level": "1-2",
                    "cell": [3, 4],
                    "facing_direction": 1,
                    "run_time": 1.1,
                    "powerup_stack": [-1, -1, -1],
                }
            ),
            "1-2|3,4|1|2|-1,-1,-1",
        )

    def test_rare_buffer_tracks_buckets(self) -> None:
        buffer = RareReplayBuffer(
            8,
            self.observation_space,
            self.action_space,
            n_envs=self.n_envs,
        )
        for index in range(4):
            self._add(buffer, float(index))
        self.assertEqual(buffer.size(), 4)
        counts = buffer.bucket_counts()
        self.assertTrue(counts)
        self.assertTrue(all(count > 0 for count in counts.values()))
        buffer.set_return(0, 0, 10.0)
        buffer.q_values[0, 0] = 5.0
        gap_stats = buffer.compute_gap_priorities(quantile=0.5)
        self.assertGreater(gap_stats["positive"], 0)
        self.assertGreater(float(buffer.priorities[0, 0]), 0.0)
        samples = buffer.sample(16)
        self.assertEqual(samples.observations.shape[0], 16)
        self.assertIsNotNone(buffer._probability_cache)
        buffer.set_priority(1, 0, 5.0)
        self.assertIsNone(buffer._probability_cache)

    def test_protected_buffer_mixes_online_and_anchor(self) -> None:
        anchor = RareReplayBuffer(
            8,
            self.observation_space,
            self.action_space,
            n_envs=self.n_envs,
        )
        for index in range(3):
            self._add(anchor, float(index))
        buffer = ProtectedReplayBuffer(
            16,
            self.observation_space,
            self.action_space,
            n_envs=self.n_envs,
            anchor_buffer=anchor,
            anchor_fraction=0.5,
        )
        for index in range(3):
            self._add(buffer, float(index + 10))
        self.assertEqual(buffer.size(), 6)
        samples = buffer.sample(8)
        self.assertEqual(samples.observations.shape[0], 8)
        self.assertEqual(samples.actions.shape[0], 8)
        stats = buffer.stats()
        self.assertEqual(stats["anchor_slots"], 3)
        self.assertEqual(stats["online_slots"], 3)


if __name__ == "__main__":
    unittest.main()
