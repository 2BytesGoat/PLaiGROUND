"""Tests for shared vector-environment construction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np

from plaiground.envs import factory
from plaiground.experiments.runner import Run


class TinyGym(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        info = {
            "level": "1-1",
            "finished": True,
            "timed_out": False,
            "run_time": 1.0,
            "start_source": "canonical",
            "start_kind": "canonical",
            "start_cell": None,
            "start_facing": None,
            "start_distance": None,
            "prefix_len": 0,
        }
        return np.zeros(4, dtype=np.float32), 1.0, True, False, info


class FactoryTest(unittest.TestCase):
    def _run(self, root: Path, *, n_envs: int = 1) -> Run:
        return Run(
            config={
                "env": {"level": "1-1", "n_envs": n_envs},
                "algorithm": {"name": "ppo", "timesteps": 10},
                "output": {},
            },
            config_path=root / "config.toml",
            output_dir=root / "run",
            timesteps=10,
            resume=None,
            preprocessing_path=root / "pre.pkl",
            show=False,
            device="cpu",
            verbose=0,
        )

    def test_single_env_monitor_accepts_all_info_keywords(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(Path(tmp))
            with mock.patch.object(
                factory, "make_env", return_value=TinyGym()
            ):
                vec_env = factory.build_vec_env(run, "/nonexistent", [], None)
            try:
                vec_env.reset()
                _, _, dones, infos = vec_env.step(
                    np.zeros(1, dtype=np.int64)
                )
                self.assertTrue(bool(dones[0]))
                self.assertIn("episode", infos[0])
                self.assertTrue(run.monitor_path.exists())
            finally:
                vec_env.close()

    def test_multi_level_untagged_pool_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(Path(tmp), n_envs=2)
            run.config["env"]["levels"] = ["1-1", "1-2"]
            with self.assertRaises(ValueError):
                factory.build_vec_env(
                    run,
                    "/nonexistent",
                    [{"cell": [4, 7], "facing": 1}],
                    None,
                )


if __name__ == "__main__":
    unittest.main()
