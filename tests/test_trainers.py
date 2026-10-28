"""Small model-construction tests for the PPO/DQN adapters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from plaiground.algorithms.reinforcement import dqn, ppo
from plaiground.experiments.runner import Run


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(
        -np.inf, np.inf, (4,), dtype=np.float32
    )
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0.0, True, False, {}


class TrainerConstructionTest(unittest.TestCase):
    def _run(self, root: Path, name: str) -> Run:
        algorithm = {
            "name": name,
            "timesteps": 8,
            "n_steps": 8,
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 0.0001,
            "buffer_size": 100,
            "learning_starts": 0,
        }
        model = {
            "hidden": [8, 8],
            "activation": "tanh" if name == "ppo" else "relu",
        }
        return Run(
            config={
                "env": {"level": "1-1", "n_envs": 1},
                "algorithm": algorithm,
                "model": model,
                "output": {},
            },
            config_path=root / "config.toml",
            output_dir=root / "run",
            timesteps=8,
            resume=None,
            preprocessing_path=root / "pre.pkl",
            show=False,
            device="cpu",
            verbose=0,
        )

    def test_ppo_resume_overrides_pickle_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._run(root, "ppo")
            vec_env = DummyVecEnv([TinyEnv])
            try:
                model = ppo._make_model(run, vec_env)
                checkpoint = root / "checkpoint.zip"
                model.save(str(checkpoint))
                resumed = Run(
                    **{
                        **run.__dict__,
                        "resume": checkpoint,
                    }
                )
                with mock.patch(
                    "plaiground.experiments.runner.tensorboard_available",
                    return_value=False,
                ):
                    loaded = ppo._load_resumed_model(resumed, vec_env)
                loaded.save(str(root / "resumed.zip"))
                self.assertTrue((root / "resumed.zip").exists())
            finally:
                vec_env.close()

    def test_dqn_model_constructs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(Path(tmp), "dqn")
            vec_env = DummyVecEnv([TinyEnv])
            try:
                model = dqn._make_model(run, vec_env)
                self.assertIsNotNone(model.q_net)
            finally:
                vec_env.close()


if __name__ == "__main__":
    unittest.main()
