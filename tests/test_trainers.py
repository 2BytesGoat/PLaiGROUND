"""Small model-construction tests for the PPO/DQN adapters."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from plaiground.algorithms.reinforcement import dqn, ppo
from plaiground.experiments.runner import Run
from plaiground.models.student import StudentMLP


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

    def test_dqn_warm_start_copies_student_into_q_networks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._run(root, "dqn")
            student = StudentMLP(4, [8, 8])
            checkpoint = root / "student.pt"
            torch.save(
                {
                    "input_dim": 4,
                    "hidden": [8, 8],
                    "input_kind": "raw191",
                    "state_dict": student.state_dict(),
                },
                checkpoint,
            )
            vec_env = DummyVecEnv([TinyEnv])
            try:
                model = dqn._make_model(run, vec_env)
                dqn.warm_start_from_student(model, run, checkpoint)
                student_state = student.state_dict()
                for name, source in student_state.items():
                    target = model.q_net.state_dict()[
                        f"q_net.{name.removeprefix('net.')}"
                    ]
                    self.assertTrue(torch.equal(source, target))
                for name, target in model.q_net_target.state_dict().items():
                    self.assertTrue(
                        torch.equal(target, model.q_net.state_dict()[name])
                    )
            finally:
                vec_env.close()

    def test_dqn_learning_rate_override_updates_schedule_and_optimizer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = self._run(Path(tmp), "dqn")
            vec_env = DummyVecEnv([TinyEnv])
            try:
                model = dqn._make_model(run, vec_env)
                model.learning_rate = 0.0001
                model._setup_lr_schedule()
                override = Run(
                    config={
                        **run.config,
                        "algorithm": {
                            **run.config["algorithm"],
                            "learning_rate": 0.00005,
                        },
                    },
                    config_path=run.config_path,
                    output_dir=run.output_dir,
                    timesteps=run.timesteps,
                    resume=run.resume,
                    preprocessing_path=run.preprocessing_path,
                    show=run.show,
                    device=run.device,
                    verbose=run.verbose,
                )
                dqn._apply_learning_rate_override(model, override)
                self.assertAlmostEqual(model.learning_rate, 0.00005)
                self.assertAlmostEqual(model.lr_schedule(1.0), 0.00005)
                for group in model.policy.optimizer.param_groups:
                    self.assertAlmostEqual(group["lr"], 0.00005)
            finally:
                vec_env.close()


if __name__ == "__main__":
    unittest.main()
