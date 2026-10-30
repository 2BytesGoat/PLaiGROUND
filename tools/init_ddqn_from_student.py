#!/usr/bin/env python3
"""Write a DDQN checkpoint initialized from a distilled student MLP.

This is a warm-start adapter for the supervised-to-DDQN experiment loop. It
does not launch Godot: a small dummy environment supplies the observation and
action spaces, the student's ReLU layers are copied into the Q-network and its
target copy, and the resulting checkpoint is saved for normal DDQN training or
evaluation.

Example:
    .venv/bin/python tools/init_ddqn_from_student.py \
      --config experiments/0131_ddqn_1_14_student_warmstart \
      --student runs/distill_campaign_student/students/cand_verified_onehot957_512/student.pt \
      --output runs/ddqn_1_14_student_warmstart/dqn/student_init.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from plaiground.algorithms.reinforcement.ddqn import DoubleDQN
from plaiground.algorithms.reinforcement.dqn import (
    make_model,
    warm_start_from_student,
)
from plaiground.experiments.runner import load_run, resolve_path


class SpaceEnv(gym.Env):
    """Minimal env used only to construct an SB3 model."""

    def __init__(self, observation_shape: tuple[int, ...]) -> None:
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=observation_shape,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0,
            True,
            False,
            {},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = load_run(args.config, device=args.device)
    input_size = int(
        run.config.get("observation", {}).get("model_input_size", 0)
    )
    if input_size <= 0:
        raise SystemExit(
            f"{run.config_path}: [observation].model_input_size is required"
        )
    observation_shape = (input_size,)
    vec_env = DummyVecEnv([lambda: SpaceEnv(observation_shape)])
    try:
        model = make_model(run, vec_env, DoubleDQN)
        warm_start_from_student(model, run, args.student)
        output = (
            resolve_path(args.output)
            if args.output is not None
            else run.output_dir / "dqn" / "student_init.zip"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output))
    finally:
        vec_env.close()
    print(f"saved warm-start checkpoint: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
