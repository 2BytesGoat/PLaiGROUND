#!/usr/bin/env python3
"""Run a fixed-length PPO continuation as a training-throughput A/B.

Both arms start from the same checkpoint and run the same number of timesteps;
only the Godot training override differs. The reported decisions/s is the
training loop (environment stepping plus the PPO update), which is the number
that actually matters for wall-clock time.

Every launch writes the A41 training override (the worker pool is capped to one
thread) next to the game, or into the project root for a source shim; pass
``--no-worker-cap`` for the baseline arm of an A/B.

Example (the 2026-11-04 A41 gate):
    .venv/bin/python tools/benchmark_ppo_ab.py \
        --env-path /tmp/GodotSource.app \
        --start-model runs/threaded_754/pool_absent_r1/ppo_754.zip \
        --timesteps 105600 --agents-per-process 8 --processes 8 --speedup 16
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer

from plaiground.envs.runtime import (
    DISABLE_TRAINING_OVERRIDE,
    remove_training_override,
)
from plaiground.envs.vec_env import MultiAgentGodotVecEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-path", required=True)
    parser.add_argument("--start-model", required=True)
    parser.add_argument("--out", default=None, help="Output dir (default: runs/ppo_ab)")
    parser.add_argument("--timesteps", type=int, default=105_600)
    parser.add_argument("--agents-per-process", type=int, default=8)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--level", default="1-1")
    parser.add_argument("--speedup", type=float, default=16.0)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-worker-cap",
        action="store_true",
        help=(
            "A/B baseline arm: do not write the A41 training override and remove "
            "a harness-written override from the environment."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_worker_cap:
        os.environ[DISABLE_TRAINING_OVERRIDE] = "1"
        for removed in remove_training_override(args.env_path):
            print(f"removed training override: {removed}")

    out = Path(args.out or "runs/ppo_ab").resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = MultiAgentGodotVecEnv(
        env_path=args.env_path,
        level=args.level,
        n_agents_per_process=args.agents_per_process,
        n_processes=args.processes,
        seed=args.seed,
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        max_decisions=150,
        base_port=args.base_port,
        output_dir=out / "godot",
        obs_encoding="multi_hot",
        reward_profile="time",
        training_no_cosmetics=True,
    )
    try:
        model = PPO.load(args.start_model, env=env, device=args.device)
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        model.n_epochs = args.epochs
        model.rollout_buffer = RolloutBuffer(
            model.n_steps,
            model.observation_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=env.num_envs,
        )
        model._last_obs = None
        model._last_episode_starts = None
        print(
            f"start steps={model.num_timesteps} envs={env.num_envs} "
            f"n_steps={model.n_steps} batch={model.batch_size}",
            flush=True,
        )
        started = time.perf_counter()
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        elapsed = time.perf_counter() - started
        print(
            f"RESULT steps={model.num_timesteps} wall_s={elapsed:.2f} "
            f"decisions_per_s={args.timesteps / elapsed:.1f}",
            flush=True,
        )
        model.save(str(out / "ppo_ab.zip"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
