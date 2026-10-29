#!/usr/bin/env python3
"""Collect and save a DQN replay buffer without training.

This is a one-off warmup tool for DQN experiments. It loads the policy at the
requested epsilon, steps the environment, stores every transition in the
model's replay buffer, and writes the buffer + episode telemetry.

Example:
    .venv/bin/python tools/collect_dqn_buffer.py \
      --config experiments/0012_dqn_clean_start_1_1_8x1_1m \
      --model runs/dqn_clean_start_1_1_8x1_1m/dqn/dqn_final.zip \
      --transitions 200000 --epsilon 0.02
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from statistics import median

import numpy as np
from stable_baselines3 import DQN

from plaiground.envs.factory import build_vec_env
from plaiground.envs.runtime import resolve_env_path
from plaiground.experiments.runner import (
    load_run,
    resolve_path,
    train_start_pool,
)
from plaiground.algorithms.reinforcement.replay import RareReplayBuffer
from plaiground.observations.lean_io import load_lean_preprocessor
from plaiground.observations.lean_stick import LeanStickPreprocessor


def _episode_rows(infos: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for info in infos:
        episode = info.get("episode")
        if not isinstance(episode, dict):
            continue
        row = dict(episode)
        for key in ("start_kind", "start_source", "start_cell", "run_time",
                    "timed_out", "finished", "level"):
            if key in info:
                row[key] = info[key]
        rows.append(row)
    return rows


def _fmt_stats(name: str, rows: list[dict]) -> dict:
    canonical = [
        row
        for row in rows
        if str(row.get("start_kind")) == "canonical"
        and row.get("run_time") is not None
    ]
    finished = [row for row in canonical if bool(row.get("finished"))]
    deaths = [
        row
        for row in canonical
        if not bool(row.get("finished")) and not bool(row.get("timed_out"))
    ]
    finished_times = sorted(float(row["run_time"]) for row in finished)
    death_times = sorted(float(row["run_time"]) for row in deaths)
    stats = {
        "episodes": len(rows),
        "canonical_episodes": len(canonical),
        "canonical_finished": len(finished_times),
        "canonical_deaths": len(death_times),
        # ``best_time`` is kept for existing readers; it now means best finish.
        "best_time": min(finished_times) if finished_times else None,
        "best_finished_time": min(finished_times) if finished_times else None,
        "best_death_time": min(death_times) if death_times else None,
        "median_time": median(finished_times) if finished_times else None,
        "below_1_90": sum(value < 1.90 for value in finished_times),
        "below_1_75": sum(value < 1.75 for value in finished_times),
        "below_1_70": sum(value < 1.70 for value in finished_times),
        "below_1_6334": sum(value < 1.6334 for value in finished_times),
        "timeouts": sum(bool(row.get("timed_out")) for row in rows),
    }
    print(
        f"[collect {name}] episodes={stats['episodes']} "
        f"canonical={stats['canonical_episodes']} "
        f"finished={stats['canonical_finished']} "
        f"deaths={stats['canonical_deaths']} "
        f"best_finished={stats['best_finished_time']} "
        f"best_death={stats['best_death_time']} "
        f"median={stats['median_time']} "
        f"<1.75={stats['below_1_75']} <1.633={stats['below_1_6334']} "
        f"timeouts={stats['timeouts']}",
        flush=True,
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--transitions", type=int, default=200_000)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--rare-buffer",
        action="store_true",
        help="Store current-state buckets, returns, Q values, and gap priorities.",
    )
    parser.add_argument(
        "--gap-quantile",
        type=float,
        default=0.9,
        help="Keep positive return/Q gaps above this quantile.",
    )
    args = parser.parse_args()

    run = load_run(args.config)
    model_path = resolve_path(args.model)
    out_dir = (
        resolve_path(args.output)
        if args.output is not None
        else model_path.parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = None
    if run.preprocessing in {"lean", "lean_stick"}:
        preprocessor = load_lean_preprocessor(run.preprocessing_path)
        if run.preprocessing == "lean_stick" and not isinstance(
            preprocessor, LeanStickPreprocessor
        ):
            preprocessor = LeanStickPreprocessor(preprocessor)
    env = build_vec_env(
        run,
        resolve_env_path(),
        train_start_pool(run, required=True),
        preprocessor=preprocessor,
    )
    model = DQN.load(str(model_path), env=env, device=args.device)
    model.exploration_rate = float(args.epsilon)
    if args.rare_buffer:
        model.replay_buffer = RareReplayBuffer(
            buffer_size=int(run.algorithm.get("buffer_size", 200_000)),
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=model.device,
            n_envs=env.num_envs,
        )
        print(
            f"using rare replay buffer: "
            f"{model.replay_buffer.buffer_size * model.replay_buffer.n_envs} transitions",
            flush=True,
        )

    buffer = model.replay_buffer
    assert buffer is not None
    buffer.reset()
    target = int(args.transitions)
    collected = 0
    vector_steps = 0
    rows: list[dict] = []
    episode_transitions: list[list[tuple[int, float]]] = [
        [] for _ in range(env.num_envs)
    ]
    obs = env.reset()
    try:
        while collected < target:
            actions, _ = model.predict(obs, deterministic=False)
            actions = np.asarray(actions, dtype=np.int64).reshape(-1)
            slot = int(buffer.pos)
            new_obs, rewards, dones, infos = env.step(actions)
            buffer.add(
                obs,
                new_obs,
                actions,
                rewards,
                dones,
                infos,
            )
            for env_index in range(env.num_envs):
                episode_transitions[env_index].append(
                    (slot, float(rewards[env_index]))
                )
                if bool(dones[env_index]):
                    running = 0.0
                    for stored_slot, reward in reversed(
                        episode_transitions[env_index]
                    ):
                        running = reward + float(model.gamma) * running
                        if hasattr(buffer, "set_return"):
                            buffer.set_return(
                                stored_slot, env_index, running
                            )
                    episode_transitions[env_index].clear()
            collected += int(env.num_envs)
            vector_steps += 1
            rows.extend(_episode_rows(infos))
            obs = new_obs
            if vector_steps % 50 == 0:
                print(
                    f"[collect] transitions={collected}/{target} "
                    f"episodes={len(rows)}",
                    flush=True,
                )
    finally:
        env.close()

    gap_stats = {}
    if args.rare_buffer and hasattr(buffer, "compute_q_values"):
        q_count = buffer.compute_q_values(model.q_net)
        gap_stats = buffer.compute_gap_priorities(args.gap_quantile)
        print(
            f"[collect gaps] q={q_count} threshold={gap_stats['threshold']} "
            f"positive={gap_stats['positive']} "
            f"max_gap={gap_stats['max_gap']:.3f}",
            flush=True,
        )

    stats = _fmt_stats("final", rows)
    buffer_path = out_dir / f"replay_buffer_{target // 1000}k.pkl"
    with buffer_path.open("wb") as handle:
        pickle.dump(buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    episode_path = out_dir / f"collect_episodes_{target // 1000}k.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with episode_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "config": str(run.config_path),
        "model": str(model_path),
        "buffer": str(buffer_path),
        "episodes": str(episode_path),
        "transitions": collected,
        "vector_steps": vector_steps,
        "epsilon": float(args.epsilon),
        "n_envs": int(env.num_envs),
        "state_buckets": len(getattr(buffer, "bucket_names", {})),
        "priority_max": float(buffer.priorities[: buffer.size()].max())
        if hasattr(buffer, "priorities")
        else None,
        "gap_quantile": float(args.gap_quantile),
        "gap_stats": gap_stats,
        "stats": stats,
    }
    metadata_path = out_dir / f"replay_buffer_{target // 1000}k.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"buffer={buffer_path} size={buffer_path.stat().st_size / 1e9:.2f} GB")
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
