#!/usr/bin/env python3
"""Collect finish-validated rollouts from a DDQN policy for distillation.

The policy drives the ``lean_stick`` 191 environment. By default the greedy
policy action is the training label, which is correct when the deterministic
DDQN itself finishes; epsilon-greedy noise lets the collector visit recovery
states. Use ``--label-policy executed`` when only noisy rollouts finish. Both
label kinds are saved. Every episode is saved with a per-route ``route_success``
flag. The student trainer filters on that flag, so failed explorations remain
available for diagnostics but never enter training.

This is the DDQN counterpart to ``collect_noisy_teacher_paths.py``. It is meant
for the 1-14 loop where no NEAT route pool exists.

Example:
    .venv/bin/python tools/collect_ddqn_paths.py \
      --level 1-14 \
      --model runs/ddqn_1_14_student_warmstart/dqn/dqn_final.zip \
      --preprocessing runs/ddqn_fresh_1_1_lean_stick/preprocessing.pkl \
      --output-dir runs/ddqn_1_14_student_warmstart/paths/final \
      --episodes 32 --agents 8 --epsilon 0.02 \
      --reassert-held-jump --port 37800
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from plaiground.envs.godot import GodotEnv
from plaiground.envs.runtime import isolated_home, resolve_env_path
from plaiground.observations.lean_io import load_lean_preprocessor
from plaiground.observations.lean_stick import LeanStickPreprocessor


def make_env(
    *,
    level: str,
    port: int,
    output_dir: Path,
    num_agents: int,
    action_repeat: int,
    speedup: float,
    timeout: int,
    show: bool,
    reassert_held_jump: bool,
) -> GodotEnv:
    GodotEnv.DEFAULT_TIMEOUT = timeout
    return GodotEnv(
        env_path=resolve_env_path(),
        port=port,
        show_window=show,
        seed=42,
        action_repeat=action_repeat,
        speedup=speedup,
        nb_agents=num_agents,
        level=level,
        convert_action_space=True,
        log_path=str(output_dir / "rollout.log"),
        env=isolated_home(output_dir, "home"),
        reward_profile="default",
        obs_encoding="lean_stick",
        reassert_held_jump="true" if reassert_held_jump else "false",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument(
        "--label-policy",
        choices=["greedy", "executed"],
        default="greedy",
        help=(
            "Training label for each state. Use greedy when the deterministic "
            "DDQN finishes; use executed when success depends on exploration."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "holdout"], default="train")
    parser.add_argument("--port", type=int, default=37800)
    parser.add_argument("--max-decisions", type=int, default=180)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--speedup", type=float, default=12.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-reassert-held-jump", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_lean_preprocessor(args.preprocessing)
    preprocessor = (
        loaded
        if isinstance(loaded, LeanStickPreprocessor)
        else LeanStickPreprocessor(loaded)
    )
    model = DQN.load(str(args.model), device="cpu")
    if int(model.observation_space.shape[0]) != preprocessor.output_size:
        raise SystemExit(
            f"{args.model}: expected {preprocessor.output_size} inputs, got "
            f"{model.observation_space.shape}"
        )
    rng = np.random.default_rng(args.seed)

    env = make_env(
        level=args.level,
        port=args.port,
        output_dir=args.output_dir,
        num_agents=args.agents,
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        timeout=args.timeout,
        show=args.show,
        reassert_held_jump=not args.no_reassert_held_jump,
    )

    raw_rows: list[np.ndarray] = []
    onehot_rows: list[np.ndarray] = []
    greedy_rows: list[int] = []
    executed_rows: list[int] = []
    route_index_rows: list[int] = []
    step_rows: list[int] = []
    success_flags: list[bool] = []
    signatures: list[str] = []
    episode_results: list[dict] = []

    try:
        next_route = 0
        while next_route < args.episodes:
            batch_size = min(args.agents, args.episodes - next_route)
            obs, _ = env.reset()
            batch_raw: list[list[np.ndarray]] = [[] for _ in range(batch_size)]
            batch_onehot: list[list[np.ndarray]] = [
                [] for _ in range(batch_size)
            ]
            batch_actions: list[list[int]] = [[] for _ in range(batch_size)]
            batch_executed: list[list[int]] = [[] for _ in range(batch_size)]
            batch_steps: list[list[int]] = [[] for _ in range(batch_size)]
            done = np.zeros(batch_size, dtype=bool)
            finished = np.zeros(batch_size, dtype=bool)
            run_times: list[float | None] = [None] * batch_size

            for step in range(args.max_decisions):
                active = np.flatnonzero(~done)
                if active.size == 0:
                    break
                raw = np.asarray(
                    [obs[index]["obs"] for index in active], dtype=np.float32
                )
                onehot = np.asarray(
                    preprocessor.transform(raw), dtype=np.float32
                )
                labels, _ = model.predict(onehot, deterministic=True)
                labels = np.asarray(labels, dtype=np.int64).reshape(-1)
                noisy = labels.copy()
                flips = rng.random(len(noisy)) < args.epsilon
                noisy[flips] = 1 - noisy[flips]

                for offset, index in enumerate(active):
                    batch_raw[index].append(raw[offset])
                    batch_onehot[index].append(onehot[offset])
                    batch_actions[index].append(int(labels[offset]))
                    batch_executed[index].append(int(noisy[offset]))
                    batch_steps[index].append(step)

                actions = np.zeros((args.agents, 1), dtype=np.int64)
                actions[active, 0] = noisy
                obs, _rewards, dones, _timeouts, infos = env.step(actions)
                for offset, index in enumerate(active):
                    if not bool(dones[index]):
                        continue
                    done[index] = True
                    info = infos[index] if index < len(infos) else {}
                    finished[index] = bool(info.get("finished", False))
                    if finished[index]:
                        value = info.get("run_time")
                        if value is not None:
                            run_times[index] = float(value)

            for index in range(batch_size):
                route_index = next_route + index
                ok = bool(finished[index])
                if batch_actions[index]:
                    signature = (
                        f"ddqn:{args.model.stem}:{route_index:03d}"
                    )
                    raw_rows.extend(batch_raw[index])
                    onehot_rows.extend(batch_onehot[index])
                    greedy_rows.extend(batch_actions[index])
                    executed_rows.extend(batch_executed[index])
                    route_index_rows.extend(
                        [route_index] * len(batch_actions[index])
                    )
                    step_rows.extend(batch_steps[index])
                    success_flags.extend([ok] * len(batch_actions[index]))
                    signatures.extend([signature] * len(batch_actions[index]))
                episode_results.append(
                    {
                        "episode": route_index,
                        "steps": len(batch_actions[index]),
                        "finished": ok,
                        "run_time": run_times[index],
                    }
                )
                print(
                    f"[episode {route_index:03d}] steps="
                    f"{len(batch_actions[index])} finished={ok} "
                    f"time={run_times[index]}",
                    flush=True,
                )
            next_route += batch_size

        successes = sum(1 for result in episode_results if result["finished"])
        if executed_rows:
            labels = (
                greedy_rows
                if args.label_policy == "greedy"
                else executed_rows
            )
            np.savez_compressed(
                args.output_dir / "transitions.npz",
                obs_raw=np.asarray(raw_rows, dtype=np.float32),
                obs_onehot=np.asarray(onehot_rows, dtype=np.float32),
                actions=np.asarray(labels, dtype=np.int64),
                greedy_actions=np.asarray(greedy_rows, dtype=np.int64),
                executed_actions=np.asarray(executed_rows, dtype=np.int64),
                route_index=np.asarray(route_index_rows, dtype=np.int32),
                step=np.asarray(step_rows, dtype=np.int32),
                route_success=np.asarray(success_flags, dtype=bool),
                split=np.asarray([args.split] * len(executed_rows)),
                route_signature=np.asarray(signatures),
                level=np.asarray([args.level] * len(executed_rows)),
            )
        metadata = {
            "level": args.level,
            "model": str(args.model),
            "split": args.split,
            "agents": args.agents,
            "episodes": args.episodes,
            "epsilon": args.epsilon,
            "seed": args.seed,
            "action_repeat": args.action_repeat,
            "reassert_held_jump": not args.no_reassert_held_jump,
            "successes": successes,
            "failures": args.episodes - successes,
            "label_policy": args.label_policy,
            "transitions": len(executed_rows),
            "successful_transitions": sum(
                1 for flag in success_flags if flag
            ),
            "episode_results": episode_results,
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    key: metadata[key]
                    for key in (
                        "level",
                        "episodes",
                        "epsilon",
                        "successes",
                        "failures",
                        "transitions",
                        "successful_transitions",
                    )
                },
                indent=2,
            )
        )
        return 0 if successes else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
