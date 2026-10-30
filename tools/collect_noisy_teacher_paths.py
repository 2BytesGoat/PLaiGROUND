#!/usr/bin/env python3
"""Collect finish-validated noisy teacher rollouts for late-level distillation.

The canonical collector records one clean path per teacher. That is enough to
reproduce the path in open loop, but it gives the student no examples of how to
recover after a small mistake. Plain student-driven DAgger adds those recovery
states, but it can also add states from runs that never finish, which is exactly
how a distillation set gets muddy.

This collector keeps the teacher in the loop. It executes the teacher action
with epsilon-greedy exploration, records the teacher's clean action as the
training label, and marks the whole episode ``route_success=True`` only when
both the 58-input teacher env and the 191-input replay env actually finish.
The trainer already filters on ``route_success``, so failed explorations can be
kept for diagnostics without entering the distillation set.

For 1-11 use true held-jump semantics on both sides::

    .venv/bin/python tools/collect_noisy_teacher_paths.py \
      --level 1-11 \
      --model archive/2026-10-27-cleanup/runs/teacher_champions/1-11/ppo_pilot.zip \
      --preprocessing runs/ddqn_fresh_1_1_lean_stick/preprocessing.pkl \
      --output-dir runs/distill_campaign_student/raw_noisy/1-11 \
      --episodes 40 --epsilon 0.05 \
      --reassert-held-jump --teacher-reassert-held-jump \
      --port 37600 --replay-port 37700

For 1-14 the only working teacher is the deployed 58-input student with legacy
edge-triggered semantics. Omit both reassertion flags there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from plaiground.envs.godot import GodotEnv
from plaiground.envs.runtime import isolated_home, resolve_env_path
from plaiground.observations.lean_io import load_lean_preprocessor
from plaiground.observations.lean_stick import LeanStickPreprocessor


class JSONPolicy:
    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        weights = data["weights"]
        hidden = [int(value) for value in data["hidden_sizes"]]
        self.input_dim = int(data["input_size"])
        self.w1 = np.asarray(weights["w1"], dtype=np.float32).reshape(
            hidden[0], self.input_dim
        )
        self.b1 = np.asarray(weights["b1"], dtype=np.float32).reshape(hidden[0])
        self.w2 = np.asarray(weights["w2"], dtype=np.float32).reshape(
            hidden[1], hidden[0]
        )
        self.b2 = np.asarray(weights["b2"], dtype=np.float32).reshape(hidden[1])
        self.w3 = np.asarray(weights["w3"], dtype=np.float32).reshape(2, hidden[1])
        self.b3 = np.asarray(weights["b3"], dtype=np.float32).reshape(2)

    def predict(self, observations, deterministic: bool = True):
        values = np.asarray(observations, dtype=np.float32)
        single = values.ndim == 1
        batch = values.reshape(1, -1) if single else values
        h1 = np.maximum(batch @ self.w1.T + self.b1, 0.0)
        h2 = np.maximum(h1 @ self.w2.T + self.b2, 0.0)
        logits = h2 @ self.w3.T + self.b3
        actions = np.argmax(logits, axis=1).astype(np.int64)
        return (actions[0] if single else actions), None


def make_env(
    *,
    level: str,
    encoding: str,
    port: int,
    output_dir: Path,
    tag: str,
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
        nb_agents=1,
        level=level,
        convert_action_space=True,
        log_path=str(output_dir / f"{tag}.log"),
        env=isolated_home(output_dir, tag),
        reward_profile="default",
        obs_encoding=encoding,
        reassert_held_jump="true" if reassert_held_jump else "false",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "holdout"], default="train")
    parser.add_argument("--port", type=int, default=37600)
    parser.add_argument("--replay-port", type=int, default=37700)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--speedup", type=float, default=12.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--reassert-held-jump",
        action="store_true",
        help="Use held-jump reassertion on the 191 replay side.",
    )
    parser.add_argument(
        "--teacher-reassert-held-jump",
        action="store_true",
        help="Also use held-jump reassertion on the 58 teacher side.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.model.suffix.lower()
    teacher_kind = "json" if suffix == ".json" else "ppo"
    teacher = (
        JSONPolicy(args.model)
        if teacher_kind == "json"
        else PPO.load(str(args.model), device="cpu")
    )
    if teacher_kind == "ppo" and int(teacher.observation_space.shape[0]) != 58:
        raise SystemExit(
            f"{args.model}: expected a 58-input PPO teacher, got "
            f"{teacher.observation_space.shape}"
        )
    if teacher_kind == "json" and int(teacher.input_dim) != 58:
        raise SystemExit(f"{args.model}: expected a 58-input JSON teacher")

    loaded = load_lean_preprocessor(args.preprocessing)
    preprocessor = (
        loaded
        if isinstance(loaded, LeanStickPreprocessor)
        else LeanStickPreprocessor(loaded)
    )
    rng = np.random.default_rng(args.seed)

    teacher_env = make_env(
        level=args.level,
        encoding="legacy",
        port=args.port,
        output_dir=args.output_dir,
        tag="teacher",
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        timeout=args.timeout,
        show=args.show,
        reassert_held_jump=args.teacher_reassert_held_jump,
    )
    replay_env = make_env(
        level=args.level,
        encoding="lean_stick",
        port=args.replay_port,
        output_dir=args.output_dir,
        tag="replay",
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        timeout=args.timeout,
        show=args.show,
        reassert_held_jump=args.reassert_held_jump,
    )

    legacy_rows: list[np.ndarray] = []
    raw_rows: list[np.ndarray] = []
    onehot_rows: list[np.ndarray] = []
    actions: list[int] = []
    steps: list[int] = []
    signatures: list[str] = []
    success_flags: list[bool] = []
    episode_results: list[dict] = []

    try:
        for episode in range(args.episodes):
            teacher_obs, _ = teacher_env.reset()
            replay_obs, _ = replay_env.reset()
            episode_legacy: list[np.ndarray] = []
            episode_raw: list[np.ndarray] = []
            episode_onehot: list[np.ndarray] = []
            episode_actions: list[int] = []
            teacher_done = False
            replay_done = False
            teacher_finished = False
            replay_finished = False
            teacher_time = None
            replay_time = None
            steps_taken = 0

            for step in range(args.max_decisions):
                if teacher_done or replay_done:
                    break
                legacy = np.asarray(teacher_obs[0]["obs"], dtype=np.float32)
                action_values, _ = teacher.predict(
                    legacy.reshape(1, -1), deterministic=True
                )
                clean_action = int(np.asarray(action_values).reshape(-1)[0])
                raw = np.asarray(replay_obs[0]["obs"], dtype=np.float32)
                onehot = np.asarray(
                    preprocessor.transform(raw.reshape(1, -1)), dtype=np.float32
                )[0]
                episode_legacy.append(legacy)
                episode_raw.append(raw)
                episode_onehot.append(onehot)
                episode_actions.append(clean_action)

                executed_action = clean_action
                if rng.random() < args.epsilon:
                    executed_action = int(rng.integers(0, 2))

                teacher_obs, _r, teacher_dones, _t, teacher_infos = teacher_env.step(
                    np.asarray([[executed_action]], dtype=np.int64)
                )
                replay_obs, _r2, replay_dones, _t2, replay_infos = replay_env.step(
                    np.asarray([[executed_action]], dtype=np.int64)
                )
                steps_taken = step + 1
                teacher_info = teacher_infos[0] if teacher_infos else {}
                replay_info = replay_infos[0] if replay_infos else {}
                if bool(teacher_dones[0]):
                    teacher_done = True
                    teacher_finished = bool(teacher_info.get("finished", False))
                    teacher_time = teacher_info.get("run_time") if teacher_finished else None
                if bool(replay_dones[0]):
                    replay_done = True
                    replay_finished = bool(replay_info.get("finished", False))
                    replay_time = replay_info.get("run_time") if replay_finished else None

            episode_ok = bool(teacher_finished and replay_finished)
            if episode_actions:
                signature = f"{args.model.name}:noisy:{episode:03d}"
                legacy_rows.extend(episode_legacy)
                raw_rows.extend(episode_raw)
                onehot_rows.extend(episode_onehot)
                actions.extend(episode_actions)
                steps.extend(range(len(episode_actions)))
                signatures.extend([signature] * len(episode_actions))
                success_flags.extend([episode_ok] * len(episode_actions))
            episode_results.append(
                {
                    "episode": episode,
                    "executed_actions": len(episode_actions),
                    "steps": steps_taken,
                    "teacher_finished": teacher_finished,
                    "replay_finished": replay_finished,
                    "finished": episode_ok,
                    "teacher_run_time": teacher_time,
                    "replay_run_time": replay_time,
                }
            )
            print(
                f"[episode {episode:03d}] steps={steps_taken} "
                f"teacher_finished={teacher_finished} "
                f"replay_finished={replay_finished} keep={episode_ok}"
            )

        successes = sum(1 for result in episode_results if result["finished"])
        if actions:
            np.savez_compressed(
                args.output_dir / "transitions.npz",
                obs_legacy=np.asarray(legacy_rows, dtype=np.float32),
                obs_raw=np.asarray(raw_rows, dtype=np.float32),
                obs_onehot=np.asarray(onehot_rows, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.int64),
                step=np.asarray(steps, dtype=np.int32),
                route_success=np.asarray(success_flags, dtype=bool),
                split=np.asarray([args.split] * len(actions)),
                route_signature=np.asarray(signatures),
                level=np.asarray([args.level] * len(actions)),
            )
        metadata = {
            "level": args.level,
            "model": str(args.model),
            "teacher_kind": teacher_kind,
            "split": args.split,
            "action_repeat": args.action_repeat,
            "reassert_held_jump_replay": args.reassert_held_jump,
            "reassert_held_jump_teacher": args.teacher_reassert_held_jump,
            "episodes": args.episodes,
            "epsilon": args.epsilon,
            "seed": args.seed,
            "successes": successes,
            "failures": args.episodes - successes,
            "transitions": len(actions),
            "successful_transitions": sum(
                1 for flag in success_flags if flag
            ),
            "episode_results": episode_results,
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(json.dumps({k: metadata[k] for k in (
            "level", "episodes", "epsilon", "successes", "failures",
            "transitions", "successful_transitions",
        )}, indent=2))
        return 0 if successes else 1
    finally:
        teacher_env.close()
        replay_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
