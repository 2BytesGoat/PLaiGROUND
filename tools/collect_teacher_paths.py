#!/usr/bin/env python3
"""Collect a canonical 58-input teacher path and its 191-input replay.

For 1-11 .. 1-16 there is no DDQN expert. The archived PPO teacher or the
deployed 58-value student supplies actions. A second ``lean_stick`` process
replays the same actions from the same canonical start so the distillation set
also has the 191 raw and 957 one-hot observations used by the raw-191 and
one-hot candidates.

Both processes use legacy edge-triggered jump semantics for this replay. The
58-input teachers were trained before ``reassert_held_jump`` existed; using the
same semantics on both sides keeps the recorded trajectory aligned.
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
    parser.add_argument("--split", choices=["train", "holdout"], default="train")
    parser.add_argument("--port", type=int, default=32200)
    parser.add_argument("--replay-port", type=int, default=32210)
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
    loaded = load_lean_preprocessor(args.preprocessing)
    preprocessor = (
        loaded
        if isinstance(loaded, LeanStickPreprocessor)
        else LeanStickPreprocessor(loaded)
    )

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
    try:
        teacher_obs, _ = teacher_env.reset()
        replay_obs, _ = replay_env.reset()
        legacy_rows: list[np.ndarray] = []
        raw_rows: list[np.ndarray] = []
        onehot_rows: list[np.ndarray] = []
        actions: list[int] = []
        teacher_done = False
        replay_done = False
        teacher_finished = False
        replay_finished = False
        teacher_time = None
        replay_time = None
        teacher_steps = 0
        replay_steps = 0

        for step in range(args.max_decisions):
            if teacher_done or replay_done:
                break
            legacy = np.asarray(teacher_obs[0]["obs"], dtype=np.float32)
            policy_obs = legacy.reshape(1, -1)
            action_values, _ = teacher.predict(policy_obs, deterministic=True)
            action = int(np.asarray(action_values).reshape(-1)[0])
            raw = np.asarray(replay_obs[0]["obs"], dtype=np.float32)
            onehot = np.asarray(
                preprocessor.transform(raw.reshape(1, -1)), dtype=np.float32
            )[0]
            legacy_rows.append(legacy)
            raw_rows.append(raw)
            onehot_rows.append(onehot)
            actions.append(action)

            teacher_obs, _r, teacher_dones, _t, teacher_infos = teacher_env.step(
                np.asarray([[action]], dtype=np.int64)
            )
            replay_obs, _r2, replay_dones, _t2, replay_infos = replay_env.step(
                np.asarray([[action]], dtype=np.int64)
            )
            teacher_steps = step + 1
            replay_steps = step + 1
            teacher_info = teacher_infos[0] if teacher_infos else {}
            replay_info = replay_infos[0] if replay_infos else {}
            if bool(teacher_dones[0]):
                teacher_done = True
                teacher_finished = bool(teacher_info.get("finished", False))
                if teacher_finished:
                    teacher_time = teacher_info.get("run_time")
            if bool(replay_dones[0]):
                replay_done = True
                replay_finished = bool(replay_info.get("finished", False))
                if replay_finished:
                    replay_time = replay_info.get("run_time")

        np.savez_compressed(
            args.output_dir / "transitions.npz",
            obs_legacy=np.asarray(legacy_rows, dtype=np.float32),
            obs_raw=np.asarray(raw_rows, dtype=np.float32),
            obs_onehot=np.asarray(onehot_rows, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            step=np.arange(len(actions), dtype=np.int32),
            route_success=np.asarray(
                [bool(teacher_finished and replay_finished)] * len(actions),
                dtype=bool,
            ),
            split=np.asarray([args.split] * len(actions)),
            route_signature=np.asarray(
                [f"{args.model.name}:canonical"] * len(actions)
            ),
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
            "teacher": {
                "finished": teacher_finished,
                "run_time": teacher_time,
                "steps": teacher_steps,
            },
            "replay": {
                "finished": replay_finished,
                "run_time": replay_time,
                "steps": replay_steps,
            },
            "transitions": len(actions),
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2))
        return 0
    finally:
        teacher_env.close()
        replay_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
