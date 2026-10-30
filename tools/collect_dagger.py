#!/usr/bin/env python3
"""Collect DAgger-style corrections around a distilled student.

The student drives a ``lean_stick`` 191 environment. The teacher labels the
state the student actually reaches:

* DDQN expert levels: the teacher sees the same 191 state through the fitted
  957 one-hot preprocessor, so one environment is enough.
* 1-11 .. 1-16: the teacher is a 58-input PPO/JSON policy, so a second 58-input
  environment replays the student's actions and supplies the label.

Exploration noise is applied to the student action so several agents starting
from the same canonical state visit different states. Every visited state is
kept, including states on runs that eventually die; the teacher action is the
training label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN, PPO

from plaiground.envs.godot import GodotEnv
from plaiground.envs.runtime import isolated_home, resolve_env_path
from plaiground.models.student import StudentMLP
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


def load_preprocessor(path: Path) -> LeanStickPreprocessor:
    loaded = load_lean_preprocessor(path)
    return loaded if isinstance(loaded, LeanStickPreprocessor) else LeanStickPreprocessor(loaded)


def make_env(
    level: str,
    encoding: str,
    num_agents: int,
    port: int,
    output_dir: Path,
    tag: str,
    *,
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
        log_path=str(output_dir / f"{tag}.log"),
        env=isolated_home(output_dir, tag),
        reward_profile="default",
        obs_encoding=encoding,
        reassert_held_jump="true" if reassert_held_jump else "false",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--teacher-kind", choices=["dqn", "ppo", "json"], required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--split", choices=["train", "holdout"], default="train")
    parser.add_argument("--port", type=int, default=34800)
    parser.add_argument("--teacher-port", type=int, default=34810)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--speedup", type=float, default=12.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-reassert-held-jump", action="store_true")
    parser.add_argument("--teacher-reassert-held-jump", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    np.random.seed(0)
    checkpoint = torch.load(args.student, map_location="cpu", weights_only=False)
    student = StudentMLP(
        int(checkpoint["input_dim"]), [int(v) for v in checkpoint["hidden"]]
    )
    student.load_state_dict(checkpoint["state_dict"])
    student.eval()
    student_input_kind = str(checkpoint.get("input_kind", "raw191"))
    preprocessor = load_preprocessor(args.preprocessing)

    if args.teacher_kind == "dqn":
        teacher = DQN.load(str(args.teacher), device="cpu")
        if int(teacher.observation_space.shape[0]) != 957:
            raise SystemExit("DDQN teacher must expect 957 inputs")
        teacher_env = None
    elif args.teacher_kind == "ppo":
        teacher = PPO.load(str(args.teacher), device="cpu")
        if int(teacher.observation_space.shape[0]) != 58:
            raise SystemExit("PPO teacher must expect 58 inputs")
        teacher_env = make_env(
            args.level,
            "legacy",
            args.agents,
            args.teacher_port,
            args.output_dir,
            "teacher",
            action_repeat=args.action_repeat,
            speedup=args.speedup,
            timeout=args.timeout,
            show=args.show,
            reassert_held_jump=args.teacher_reassert_held_jump,
        )
    else:
        teacher = JSONPolicy(args.teacher)
        teacher_env = make_env(
            args.level,
            "legacy",
            args.agents,
            args.teacher_port,
            args.output_dir,
            "teacher",
            action_repeat=args.action_repeat,
            speedup=args.speedup,
            timeout=args.timeout,
            show=args.show,
            reassert_held_jump=args.teacher_reassert_held_jump,
        )

    student_env = make_env(
        args.level,
        "lean_stick",
        args.agents,
        args.port,
        args.output_dir,
        "student",
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        timeout=args.timeout,
        show=args.show,
        reassert_held_jump=not args.no_reassert_held_jump,
    )
    try:
        student_obs, _ = student_env.reset()
        teacher_obs = None
        if teacher_env is not None:
            teacher_obs, _ = teacher_env.reset()
        count = args.agents
        done = np.zeros(count, dtype=bool)
        student_finished = np.zeros(count, dtype=bool)
        rng = np.random.default_rng(0)

        raw_rows: list[np.ndarray] = []
        onehot_rows: list[np.ndarray] = []
        teacher_actions: list[int] = []
        student_actions: list[int] = []
        route_index: list[int] = []
        step_rows: list[int] = []

        for step in range(args.max_decisions):
            active = np.flatnonzero(~done)
            if active.size == 0:
                break
            raw = np.asarray(
                [student_obs[index]["obs"] for index in active], dtype=np.float32
            )
            onehot = np.asarray(preprocessor.transform(raw), dtype=np.float32)
            student_batch = onehot if student_input_kind == "onehot957" else raw
            with torch.no_grad():
                logits = student(torch.as_tensor(student_batch, dtype=torch.float32))
                greedy = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
            noisy = greedy.copy()
            flips = rng.random(len(noisy)) < args.epsilon
            noisy[flips] = 1 - noisy[flips]

            if teacher_env is not None:
                teacher_raw = np.asarray(
                    [teacher_obs[index]["obs"] for index in active],
                    dtype=np.float32,
                )
                teacher_batch = teacher_raw
            else:
                teacher_batch = onehot
            if args.teacher_kind == "ppo":
                labels, _ = teacher.predict(teacher_batch, deterministic=True)
                labels = np.asarray(labels, dtype=np.int64).reshape(-1)
            else:
                labels, _ = teacher.predict(teacher_batch, deterministic=True)
                labels = np.asarray(labels, dtype=np.int64).reshape(-1)

            actions = np.zeros((count, 1), dtype=np.int64)
            actions[active, 0] = noisy
            student_obs, _r, student_dones, _t, student_infos = student_env.step(actions)
            if teacher_env is not None:
                teacher_obs, _r2, teacher_dones, _t2, _teacher_infos = teacher_env.step(
                    actions
                )

            for offset, index in enumerate(active):
                raw_rows.append(raw[offset])
                onehot_rows.append(onehot[offset])
                teacher_actions.append(int(labels[offset]))
                student_actions.append(int(noisy[offset]))
                route_index.append(index)
                step_rows.append(step)
                if bool(student_dones[index]):
                    done[index] = True
                    student_finished[index] = bool(
                        student_infos[index].get("finished", False)
                    )
                if teacher_env is not None and bool(teacher_dones[index]):
                    done[index] = True

        np.savez_compressed(
            args.output_dir / "transitions.npz",
            obs_raw=np.asarray(raw_rows, dtype=np.float32),
            obs_onehot=np.asarray(onehot_rows, dtype=np.float32),
            actions=np.asarray(teacher_actions, dtype=np.int64),
            student_actions=np.asarray(student_actions, dtype=np.int64),
            route_index=np.asarray(route_index, dtype=np.int32),
            step=np.asarray(step_rows, dtype=np.int32),
            route_success=np.ones(len(teacher_actions), dtype=bool),
            split=np.asarray([args.split] * len(teacher_actions)),
            route_signature=np.asarray(
                [f"dagger:{args.teacher.name}:{index}" for index in route_index]
            ),
            level=np.asarray([args.level] * len(teacher_actions)),
        )
        metadata = {
            "level": args.level,
            "student": str(args.student),
            "teacher": str(args.teacher),
            "teacher_kind": args.teacher_kind,
            "agents": args.agents,
            "epsilon": args.epsilon,
            "split": args.split,
            "transitions": len(teacher_actions),
            "student_finished": int(student_finished.sum()),
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2))
        return 0
    finally:
        student_env.close()
        if teacher_env is not None:
            teacher_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
