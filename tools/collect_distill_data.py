#!/usr/bin/env python3
"""Collect teacher rollouts for the campaign distillation pass.

The collector runs one Godot process per level with one agent per start state.
For a 191 ``lean_stick`` teacher (the selected DDQN experts) it records both
the raw 191 observation and the fitted 957 one-hot input, plus the teacher
action. For a 58 ``legacy`` teacher (the archived PPO teachers or the deployed
student) it records the 58 observation and action; a separate paired replay
writes the matching 191/957 observations.

Example:
    .venv/bin/python tools/collect_distill_data.py expert \
      --level 1-6 \
      --model runs/ddqn_1_6_start_repair/dqn/checkpoints/ddqn_1814952_steps.zip \
      --pool runs/neat_speed_1_6_from_0090/pool_state_70/train_start_pool.json \
      --pool runs/neat_speed_1_6_from_0090/pool_state_70/holdout_start_pool.json \
      --preprocessing runs/ddqn_fresh_1_1_lean_stick/preprocessing.pkl \
      --output-dir runs/distill_campaign_student/raw/1-6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO

from plaiground.envs.godot import GodotEnv
from plaiground.envs.runtime import isolated_home, resolve_env_path
from plaiground.envs.single_agent import RESET_STATE_KEYS
from plaiground.observations.lean_io import load_lean_preprocessor
from plaiground.observations.lean_stick import LeanStickPreprocessor


class JSONPolicy:
    """Minimal forward pass for the deployed flat JSON MLP."""

    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        weights = data["weights"]
        hidden = [int(value) for value in data["hidden_sizes"]]
        if len(hidden) != 2:
            raise ValueError(f"{path}: expected two hidden layers, got {hidden}")
        self.input_dim = int(data["input_size"])
        self.hidden = hidden
        self.w1 = np.asarray(weights["w1"], dtype=np.float32).reshape(
            hidden[0], self.input_dim
        )
        self.b1 = np.asarray(weights["b1"], dtype=np.float32).reshape(hidden[0])
        self.w2 = np.asarray(weights["w2"], dtype=np.float32).reshape(
            hidden[1], hidden[0]
        )
        self.b2 = np.asarray(weights["b2"], dtype=np.float32).reshape(hidden[1])
        self.w3 = np.asarray(weights["w3"], dtype=np.float32).reshape(
            2, hidden[1]
        )
        self.b3 = np.asarray(weights["b3"], dtype=np.float32).reshape(2)
        self.action_repeat = int(data.get("action_repeat", 4))

    def predict(self, observations, deterministic: bool = True):
        values = np.asarray(observations, dtype=np.float32)
        single = values.ndim == 1
        batch = values.reshape(1, -1) if single else values
        if batch.shape[1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} inputs, got {batch.shape[1]}"
            )
        h1 = np.maximum(batch @ self.w1.T + self.b1, 0.0)
        h2 = np.maximum(h1 @ self.w2.T + self.b2, 0.0)
        logits = h2 @ self.w3.T + self.b3
        actions = np.argmax(logits, axis=1).astype(np.int64)
        return (actions[0] if single else actions), None


def load_pool(paths: list[Path], level: str) -> list[dict]:
    starts: list[dict] = []
    seen: set[tuple] = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("starts", data) if isinstance(data, dict) else data
        default_split = "holdout" if "holdout" in path.name else "train"
        for entry in entries:
            if entry.get("level") not in (None, level):
                continue
            item = dict(entry)
            item.setdefault("split", default_split)
            if item.get("canonical"):
                key = ("canonical",)
            else:
                key = (
                    "state",
                    tuple(int(v) for v in item.get("cell", [])),
                    int(item.get("facing", 1)),
                    round(float(item.get("run_time_at_state", item.get("run_time", 0.0))), 6),
                    str(item.get("source_route", "")),
                )
            if key in seen:
                continue
            seen.add(key)
            starts.append(item)
    if not starts:
        raise SystemExit(f"no starts found in {', '.join(str(p) for p in paths)}")
    return starts


def spawns_for(starts: list[dict]) -> list[dict]:
    spawns: list[dict] = []
    for index, entry in enumerate(starts):
        if entry.get("canonical"):
            continue
        spawn = {
            "agent": index,
            "cell": [int(entry["cell"][0]), int(entry["cell"][1])],
            "facing": 1 if int(entry.get("facing", 1)) >= 0 else -1,
        }
        for key in RESET_STATE_KEYS:
            if key in entry:
                spawn[key] = entry[key]
        spawns.append(spawn)
    return spawns


def start_key(entry: dict) -> str:
    if entry.get("canonical"):
        return "canonical"
    return (
        f"{entry.get('cell')}|{entry.get('facing')}|"
        f"{entry.get('run_time_at_state', entry.get('run_time'))}|"
        f"{entry.get('source_route', '')}"
    )


def model_predict(model, observations: np.ndarray, kind: str) -> np.ndarray:
    if kind == "json":
        actions, _ = model.predict(observations, deterministic=True)
        return np.asarray(actions, dtype=np.int64).reshape(-1)
    actions, _ = model.predict(observations, deterministic=True)
    return np.asarray(actions, dtype=np.int64).reshape(-1)


def collect(
    *,
    level: str,
    model_path: Path,
    model_kind: str,
    starts: list[dict],
    output_dir: Path,
    preprocessing_path: Path,
    max_decisions: int,
    action_repeat: int,
    speedup: float,
    port: int,
    timeout: int,
    show: bool,
    reassert_held_jump: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor: LeanStickPreprocessor | None = None
    if model_kind == "dqn":
        loaded = load_lean_preprocessor(preprocessing_path)
        preprocessor = (
            loaded
            if isinstance(loaded, LeanStickPreprocessor)
            else LeanStickPreprocessor(loaded)
        )
        model = DQN.load(str(model_path), device="cpu")
        if int(model.observation_space.shape[0]) != preprocessor.output_size:
            raise SystemExit(
                f"{model_path}: model expects {model.observation_space.shape[0]}, "
                f"preprocessor emits {preprocessor.output_size}"
            )
    elif model_kind == "ppo":
        model = PPO.load(str(model_path), device="cpu")
    elif model_kind == "json":
        model = JSONPolicy(model_path)
    else:
        raise SystemExit(f"unknown model kind: {model_kind}")

    env_path = resolve_env_path()
    GodotEnv.DEFAULT_TIMEOUT = timeout
    log_path = output_dir / "godot.log"
    env = GodotEnv(
        env_path=env_path,
        port=port,
        show_window=show,
        seed=42,
        action_repeat=action_repeat,
        speedup=speedup,
        nb_agents=len(starts),
        level=level,
        convert_action_space=True,
        log_path=str(log_path),
        env=isolated_home(output_dir, "home"),
        reward_profile="default",
        obs_encoding="lean_stick" if model_kind == "dqn" else "legacy",
        reassert_held_jump="true" if reassert_held_jump else "false",
    )
    try:
        spawns = spawns_for(starts)
        obs, infos = env.reset(spawns=spawns) if spawns else env.reset()
        count = len(starts)
        done = np.zeros(count, dtype=bool)
        finished = np.zeros(count, dtype=bool)
        timed_out = np.zeros(count, dtype=bool)
        deaths = np.zeros(count, dtype=np.int32)
        steps = np.zeros(count, dtype=np.int32)
        run_times = np.full(count, np.nan, dtype=np.float64)
        spawn_ok = np.ones(count, dtype=bool)

        for index, info in enumerate(infos[:count]):
            expected = starts[index].get("cell")
            if expected is None:
                continue
            actual = info.get("spawn_cell")
            spawn_ok[index] = bool(
                isinstance(actual, (list, tuple))
                and len(actual) == 2
                and [int(actual[0]), int(actual[1])]
                == [int(expected[0]), int(expected[1])]
            )

        raw_by_agent: list[list[np.ndarray]] = [[] for _ in range(count)]
        onehot_by_agent: list[list[np.ndarray]] = [[] for _ in range(count)]
        legacy_by_agent: list[list[np.ndarray]] = [[] for _ in range(count)]
        actions_by_agent: list[list[int]] = [[] for _ in range(count)]
        last_infos: list[dict] = [dict(info) for info in infos[:count]]

        for step in range(max_decisions):
            active = np.flatnonzero(~done)
            if active.size == 0:
                break
            raw = np.asarray(
                [obs[index]["obs"] for index in active], dtype=np.float32
            )
            if model_kind == "dqn":
                if preprocessor is None:
                    raise AssertionError("missing preprocessor")
                policy_obs = np.asarray(
                    preprocessor.transform(raw), dtype=np.float32
                )
            else:
                policy_obs = raw

            actions_active = model_predict(model, policy_obs, model_kind)
            for offset, index in enumerate(active):
                if model_kind == "dqn":
                    raw_by_agent[index].append(raw[offset].astype(np.float32))
                    onehot_by_agent[index].append(
                        policy_obs[offset].astype(np.float32)
                    )
                else:
                    legacy_by_agent[index].append(
                        raw[offset].astype(np.float32)
                    )
                actions_by_agent[index].append(int(actions_active[offset]))

            actions = np.zeros((count, 1), dtype=np.int64)
            actions[active, 0] = actions_active
            obs, _rewards, dones, _timeouts, step_infos = env.step(actions)
            for index in active:
                info = dict(step_infos[index]) if index < len(step_infos) else {}
                last_infos[index] = info
                if bool(dones[index]):
                    done[index] = True
                    finished[index] = bool(info.get("finished", False))
                    steps[index] = step + 1
                    if finished[index]:
                        run_time = info.get("run_time")
                        if isinstance(run_time, (int, float)) and np.isfinite(run_time):
                            run_times[index] = float(run_time)
                    else:
                        timed_out[index] = step + 1 >= max_decisions
                if "deaths" in info:
                    try:
                        deaths[index] = int(info["deaths"])
                    except (TypeError, ValueError):
                        pass
            if done.all():
                break

        for index in np.flatnonzero(~done):
            steps[index] = max_decisions
            timed_out[index] = True

        rows: list[dict] = []
        raw_flat: list[np.ndarray] = []
        onehot_flat: list[np.ndarray] = []
        legacy_flat: list[np.ndarray] = []
        action_flat: list[int] = []
        route_index_flat: list[int] = []
        step_flat: list[int] = []
        success_flat: list[bool] = []
        split_flat: list[str] = []
        signature_flat: list[str] = []

        for index, entry in enumerate(starts):
            ok = bool(finished[index])
            raw = raw_by_agent[index]
            onehot = onehot_by_agent[index]
            legacy = legacy_by_agent[index]
            actions = actions_by_agent[index]
            for step_index, action in enumerate(actions):
                if model_kind == "dqn":
                    raw_flat.append(raw[step_index])
                    onehot_flat.append(onehot[step_index])
                else:
                    legacy_flat.append(legacy[step_index])
                action_flat.append(action)
                route_index_flat.append(index)
                step_flat.append(step_index)
                success_flat.append(ok)
                split_flat.append(str(entry.get("split", "train")))
                signature_flat.append(str(entry.get("source_route", "")))
            rows.append(
                {
                    "index": index,
                    "level": level,
                    "split": entry.get("split", "train"),
                    "kind": entry.get("kind", "canonical" if entry.get("canonical") else "state"),
                    "start_cell": entry.get("cell"),
                    "start_facing": entry.get("facing"),
                    "source_route": entry.get("source_route", ""),
                    "source_time": entry.get("source_time"),
                    "run_time_at_state": entry.get("run_time_at_state", entry.get("run_time")),
                    "cluster_id": entry.get("cluster_id", ""),
                    "spawn_ok": bool(spawn_ok[index]),
                    "finished": ok,
                    "timed_out": bool(timed_out[index]),
                    "deaths": int(deaths[index]),
                    "steps": int(steps[index]),
                    "run_time": float(run_times[index]) if np.isfinite(run_times[index]) else None,
                    "transitions": len(actions),
                }
            )

        arrays: dict[str, np.ndarray] = {
            "actions": np.asarray(action_flat, dtype=np.int64),
            "route_index": np.asarray(route_index_flat, dtype=np.int32),
            "step": np.asarray(step_flat, dtype=np.int32),
            "route_success": np.asarray(success_flat, dtype=bool),
            "split": np.asarray(split_flat),
            "route_signature": np.asarray(signature_flat),
            "level": np.asarray([level] * len(action_flat)),
        }
        if model_kind == "dqn":
            arrays["obs_raw"] = np.asarray(raw_flat, dtype=np.float32)
            arrays["obs_onehot"] = np.asarray(onehot_flat, dtype=np.float32)
        else:
            arrays["obs_legacy"] = np.asarray(legacy_flat, dtype=np.float32)
        np.savez_compressed(output_dir / "transitions.npz", **arrays)
        metadata = {
            "level": level,
            "model": str(model_path),
            "model_kind": model_kind,
            "action_repeat": action_repeat,
            "max_decisions": max_decisions,
            "reassert_held_jump": reassert_held_jump,
            "starts": rows,
            "summary": {
                "starts": len(rows),
                "finished": int(finished.sum()),
                "deaths": int((~finished & ~timed_out).sum()),
                "timeouts": int(timed_out.sum()),
                "spawn_failures": int((~spawn_ok).sum()),
                "transitions": len(action_flat),
                "best_time": float(np.nanmin(run_times)) if np.isfinite(run_times).any() else None,
                "median_time": float(np.nanmedian(run_times)) if np.isfinite(run_times).any() else None,
            },
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return metadata
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["expert", "teacher"])
    parser.add_argument("--level", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pool", type=Path, action="append", required=True)
    parser.add_argument("--preprocessing", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--speedup", type=float, default=12.0)
    parser.add_argument("--port", type=int, default=32000)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--no-reassert-held-jump",
        action="store_true",
        help="Use legacy edge-triggered jump semantics (for 58-input PPO replay).",
    )
    args = parser.parse_args()
    if args.kind == "expert":
        args.model_kind = "dqn"
    else:
        suffix = args.model.suffix.lower()
        args.model_kind = "json" if suffix == ".json" else "ppo"
    if args.model_kind == "dqn" and args.preprocessing is None:
        parser.error("--preprocessing is required for a DDQN expert")
    return args


def main() -> int:
    args = parse_args()
    starts = load_pool(args.pool, args.level)
    metadata = collect(
        level=args.level,
        model_path=args.model,
        model_kind=args.model_kind,
        starts=starts,
        output_dir=args.output_dir,
        preprocessing_path=args.preprocessing or Path("."),
        max_decisions=args.max_decisions,
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        port=args.port,
        timeout=args.timeout,
        show=args.show,
        reassert_held_jump=not args.no_reassert_held_jump,
    )
    print(json.dumps(metadata["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
