"""Collect, clean, and evaluate full-state 1-1 trajectory checkpoints.

The existing spawn holdout is a cold-cell test: it restores cell + facing and
zeroes velocity, run time, and powerups. This tool records successful
trajectories, samples grounded states along them, and restores the full
player/controller state before evaluating the policy.

Usage:
    .venv/bin/python src/30_trajectory_holdout.py collect \
        --model runs/spawn_1_1/ppo58_1p75_stage2/spawn_best_holdout.zip \
        --obs-encoding legacy
    .venv/bin/python src/30_trajectory_holdout.py clean
    .venv/bin/python src/30_trajectory_holdout.py eval \
        --model runs/spawn_1_1/ppo58_1p75_stage2/spawn_best_holdout.zip \
        --obs-encoding legacy --split holdout

The collector only keeps episodes that finish. The cleaner keeps grounded,
on-floor, no-powerup states and excludes the exit and its immediate approach.
The evaluator restores those states through the optional reset `state` field.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import PPO

from wrappers.godot_env import GodotEnv
import obs_spec

LEVEL = "1-1"
DEFAULT_MODEL = "runs/spawn_1_1/ppo58_1p75_stage2/spawn_best_holdout.zip"
DEFAULT_POOL = "runs/spawn_1_1/pool.json"
DEFAULT_ANALYSIS = "../dragon-jump-remaster/runs/level_analysis/1-1.json"
DEFAULT_OUT = "runs/state_holdout/1-1"
DEFAULT_PORT = 12700


def resolve_env_path() -> str:
    config_path = Path(__file__).resolve().with_name(".config")
    load_dotenv(config_path, override=True)
    env_dir = os.getenv("ENV_PATH", "../environments")
    env_name = os.getenv("ENV_NAME", "DragonJump")
    config_dir = config_path.parent
    repo_dir = config_dir.parent
    if os.path.isabs(env_dir):
        resolved = Path(env_dir)
    else:
        resolved = None
        for base in (Path.cwd(), config_dir, repo_dir):
            candidate = (base / env_dir).resolve()
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            resolved = (Path.cwd() / env_dir).resolve()
    return str(resolved / env_name)


def isolated_home(output_dir: Path, name: str) -> dict[str, str]:
    root = (output_dir / "userdata" / name).resolve()
    root.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["HOME"] = str(root)
    child_env["XDG_DATA_HOME"] = str(root / "xdg_data")
    child_env["XDG_CONFIG_HOME"] = str(root / "xdg_config")
    return child_env


def load_pool(path: Path) -> tuple[list[list[int]], list[list[int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    train = payload.get("train", [])
    holdout = payload.get("holdout", [])
    train_cells = [list(entry["cell"]) for entry in train if "cell" in entry]
    holdout_cells = [list(entry["cell"]) for entry in holdout if "cell" in entry]
    return train_cells, holdout_cells


def load_exit_cell(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    exit_cell = payload.get("exit")
    if not (isinstance(exit_cell, list) and len(exit_cell) == 2):
        raise ValueError(f"{path} has no [x, y] exit field")
    return [int(exit_cell[0]), int(exit_cell[1])]


def state_from_info(info: dict) -> dict:
    position = info.get("global_position", [0.0, 0.0])
    velocity = info.get("velocity", [0.0, 0.0])
    best = info.get("best_run_time", 1e9)
    try:
        best_value = float(best)
    except (TypeError, ValueError):
        best_value = 1e9
    if not math.isfinite(best_value):
        best_value = 1e9
    return {
        "position": [float(position[0]), float(position[1])],
        "velocity": [float(velocity[0]), float(velocity[1])],
        "state_name": str(info.get("state", "Move")),
        "run_time": float(info.get("run_time", 0.0)),
        "best_run_time": best_value,
        "run_active": bool(info.get("run_active", True)),
        "started_walking": bool(info.get("started_walking", True)),
        "cell": [int(info.get("cell", [-1, -1])[0]), int(info.get("cell", [-1, -1])[1])],
        "facing": int(info.get("facing_direction", 1)),
        "on_floor": bool(info.get("on_floor", False)),
        "powerup_count": int(info.get("powerup_count", 0)),
    }


def make_env(args, name: str) -> GodotEnv:
    log_dir = args.output_dir / "godot"
    log_dir.mkdir(parents=True, exist_ok=True)
    return GodotEnv(
        env_path=resolve_env_path(),
        port=args.port,
        show_window=False,
        seed=args.seed,
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        nb_agents=1,
        level=LEVEL,
        convert_action_space=True,
        log_path=str(args.output_dir / "godot" / f"{name}.log"),
        env=isolated_home(args.output_dir, name),
        reward_profile="time",
        obs_encoding=args.obs_encoding,
    )


def choose_action(model: PPO, obs: dict, rng: np.random.Generator, args) -> int:
    values = np.asarray(obs["obs"], dtype=np.float32)
    if args.stochastic:
        action, _ = model.predict(values, deterministic=False)
    else:
        action, _ = model.predict(values, deterministic=True)
    return int(action)


def collect(args) -> int:
    output = args.output_dir / "trajectories"
    output.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(args.model), device="cpu")
    train_cells, holdout_cells = load_pool(args.spawn_pool)
    exit_cell = load_exit_cell(args.level_analysis)
    starts: list[tuple[str, list[int] | None]] = [("canonical", None)]
    starts.extend(("train", cell) for cell in train_cells)
    starts.extend(("holdout", cell) for cell in holdout_cells)
    starts = [
        (source, cell)
        for source, cell in starts
        if cell is None or cell != exit_cell
    ]

    rng = np.random.default_rng(args.seed)
    env = make_env(args, "collect")
    saved = 0
    try:
        for index, (source, cell) in enumerate(starts):
            if cell is None:
                obs, _ = env.reset()
                label = "canonical"
            else:
                obs, _ = env.reset(
                    spawns=[{"agent": 0, "cell": cell, "facing": 1}]
                )
                label = f"{source}_{cell[0]}_{cell[1]}"
            steps: list[dict] = []
            finished = False
            final_time = 0.0
            for step in range(args.max_decisions):
                action = choose_action(model, obs[0], rng, args)
                obs, _rewards, dones, _timeouts, infos = env.step(
                    np.asarray([[action]], dtype=np.int64)
                )
                info = infos[0] if infos else {}
                if info:
                    state = state_from_info(info)
                    steps.append({"step": step, "action": action, "state": state})
                if dones[0]:
                    finished = bool(info.get("finished", False))
                    final_time = float(info.get("run_time", 0.0))
                    break
            print(
                f"[collect {index + 1}/{len(starts)}] {label} "
                f"finished={finished} time={final_time:.3f} steps={len(steps)}",
                flush=True,
            )
            if not finished or not steps:
                continue
            payload = {
                "level": LEVEL,
                "source": source if cell is not None else "canonical",
                "start_cell": cell,
                "finished": True,
                "time": final_time,
                "steps": steps,
            }
            path = output / f"traj_{source}_{label}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            saved += 1
    finally:
        env.close()
    print(f"[collect] saved {saved} successful trajectories to {output}", flush=True)
    return saved


def clean(args) -> int:
    trajectories = sorted((args.output_dir / "trajectories").glob("traj_*.json"))
    if not trajectories:
        raise FileNotFoundError("no trajectories; run collect first")
    exit_cell = load_exit_cell(args.level_analysis)
    train: list[dict] = []
    holdout: list[dict] = []
    for path in trajectories:
        payload = json.loads(path.read_text(encoding="utf-8"))
        steps = payload.get("steps", [])
        path_cells = [
            tuple(step["state"]["cell"])
            for step in steps
            if step.get("state")
        ]
        if not path_cells:
            continue
        source = str(payload.get("source", "train"))
        start_cell = payload.get("start_cell")
        sampled = 0
        for step in steps:
            state = step.get("state", {})
            if not state:
                continue
            cell = state.get("cell", [-1, -1])
            cell_tuple = (int(cell[0]), int(cell[1]))
            if not state.get("on_floor", False):
                continue
            if state.get("powerup_count", 0) != 0:
                continue
            if cell_tuple == tuple(exit_cell):
                continue
            if abs(cell_tuple[0] - exit_cell[0]) <= 2 and abs(
                cell_tuple[1] - exit_cell[1]
            ) <= 2:
                continue
            if state.get("state_name", "") not in ("Idle", "Move"):
                continue
            if float(state.get("run_time", 0.0)) < 0.4:
                continue
            if step.get("step", 0) % args.stride != 0:
                continue
            record = dict(state)
            record.update(
                {
                    "path_source": source,
                    "path_id": path.stem,
                    "path_step": int(step.get("step", 0)),
                    "path_cells": path_cells,
                    "start_cell": start_cell,
                }
            )
            if source == "holdout":
                holdout.append(record)
            else:
                train.append(record)
            sampled += 1
            if sampled >= args.per_path:
                break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "states_train.json").write_text(
        json.dumps(train, indent=2), encoding="utf-8"
    )
    (args.output_dir / "states_holdout.json").write_text(
        json.dumps(holdout, indent=2), encoding="utf-8"
    )
    print(
        f"[clean] train={len(train)} holdout={len(holdout)} "
        f"from {len(trajectories)} trajectories",
        flush=True,
    )
    return len(train) + len(holdout)


def eval_states(args) -> dict:
    states_path = args.output_dir / f"states_{args.split}.json"
    states = json.loads(states_path.read_text(encoding="utf-8"))
    model = PPO.load(str(args.model), device="cpu")
    env = make_env(args, f"eval_{args.split}")
    results: list[dict] = []
    try:
        for index, state in enumerate(states):
            spawn = {
                "agent": 0,
                "cell": state["cell"],
                "facing": state.get("facing", 1),
                "state": state,
            }
            obs, _ = env.reset(spawns=[spawn])
            finished = False
            final_time = 0.0
            steps = 0
            resets = 0
            deviation_steps = 0
            reference = {tuple(cell) for cell in state.get("path_cells", [])}
            for step in range(args.max_decisions):
                action = choose_action(
                    model, obs[0], np.random.default_rng(args.seed + index), args
                )
                obs, _rewards, dones, _timeouts, infos = env.step(
                    np.asarray([[action]], dtype=np.int64)
                )
                steps = step + 1
                info = infos[0] if infos else {}
                cell = tuple(info.get("cell", [-1, -1]))
                if args.force_path:
                    if reference and cell not in reference:
                        deviation_steps += 1
                    else:
                        deviation_steps = 0
                    if deviation_steps >= args.deviation_limit and resets < args.max_resets:
                        resets += 1
                        deviation_steps = 0
                        obs, _ = env.reset(spawns=[spawn])
                        continue
                if dones[0]:
                    finished = bool(info.get("finished", False))
                    final_time = float(info.get("run_time", 0.0))
                    break
            results.append(
                {
                    "index": index,
                    "path_id": state.get("path_id", ""),
                    "path_step": state.get("path_step", 0),
                    "start_cell": state.get("start_cell"),
                    "checkpoint_time": state.get("run_time", 0.0),
                    "finished": finished,
                    "final_time": final_time,
                    "steps": steps,
                    "resets": resets,
                }
            )
            print(
                f"[eval {index + 1}/{len(states)}] "
                f"{state.get('path_id', '')} step={state.get('path_step', 0)} "
                f"finished={finished} time={final_time:.3f} resets={resets}",
                flush=True,
            )
    finally:
        env.close()
    output = args.output_dir / f"eval_{args.split}.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[eval] wrote {output}", flush=True)
    return {
        "count": len(results),
        "finished": sum(1 for item in results if item["finished"]),
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "clean", "eval"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--obs-encoding", default="legacy", choices=["legacy", "multi_hot"])
    parser.add_argument("--spawn-pool", default=DEFAULT_POOL)
    parser.add_argument("--level-analysis", default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--per-path", type=int, default=6)
    parser.add_argument("--split", choices=["train", "holdout"], default="holdout")
    parser.add_argument("--force-path", action="store_true")
    parser.add_argument("--deviation-limit", type=int, default=3)
    parser.add_argument("--max-resets", type=int, default=2)
    args = parser.parse_args()
    args.model = Path(args.model).resolve()
    args.spawn_pool = Path(args.spawn_pool).resolve()
    args.level_analysis = Path(args.level_analysis).resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if not args.spawn_pool.exists():
        raise FileNotFoundError(args.spawn_pool)
    if not args.level_analysis.exists():
        raise FileNotFoundError(args.level_analysis)

    if args.command == "collect":
        return 0 if collect(args) else 1
    if args.command == "clean":
        return 0 if clean(args) else 1
    result = eval_states(args)
    return 0 if result["count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
