#!/usr/bin/env python3
"""Trace deterministic PPO rollouts from saved checkpoints.

This is a diagnostic tool, not a training or promotion path. It records the
per-step action, cell, position, velocity, and run time so two checkpoints can
be compared at the first divergence or checked for a repeated stuck loop.

Examples:
    .venv/bin/python tools/trace_ppo.py \
        --model runs/.../checkpoints/ppo-400008.zip \
        --preprocessor runs/.../preprocessing.pkl

    .venv/bin/python tools/trace_ppo.py \
        --model runs/.../checkpoints/ppo-575008.zip \
        --preprocessor runs/.../preprocessing.pkl \
        --force-step 5 --force-action 0
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from stable_baselines3 import PPO

from plaiground.envs.runtime import isolated_home, resolve_env_path
from plaiground.envs.godot import GodotEnv
from plaiground.observations.lean_io import load_lean_preprocessor

REPO = Path(__file__).resolve().parents[1]


def _fmt(value: Any) -> str:
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.4f}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_fmt(item) for item in value) + "]"
    return str(value)


def _trace(
    model_path: Path,
    preprocessor_path: Path,
    *,
    seed: int,
    stochastic: bool,
    port: int,
    cell: list[int] | None,
    facing: int,
    max_decisions: int,
    force_step: int | None,
    force_action: int | None,
    output: Path | None,
) -> list[dict[str, Any]]:
    preprocessor = load_lean_preprocessor(preprocessor_path)
    model = PPO.load(str(model_path), device="cpu")
    out_root = (
        output.expanduser().resolve()
        if output is not None
        else Path(tempfile.mkdtemp(prefix="ppo_trace_"))
    )
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "godot.log"
    env = GodotEnv(
        env_path=resolve_env_path(),
        port=port,
        show_window=False,
        seed=seed,
        action_repeat=4,
        speedup=8.0,
        nb_agents=1,
        level="1-1",
        convert_action_space=True,
        log_path=str(log_path),
        env=isolated_home(out_root, "trace"),
        reward_profile="default",
        obs_encoding="lean",
    )
    rows: list[dict[str, Any]] = []
    try:
        spawns = None
        if cell is not None:
            spawns = [{"agent": 0, "cell": list(cell), "facing": facing}]
        obs, infos = env.reset(spawns=spawns) if spawns else env.reset()
        reset_info = dict(infos[0]) if infos else {}
        finished = False
        timed_out = False
        for step in range(max_decisions):
            raw = np.asarray([item["obs"] for item in obs], dtype=np.float32)
            policy_obs = np.asarray(
                preprocessor.transform(raw), dtype=np.float32
            )
            natural_action, _ = model.predict(
                policy_obs,
                deterministic=not stochastic,
            )
            natural_action = int(np.asarray(natural_action).reshape(-1)[0])
            action = natural_action
            forced = force_step is not None and step == force_step
            if forced:
                action = int(force_action)
            obs, rewards, dones, _timeouts, infos = env.step(
                np.asarray([[action]], dtype=np.int64)
            )
            info = dict(infos[0]) if infos else {}
            rows.append(
                {
                    "step": step,
                    "action": action,
                    "natural_action": natural_action,
                    "forced": forced,
                    "reward": float(rewards[0]),
                    "done": bool(dones[0]),
                    "cell": info.get("cell"),
                    "run_time": info.get("run_time"),
                    "state": info.get("state"),
                    "on_floor": info.get("on_floor"),
                    "position": info.get("global_position"),
                    "velocity": info.get("velocity"),
                    "spawn_cell": info.get("spawn_cell"),
                    "last_reset_spawns": env.last_reset_spawns,
                }
            )
            if bool(dones[0]):
                finished = bool(rewards[0] > 0.0 or info.get("finished"))
                break
        else:
            timed_out = True

        print(
            f"\nMODEL {model_path.name} seed={seed} stochastic={stochastic} "
            f"spawn={cell} finished={finished} timeout={timed_out} "
            f"steps={len(rows)} reset={_fmt(reset_info.get('cell'))}"
        )
        print("step act cell run_time state floor position velocity done")
        for row in rows:
            print(
                f"{row['step']:3d} {row['action']} {_fmt(row['cell']):10} "
                f"{_fmt(row['run_time']):9} {str(row['state']):10} "
                f"{row['on_floor']} {_fmt(row['position']):20} "
                f"{_fmt(row['velocity']):16} {row['done']}"
            )
        trace_path = out_root / f"trace-{model_path.stem}.json"
        trace_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"trace={trace_path}")
        print(f"godot_log={log_path}")
        return rows
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        required=True,
        help="Checkpoint to trace. Repeat for multiple checkpoints.",
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        required=True,
        help="Fitted lean preprocessor pickle.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--port", type=int, default=13400)
    parser.add_argument("--cell", type=int, nargs=2)
    parser.add_argument("--facing", type=int, default=1)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--force-step", type=int, default=None)
    parser.add_argument("--force-action", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for traces. Defaults to a temp directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for index, model in enumerate(args.model):
        _trace(
            model.expanduser().resolve(),
            args.preprocessor.expanduser().resolve(),
            seed=args.seed,
            stochastic=args.stochastic,
            port=args.port + index,
            cell=args.cell,
            facing=args.facing,
            max_decisions=args.max_decisions,
            force_step=args.force_step,
            force_action=args.force_action,
            output=args.output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
