#!/usr/bin/env python3
"""Evaluate a 754-value PPO model on canonical/medium/hard cell starts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import PPO

REPO = Path(__file__).resolve().parents[3]
from plaiground.envs.godot import GodotEnv

BASE = REPO / "runs" / "neat_1_1_8x24_archive"
DEFAULT_MODEL = REPO / "runs" / "ppo_754_ft_512" / "best_model.zip"
DEFAULT_STARTS = BASE / "strata" / "start_strata.json"
DEFAULT_OUTPUT = REPO / "runs" / "ppo_754_ft_512" / "strata_eval"


def resolve_env_path() -> str:
    config_path = REPO / "src" / ".config"
    load_dotenv(config_path, override=True)
    env_dir = os.getenv("ENV_PATH", "./environments")
    env_name = os.getenv("ENV_NAME", "DragonJump")
    if os.path.isabs(env_dir):
        resolved = Path(env_dir)
    else:
        resolved = None
        for base in (Path.cwd(), config_path.parent, REPO):
            candidate = (base / env_dir).resolve()
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            resolved = (Path.cwd() / env_dir).resolve()
    return str(resolved / env_name)


def isolated_home(output_dir: Path) -> dict[str, str]:
    root = (output_dir / "userdata").resolve()
    root.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["HOME"] = str(root)
    child_env["XDG_DATA_HOME"] = str(root / "xdg_data")
    child_env["XDG_CONFIG_HOME"] = str(root / "xdg_config")
    return child_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--starts", type=Path, default=DEFAULT_STARTS)
    parser.add_argument("--level", default="1-1")
    parser.add_argument(
        "--obs-encoding",
        default="multi_hot",
        choices=("legacy", "multi_hot", "multi_hot_sensor", "raw"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strata",
        default="canonical,medium,hard,focus",
        help="Comma-separated strata to evaluate.",
    )
    parser.add_argument("--port", type=int, default=16700)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument(
        "--zero-sensor",
        action="store_true",
        help="Ablation: zero the 40 appended multi_hot_sensor values at evaluation.",
    )
    parser.add_argument(
        "--start-jump",
        dest="start_jump",
        action="store_true",
        help="Compatibility: press jump once after reset for old checkpoints.",
    )
    parser.add_argument(
        "--no-start-jump",
        dest="start_jump",
        action="store_false",
        help="Leave the first action policy-owned (default; training reset "
        "force-starts the run).",
    )
    parser.set_defaults(start_jump=False)
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.starts = args.starts.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_strata = [
        item.strip() for item in args.strata.split(",") if item.strip()
    ]
    starts = [
        start
        for start in json.loads(args.starts.read_text(encoding="utf-8"))
        if start.get("stratum") in selected_strata
    ]
    if not starts:
        raise RuntimeError("no requested starts selected")
    print(
        f"evaluating {len(starts)} starts ({','.join(selected_strata)}) "
        f"from {args.starts}",
        flush=True,
    )

    model = PPO.load(str(args.model), device="cpu")
    expected = int(np.prod(model.observation_space.shape))
    log_dir = args.output_dir / "godot"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = GodotEnv(
        env_path=resolve_env_path(),
        port=args.port,
        show_window=False,
        seed=42,
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        nb_agents=len(starts),
        level=args.level,
        convert_action_space=True,
        log_path=str(log_dir / "eval.log"),
        env=isolated_home(args.output_dir),
        reward_profile="time",
        obs_encoding=args.obs_encoding,
    )
    try:
        def mask_sensor(items) -> None:
            if not args.zero_sensor:
                return
            for item in items:
                values = np.asarray(item["obs"], dtype=np.float32).copy()
                if values.size >= 40:
                    values[-40:] = 0.0
                item["obs"] = values

        spawns = [
            {
                "agent": index,
                "cell": list(start["cell"]),
                "facing": int(start["facing"]),
            }
            for index, start in enumerate(starts)
        ]
        obs, _infos = env.reset(spawns=spawns)
        count = len(starts)
        if len(obs[0]["obs"]) != expected:
            raise RuntimeError(
                f"model expects {expected} observations, game returned {len(obs[0]['obs'])}"
            )
        done = np.zeros(count, dtype=bool)
        finished = np.zeros(count, dtype=bool)
        finish_times = np.full(count, np.inf, dtype=np.float64)
        steps = np.zeros(count, dtype=np.int32)
        actual_spawns: list[list[int] | None] = [None] * count

        if args.start_jump:
            # Match the training reset contract: one jump starts the run,
            # then the policy owns every subsequent decision. The jump can
            # legitimately finish an exit-adjacent spawn; record that terminal
            # state instead of feeding the agent again and calling it a timeout.
            obs, start_rewards, start_dones, _timeouts, start_infos = env.step(
                np.ones((count, 1), dtype=np.int64)
            )
            for index in range(count):
                info = start_infos[index] if index < len(start_infos) else {}
                spawn = info.get("spawn_cell", starts[index]["cell"])
                if isinstance(spawn, (list, tuple)) and len(spawn) >= 2:
                    actual_spawns[index] = [int(spawn[0]), int(spawn[1])]
                steps[index] = 1
                if bool(start_dones[index]):
                    done[index] = True
                    finished[index] = bool(info.get("finished", False))
                    if finished[index]:
                        finish_times[index] = float(
                            info.get("run_time", np.inf)
                        )
        mask_sensor(obs)

        for step in range(args.max_steps):
            if done.all():
                break
            batch = np.stack(
                [np.asarray(obs[index]["obs"], dtype=np.float32) for index in range(count)]
            )
            actions, _ = model.predict(batch, deterministic=True)
            actions = np.asarray(actions, dtype=np.int64).reshape(count, 1)
            actions[done] = 0
            obs, _rewards, dones, _timeouts, infos = env.step(actions)
            mask_sensor(obs)
            for index in range(count):
                if done[index]:
                    continue
                info = infos[index] if index < len(infos) else {}
                if actual_spawns[index] is None:
                    spawn = info.get("spawn_cell", starts[index]["cell"])
                    if isinstance(spawn, (list, tuple)) and len(spawn) >= 2:
                        actual_spawns[index] = [int(spawn[0]), int(spawn[1])]
                steps[index] = step + 1
                if bool(dones[index]):
                    done[index] = True
                    finished[index] = bool(info.get("finished", False))
                    if finished[index]:
                        finish_times[index] = float(info.get("run_time", np.inf))
            if done.all():
                break
    finally:
        env.close()

    rows = []
    for index, start in enumerate(starts):
        rows.append(
            {
                **start,
                "actual_spawn": actual_spawns[index],
                "spawn_ok": actual_spawns[index] == start["cell"],
                "finished": bool(finished[index]),
                "run_time": (
                    float(finish_times[index])
                    if finished[index] and np.isfinite(finish_times[index])
                    else None
                ),
                "steps": int(steps[index]),
            }
        )

    (args.output_dir / "results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    summary = {}
    for stratum in selected_strata:
        subset = [row for row in rows if row["stratum"] == stratum]
        if not subset:
            continue
        times = [
            row["run_time"]
            for row in subset
            if row["finished"] and row["run_time"] is not None
        ]
        summary[stratum] = {
            "starts": len(subset),
            "finished": len(times),
            "spawn_ok": sum(1 for row in subset if row["spawn_ok"]),
            "best": min(times) if times else None,
            "median": (
                sorted(times)[len(times) // 2] if times else None
            ),
            "worst": max(times) if times else None,
            "failures": [
                {
                    "cell": row["cell"],
                    "facing": row["facing"],
                    "run_time": row["run_time"],
                }
                for row in subset
                if not row["finished"]
            ],
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
