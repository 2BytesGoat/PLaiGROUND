#!/usr/bin/env python3
"""DQN on NEAT route-spawn checkpoints for one level.

NEAT routes are used only to find clean, grounded spawn cells. The DQN
collects its own transitions from those starts; NEAT actions are not used as
imitation labels.

The first experiment is 1-1 only. See docs/experiments/dqn-route-spawns.md
and docs/audit/2026-10-28-dqn-pipeline-review.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from wrappers.godot_env import GodotEnv

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROUTES = REPO / "runs/neat_1_1_8x24_archive/clean/clean_routes.json"
DEFAULT_FINISHERS = REPO / "runs/neat_1_1_8x24_archive/archive/finishers.jsonl"
DEFAULT_ANALYSIS = Path(
    "/Users/giani.statie/personal/dragon-jump-remaster/runs/level_analysis/1-1.json"
)
DEFAULT_RUN = REPO / "runs/dqn_route_spawns_1_1"


def load_ppo_module():
    path = Path(__file__).with_name("10_ppo_pilot.py")
    spec = importlib.util.spec_from_file_location("ppo_pilot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PPO = load_ppo_module()
# The exported app can take longer than the default 60 s to initialize on a
# cold macOS launch; keep the experiment from failing before the world is up.
GodotEnv.DEFAULT_TIMEOUT = 180


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def path_cells(route: dict) -> set[tuple[int, int]]:
    return {tuple(int(v) for v in cell) for cell in route.get("path_cells", [])}


def jaccard(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cluster_routes(routes: list[dict], threshold: float = 0.80) -> list[dict]:
    """Greedy route clustering; keep the fastest route per cluster."""
    clusters: list[dict] = []
    for route in sorted(routes, key=lambda item: float(item["run_time"])):
        cells = path_cells(route)
        for cluster in clusters:
            if jaccard(cells, cluster["cells"]) >= threshold:
                cluster["members"].append(route)
                break
        else:
            clusters.append({"cells": cells, "members": [route]})
    reps = []
    for index, cluster in enumerate(clusters):
        rep = min(cluster["members"], key=lambda item: float(item["run_time"]))
        rep = dict(rep)
        rep["cluster"] = index
        rep["cluster_size"] = len(cluster["members"])
        reps.append(rep)
    return reps


def select_routes(
    clean_routes: list[dict],
    top_fraction: float,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    ordered = sorted(clean_routes, key=lambda item: float(item["run_time"]))
    keep = max(1, math.ceil(len(ordered) * top_fraction))
    clusters = cluster_routes(ordered[:keep])
    clusters.sort(key=lambda item: float(item["run_time"]))

    rng = np.random.default_rng(seed)
    holdout_count = max(1, int(round(len(clusters) * holdout_fraction)))
    # Keep the fastest cluster in train so the canonical comparison is fair.
    candidates = list(range(1, len(clusters)))
    holdout_indices = set(
        int(v) for v in rng.choice(candidates, size=holdout_count, replace=False)
    )
    train = [cluster for i, cluster in enumerate(clusters) if i not in holdout_indices]
    holdout = [cluster for i, cluster in enumerate(clusters) if i in holdout_indices]
    return train, holdout


def cell_value(grid: list[list[float]], cell: list[int] | tuple[int, int]) -> float:
    x, y = int(cell[0]), int(cell[1])
    if 0 <= y < len(grid) and 0 <= x < len(grid[y]):
        return float(grid[y][x])
    return -1.0


def state_record(info: dict, route: dict, step: int) -> dict | None:
    cell = info.get("cell")
    if not (isinstance(cell, (list, tuple)) and len(cell) == 2):
        return None
    try:
        facing = int(info.get("facing_direction", 1))
    except (TypeError, ValueError):
        facing = 1
    try:
        run_time = float(info.get("run_time", 0.0))
    except (TypeError, ValueError):
        run_time = 0.0
    return {
        "cell": [int(cell[0]), int(cell[1])],
        "facing": 1 if facing >= 0 else -1,
        "on_floor": bool(info.get("on_floor", False)),
        "run_time": run_time,
        "step": step,
        "route_signature": str(route.get("route_signature", "")),
        "route_time": float(route.get("run_time", 0.0)),
        "cluster": int(route.get("cluster", -1)),
    }


def extract(
    routes_path: Path,
    finishers_path: Path,
    analysis_path: Path,
    run_dir: Path,
    top_fraction: float,
    holdout_fraction: float,
    seed: int,
    speedup: float,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    routes_dir = run_dir / "routes"
    spawn_dir = run_dir / "spawns"
    routes_dir.mkdir(exist_ok=True)
    spawn_dir.mkdir(exist_ok=True)
    (run_dir / "godot").mkdir(exist_ok=True)

    shutil.copy2(routes_path, routes_dir / "1-1_clean_routes.json")
    if finishers_path.exists():
        shutil.copy2(finishers_path, routes_dir / "1-1_finishers.jsonl")

    clean_routes = load_json(routes_path)
    analysis = load_json(analysis_path)
    train_routes, holdout_routes = select_routes(
        clean_routes, top_fraction, holdout_fraction, seed
    )

    env_path = PPO.resolve_env_path()
    game = GodotEnv(
        env_path=env_path,
        port=13000,
        show_window=False,
        seed=seed,
        action_repeat=4,
        speedup=speedup,
        nb_agents=1,
        level="1-1",
        convert_action_space=True,
        log_path=str(run_dir / "godot" / "route_replay.log"),
        env=PPO.isolated_home(run_dir, "route_replay"),
        reward_profile="default",
        obs_encoding="legacy",
    )

    records: dict[tuple[int, int], dict] = {}
    route_log: list[dict] = []
    try:
        for split, routes in (("train", train_routes), ("holdout", holdout_routes)):
            for route in routes:
                obs, infos = game.reset()
                info = infos[0] if infos else {}
                finished = False
                for step, action in enumerate(route.get("actions", [])):
                    record = state_record(info, route, step)
                    if record is not None:
                        record["split"] = split
                        record["source"] = "neat_route"
                        if (
                            record["on_floor"]
                            and record["run_time"] > 0.02
                            and cell_value(analysis["time_to_exit"], record["cell"]) >= 0.5
                        ):
                            key = (record["cell"][0], record["cell"][1], record["facing"])
                            if key not in records or record["route_time"] < records[key]["route_time"]:
                                records[key] = record
                    obs, rewards, dones, _timeouts, infos = game.step(
                        np.asarray([[int(action)]], dtype=np.int64)
                    )
                    info = infos[0] if infos else {}
                    if bool(dones[0]):
                        finished = bool(rewards[0] > 0.0 or info.get("finished", False))
                        break
                route_log.append(
                    {
                        "split": split,
                        "route_signature": route.get("route_signature"),
                        "route_time": route.get("run_time"),
                        "cluster": route.get("cluster"),
                        "finished_in_replay": finished,
                    }
                )
    finally:
        game.close()

    train_entries = []
    holdout_entries = []
    for record in records.values():
        entry = {
            "cell": record["cell"],
            "facing": record["facing"],
            "source": "neat_route",
            "source_route": record["route_signature"],
            "source_time": record["route_time"],
            "cluster": record["cluster"],
            "run_time_at_state": record["run_time"],
            "kind": "route",
        }
        if record["split"] == "train":
            train_entries.append(entry)
        else:
            holdout_entries.append(entry)

    # Canonical stays in the training pool; route starts share the rest.
    canonical_weight = 0.15
    route_weight = 0.85 / max(1, len(train_entries))
    train_pool = [
        {
            "canonical": True,
            "kind": "canonical",
            "weight": canonical_weight,
        }
    ]
    for entry in train_entries:
        entry["weight"] = route_weight
        train_pool.append(entry)

    payload = {
        "level": "1-1",
        "source": str(routes_path),
        "top_fraction": top_fraction,
        "holdout_fraction": holdout_fraction,
        "seed": seed,
        "train_routes": len(train_routes),
        "holdout_routes": len(holdout_routes),
        "train_spawns": len(train_entries),
        "holdout_spawns": len(holdout_entries),
        "canonical_weight": canonical_weight,
        "route_weight": route_weight,
        "train": train_pool,
        "holdout": holdout_entries,
        "route_replay": route_log,
    }
    (spawn_dir / "1-1_spawn_pool.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def make_train_env(rank: int, args: argparse.Namespace, env_path: str):
    return PPO.SingleAgentGodotEnv(
        env_path=env_path,
        level=args.level,
        port=args.base_port + rank,
        seed=args.seed + rank,
        action_repeat=4,
        speedup=args.speedup,
        max_decisions=args.max_decisions,
        show_window=False,
        log_path=str(args.run_dir / "godot" / f"train_{rank}.log"),
        child_env=PPO.isolated_home(args.run_dir, f"train_{rank}"),
        force_start_jump=True,
        reward_profile="time",
        obs_encoding="legacy",
        start_pool=args.train_pool,
        start_seed=args.seed + rank,
    )


def train(args: argparse.Namespace, env_path: str) -> Path:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "dqn").mkdir(exist_ok=True)
    if args.n_envs == 1:
        venv = DummyVecEnv([lambda: make_train_env(0, args, env_path)])
    else:
        venv = DummyVecEnv(
            [lambda rank=rank: make_train_env(rank, args, env_path) for rank in range(args.n_envs)]
        )
    venv = VecMonitor(venv, filename=str(args.run_dir / "dqn" / "monitor.csv"))
    callbacks = [
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_every // args.n_envs),
            save_path=str(args.run_dir / "dqn" / "checkpoints"),
            name_prefix="dqn",
        )
    ]
    model = DQN(
        "MlpPolicy",
        venv,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.25,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=str(args.run_dir / "tensorboard"),
        verbose=1,
        device="cpu",
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    model.save(args.run_dir / "dqn" / "dqn_final.zip")
    venv.close()
    return args.run_dir / "dqn" / "dqn_final.zip"


def run_eval_episode(
    model: DQN,
    env_path: str,
    run_dir: Path,
    spawn: dict | None,
    port: int,
    seed: int,
    max_decisions: int,
    speedup: float,
) -> dict:
    spawns = None
    if spawn is not None:
        spawns = [{**spawn, "agent": 0}]
    env = GodotEnv(
        env_path=env_path,
        port=port,
        show_window=False,
        seed=seed,
        action_repeat=4,
        speedup=speedup,
        nb_agents=1,
        level="1-1",
        convert_action_space=True,
        log_path=str(run_dir / "godot" / f"eval_{port}.log"),
        env=PPO.isolated_home(run_dir, f"eval_{port}"),
        reward_profile="time",
        obs_encoding="legacy",
    )
    try:
        obs, infos = env.reset(spawns=spawns)
        # Match training: the reset owns the first jump.
        obs, rewards, dones, _timeouts, infos = env.step(
            np.asarray([[1]], dtype=np.int64)
        )
        steps = 1
        finished = bool(dones[0] and rewards[0] > 0.0)
        run_time = float("inf")
        while not bool(dones[0]) and steps < max_decisions:
            model_obs = obs["obs"] if isinstance(obs, dict) else obs
            action, _ = model.predict(
                np.asarray(model_obs, dtype=np.float32).reshape(1, -1),
                deterministic=True,
            )
            obs, rewards, dones, _timeouts, infos = env.step(
                np.asarray([[int(action[0])]], dtype=np.int64)
            )
            steps += 1
            info = infos[0] if infos else {}
            finished = bool(dones[0] and (rewards[0] > 0.0 or info.get("finished", False)))
            if finished:
                run_time = float(info.get("run_time", steps * 4.0 / 60.0))
        if finished and not np.isfinite(run_time):
            run_time = steps * 4.0 / 60.0
        info = infos[0] if infos else {}
        return {
            "finished": finished,
            "time": run_time if finished else None,
            "steps": steps,
            "timeout": bool(not finished and steps >= max_decisions),
            "cell": spawn.get("cell") if spawn else None,
            "facing": spawn.get("facing") if spawn else None,
            "start": "canonical" if spawn is None else "route",
            "info": info,
        }
    finally:
        env.close()


def evaluate(args: argparse.Namespace, env_path: str, model_path: Path) -> dict:
    model = DQN.load(model_path, device="cpu")
    pool = load_json(args.run_dir / "spawns" / "1-1_spawn_pool.json")
    canonical = run_eval_episode(
        model,
        env_path,
        args.run_dir,
        None,
        args.eval_port,
        args.seed,
        args.max_decisions,
        args.speedup,
    )
    results = []
    for index, spawn in enumerate(pool.get("holdout", [])):
        results.append(
            run_eval_episode(
                model,
                env_path,
                args.run_dir,
                spawn,
                args.eval_port + 1 + index,
                args.seed + 1 + index,
                args.max_decisions,
                args.speedup,
            )
        )
    summary = {
        "canonical": canonical,
        "holdout": results,
        "holdout_finished": sum(1 for row in results if row["finished"]),
        "holdout_total": len(results),
    }
    (args.run_dir / "eval").mkdir(exist_ok=True)
    (args.run_dir / "eval" / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
    parser.add_argument("--finishers", type=Path, default=DEFAULT_FINISHERS)
    parser.add_argument("--level-analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--level", default="1-1")
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--base-port", type=int, default=13000)
    parser.add_argument("--eval-port", type=int, default=13100)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--learning-starts", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=25000)

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("extract")
    sub.add_parser("train")
    sub.add_parser("eval")
    sub.add_parser("all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_path = PPO.resolve_env_path()
    if args.command in ("extract", "all"):
        result = extract(
            args.routes,
            args.finishers,
            args.level_analysis,
            args.run_dir,
            args.top_fraction,
            args.holdout_fraction,
            args.seed,
            args.speedup,
        )
        print(
            f"spawns: train={result['train_spawns']} holdout={result['holdout_spawns']} "
            f"routes={result['train_routes']}+{result['holdout_routes']}",
            flush=True,
        )
        pool = result["train"]
        args.train_pool = pool
        (args.run_dir / "spawns" / "1-1_start_pool.json").write_text(
            json.dumps(pool, indent=2), encoding="utf-8"
        )
    else:
        pool_path = args.run_dir / "spawns" / "1-1_start_pool.json"
        if not pool_path.exists():
            raise SystemExit(f"missing {pool_path}; run extract first")
        args.train_pool = load_json(pool_path)

    if args.command in ("train", "all"):
        model_path = train(args, env_path)
        print(f"model: {model_path}", flush=True)

    if args.command in ("eval", "all"):
        model_path = args.run_dir / "dqn" / "dqn_final.zip"
        summary = evaluate(args, env_path, model_path)
        print(json.dumps({
            "canonical": summary["canonical"],
            "holdout_finished": summary["holdout_finished"],
            "holdout_total": summary["holdout_total"],
        }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
