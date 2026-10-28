#!/usr/bin/env python3
"""DQN on NEAT clean-start states for one level.

The clean-start target is raw state with source-checkpoint time, held powerups,
and a dynamic ``world_state`` snapshot. The legacy 794 ``multi_hot_sensor``
prefix path is retained only for archived baseline runs; 58-value cell
checkpoints are fallback only.

The DQN collects its own transitions from those starts; NEAT actions are not
used as imitation labels. Prefix entries replay NEAT actions during reset only.

The first experiment is 1-1 only. See docs/experiments/dqn-route-spawns.md,
docs/audit/2026-10-28-dqn-pipeline-review.md, and
docs/audit/2026-10-28-start-pool-clock-bias.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from functools import partial
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)

from plaiground.envs.godot import GodotEnv
from plaiground.envs.single_agent import RESET_STATE_KEYS, SingleAgentGodotEnv
from plaiground.observations import spec as obs_spec

REPO = Path(__file__).resolve().parents[4]
DEFAULT_ROUTES = REPO / "runs/neat_1_1_8x24_archive/clean/clean_routes.json"
DEFAULT_FINISHERS = REPO / "runs/neat_1_1_8x24_archive/archive/finishers.jsonl"
DEFAULT_ANALYSIS = Path(
    "/Users/giani.statie/personal/dragon-jump-remaster/runs/level_analysis/1-1.json"
)
DEFAULT_RUN = REPO / "runs/dqn_clean_start_1_1"


from . import ppo as PPO
# The exported app can take longer than the default 60 s to initialize on a
# cold macOS launch; keep the experiment from failing before the world is up.
GodotEnv.DEFAULT_TIMEOUT = 180


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cell_value(grid: list[list[float]], cell: list[int] | tuple[int, int]) -> float:
    x, y = int(cell[0]), int(cell[1])
    if 0 <= y < len(grid) and 0 <= x < len(grid[y]):
        return float(grid[y][x])
    return -1.0


def state_record(
    info: dict,
    route: dict,
    step: int,
    world_state: dict | None = None,
) -> dict | None:
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
    powerup_stack = info.get("powerup_stack")
    if not isinstance(powerup_stack, (list, tuple)):
        powerup_stack = [-1.0, -1.0, -1.0]
    record = {
        "cell": [int(cell[0]), int(cell[1])],
        "facing": 1 if facing >= 0 else -1,
        "on_floor": bool(info.get("on_floor", False)),
        "position": info.get("global_position", None),
        "velocity": info.get("velocity", [0.0, 0.0]),
        "state_name": str(info.get("state", "Move")),
        "powerup_stack": [int(value) for value in powerup_stack],
        "run_time": run_time,
        "best_run_time": info.get("best_run_time", float("inf")),
        "run_active": bool(info.get("run_active", True)),
        "started_walking": bool(info.get("started_walking", True)),
        "step": step,
        "route_signature": str(route.get("route_signature", "")),
        "route_time": float(route.get("run_time", 0.0)),
    }
    if world_state is not None:
        record["world_state"] = world_state
    return record


def request_world_state(game: GodotEnv, agent: int = 0) -> dict | None:
    if not game.supports_info_request:
        return None
    try:
        response = game.request_info(["world_state"], agents=[agent])
    except (RuntimeError, ValueError) as exc:
        print(f"[extract] world_state request failed: {exc}", flush=True)
        return None
    return (response.get("world_state") or {}).get(str(agent), {})


def _stack_matches_world_state(stack, world_state: dict | None) -> bool:
    if not isinstance(stack, (list, tuple)):
        return False
    expected = [int(value) for value in stack if int(value) >= 0]
    if not isinstance(world_state, dict):
        return not expected
    objects = world_state.get("objects")
    if not isinstance(objects, dict):
        return not expected
    held = []
    for entry in objects.values():
        if not isinstance(entry, dict) or not bool(entry.get("held", False)):
            continue
        kind_name = str(entry.get("type", ""))
        if kind_name not in obs_spec.POWERUP_KIND_NAMES:
            return False
        held.append((
            int(entry.get("stack_index", -1)),
            obs_spec.POWERUP_KIND_NAMES.index(kind_name),
        ))
    held.sort()
    return expected == [kind for _, kind in held]


def _valid_state(checkpoint: dict, world_state: dict | None = None) -> bool:
    if not isinstance(checkpoint, dict):
        return False
    cell = checkpoint.get("cell")
    if not (isinstance(cell, (list, tuple)) and len(cell) == 2):
        return False
    if not bool(checkpoint.get("on_floor", False)):
        return False
    if str(checkpoint.get("state_name", "Move")) not in ("Idle", "Move"):
        return False
    try:
        run_time = float(checkpoint.get("run_time", 0.0))
    except (TypeError, ValueError):
        return False
    if run_time <= 0.02:
        return False
    state_world = checkpoint.get("world_state")
    if world_state is None:
        world_state = state_world if isinstance(state_world, dict) else None
    return _stack_matches_world_state(checkpoint.get("powerup_stack", []), world_state)


def _resolve_checkpoint_path(routes_path: Path, checkpoint_file: str) -> Path | None:
    candidate = Path(checkpoint_file)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    bases = [
        routes_path.parent,
        routes_path.parent.parent,
        routes_path.parent.parent / "archive",
        REPO,
    ]
    for base in bases:
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return None


def extract_checkpoint_routes(
    clean_routes: list[dict],
    routes_path: Path,
    analysis: dict,
    run_dir: Path,
    level: str,
    min_time_to_exit: float,
) -> dict | None:
    """Build a clean-start pool from snapshots captured by ``07``.

    The one-pass route archive stores a ``checkpoint_file`` per finisher.
    Keep the fastest final route per signature, then take its checkpoint
    snapshots. No game replay is needed. Returns ``None`` when the archive has
    no usable checkpoint data so the caller can fall back to replay.
    """
    best_by_signature: dict[str, dict] = {}
    for route in clean_routes:
        if not isinstance(route, dict) or not route.get("checkpoint_file"):
            continue
        signature = str(route.get("route_signature", ""))
        if not signature:
            continue
        try:
            route_time = float(route["run_time"])
        except (KeyError, TypeError, ValueError):
            continue
        previous = best_by_signature.get(signature)
        if previous is None or route_time < float(previous["run_time"]):
            best_by_signature[signature] = route

    if not best_by_signature:
        return None

    records: dict[tuple[int, int, int, tuple[int, ...]], dict] = {}
    route_log: list[dict] = []
    for route in best_by_signature.values():
        checkpoint_path = _resolve_checkpoint_path(
            routes_path, str(route["checkpoint_file"])
        )
        if checkpoint_path is None:
            route_log.append({
                "route_signature": route.get("route_signature"),
                "route_time": route.get("run_time"),
                "checkpoints": 0,
                "error": "missing checkpoint file",
            })
            continue
        try:
            checkpoints = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            checkpoints = []
        accepted = 0
        for checkpoint in checkpoints if isinstance(checkpoints, list) else []:
            if not isinstance(checkpoint, dict):
                continue
            cell = checkpoint.get("cell")
            if not (isinstance(cell, (list, tuple)) and len(cell) == 2):
                continue
            if not _valid_state(checkpoint):
                continue
            if not bool(checkpoint.get("on_floor", False)):
                continue
            if str(checkpoint.get("state_name", "Move")) not in ("Idle", "Move"):
                continue
            try:
                run_time = float(checkpoint.get("run_time", 0.0))
            except (TypeError, ValueError):
                continue
            if run_time <= 0.02:
                continue
            if cell_value(analysis["time_to_exit"], cell) < min_time_to_exit:
                continue
            stack = checkpoint.get("powerup_stack", [-1, -1, -1])
            if not isinstance(stack, (list, tuple)):
                stack = [-1, -1, -1]
            stack_tuple = tuple(int(value) for value in stack)
            if any(value >= 0 for value in stack_tuple) and not checkpoint.get(
                "world_state"
            ):
                continue
            facing = 1 if int(checkpoint.get("facing", 1)) >= 0 else -1
            record = {
                "cell": [int(cell[0]), int(cell[1])],
                "facing": facing,
                "position": checkpoint.get("position"),
                "velocity": checkpoint.get("velocity", [0.0, 0.0]),
                "state_name": str(checkpoint.get("state_name", "Move")),
                "powerup_stack": list(stack_tuple),
                "run_time": run_time,
                "best_run_time": checkpoint.get("best_run_time", float("inf")),
                "run_active": bool(checkpoint.get("run_active", True)),
                "started_walking": bool(checkpoint.get("started_walking", True)),
                "world_state": checkpoint.get("world_state"),
                "route_signature": route.get("route_signature"),
                "route_time": float(route["run_time"]),
                "step": int(checkpoint.get("step", 0)),
            }
            key = (
                record["cell"][0],
                record["cell"][1],
                facing,
                stack_tuple,
            )
            if key not in records or run_time < float(records[key]["run_time"]):
                records[key] = record
                accepted += 1
        route_log.append({
            "route_signature": route.get("route_signature"),
            "route_time": route.get("run_time"),
            "checkpoints": accepted,
        })

    train_entries = []
    for record in records.values():
        entry = {
            "cell": record["cell"],
            "facing": record["facing"],
            "velocity": [0.0, 0.0],
            "state_name": record.get("state_name", "Move"),
            "powerup_stack": record.get("powerup_stack", [-1, -1, -1]),
            "run_time": record.get("run_time", 0.0),
            "best_run_time": record.get("best_run_time", float("inf")),
            "run_active": record.get("run_active", True),
            "started_walking": record.get("started_walking", True),
            "source": "neat_checkpoint",
            "source_route": record["route_signature"],
            "source_time": record["route_time"],
            "run_time_at_state": record["run_time"],
            "kind": "state",
        }
        position = record.get("position")
        if isinstance(position, (list, tuple)) and len(position) == 2:
            entry["position"] = [float(position[0]), float(position[1])]
        if record.get("world_state"):
            entry["world_state"] = record["world_state"]
        train_entries.append(entry)

    if not train_entries:
        return None

    canonical_weight = 0.70
    route_weight = 0.30 / max(1, len(train_entries))
    train_pool = [{
        "canonical": True,
        "kind": "canonical",
        "weight": canonical_weight,
    }]
    for entry in train_entries:
        entry["weight"] = route_weight
        train_pool.append(entry)

    payload = {
        "level": level,
        "source": str(routes_path),
        "min_time_to_exit": min_time_to_exit,
        "train_routes": len(best_by_signature),
        "holdout_routes": 0,
        "train_spawns": len(train_entries),
        "holdout_spawns": 0,
        "canonical_weight": canonical_weight,
        "route_weight": route_weight,
        "train": train_pool,
        "holdout": [],
        "route_replay": route_log,
    }
    spawn_dir = run_dir / "spawns"
    spawn_dir.mkdir(parents=True, exist_ok=True)
    (spawn_dir / f"{level}_spawn_pool.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def extract(
    routes_path: Path,
    finishers_path: Path,
    analysis_path: Path,
    run_dir: Path,
    min_time_to_exit: float,
    seed: int,
    speedup: float,
    level: str = "1-1",
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    routes_dir = run_dir / "routes"
    spawn_dir = run_dir / "spawns"
    routes_dir.mkdir(exist_ok=True)
    spawn_dir.mkdir(exist_ok=True)
    (run_dir / "godot").mkdir(exist_ok=True)

    shutil.copy2(routes_path, routes_dir / f"{level}_clean_routes.json")
    if finishers_path.exists():
        shutil.copy2(finishers_path, routes_dir / f"{level}_finishers.jsonl")

    clean_routes = load_json(routes_path)
    analysis = load_json(analysis_path)
    checkpoint_pool = extract_checkpoint_routes(
        clean_routes,
        routes_path,
        analysis,
        run_dir,
        level,
        min_time_to_exit,
    )
    if checkpoint_pool is not None:
        return checkpoint_pool

    train_routes = sorted(clean_routes, key=lambda item: float(item["run_time"]))
    holdout_routes: list[dict] = []

    env_path = PPO.resolve_env_path()
    game = GodotEnv(
        env_path=env_path,
        port=13000,
        show_window=False,
        seed=seed,
        action_repeat=4,
        speedup=speedup,
        nb_agents=1,
        level=level,
        convert_action_space=True,
        log_path=str(run_dir / "godot" / "route_replay.log"),
        env=PPO.isolated_home(run_dir, "route_replay"),
        reward_profile="default",
        obs_encoding="raw",
    )

    records: dict[tuple[int, int, int, tuple[int, ...]], dict] = {}
    route_log: list[dict] = []
    try:
        for route in train_routes:
            obs, infos = game.reset()
            info = infos[0] if infos else {}
            finished = False
            for step, action in enumerate(route.get("actions", [])):
                record = state_record(info, route, step)
                if record is not None:
                    record["source"] = "neat_route"
                    if (
                        record["on_floor"]
                        and record["state_name"] in ("Idle", "Move")
                        and record["run_time"] > 0.02
                        and cell_value(
                            analysis["time_to_exit"], record["cell"]
                        ) >= min_time_to_exit
                    ):
                        world_state = request_world_state(game, 0)
                        if world_state is not None:
                            record["world_state"] = world_state
                        if _valid_state(record, world_state):
                            stack = tuple(record["powerup_stack"])
                            key = (
                                record["cell"][0],
                                record["cell"][1],
                                record["facing"],
                                stack,
                            )
                            if key not in records or record["route_time"] < records[key]["route_time"]:
                                records[key] = record
                obs, rewards, dones, _timeouts, infos = game.step(
                    np.asarray([[int(action)]], dtype=np.int64)
                )
                info = infos[0] if infos else {}
                if bool(dones[0]):
                    finished = bool(info.get("finished", False))
                    break
            route_log.append(
                {
                    "route_signature": route.get("route_signature"),
                    "route_time": route.get("run_time"),
                    "finished_in_replay": finished,
                }
            )
    finally:
        game.close()

    train_entries = []
    for record in records.values():
        entry = {
            "cell": record["cell"],
            "facing": record["facing"],
            "velocity": [0.0, 0.0],
            "state_name": record.get("state_name", "Move"),
            "powerup_stack": record.get("powerup_stack", [-1, -1, -1]),
            "run_time": record.get("run_time", 0.0),
            "best_run_time": record.get("best_run_time", float("inf")),
            "run_active": record.get("run_active", True),
            "started_walking": record.get("started_walking", True),
            "source": "neat_route",
            "source_route": record["route_signature"],
            "source_time": record["route_time"],
            "run_time_at_state": record["run_time"],
            "kind": "state",
        }
        position = record.get("position")
        if isinstance(position, (list, tuple)) and len(position) == 2:
            entry["position"] = [float(position[0]), float(position[1])]
        if "world_state" in record:
            entry["world_state"] = record["world_state"]
        train_entries.append(entry)

    # Canonical 0.70 / clean state 0.30.
    canonical_weight = 0.70
    route_weight = 0.30 / max(1, len(train_entries))
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
        "level": level,
        "source": str(routes_path),
        "min_time_to_exit": min_time_to_exit,
        "seed": seed,
        "train_routes": len(train_routes),
        "holdout_routes": 0,
        "train_spawns": len(train_entries),
        "holdout_spawns": 0,
        "canonical_weight": canonical_weight,
        "route_weight": route_weight,
        "train": train_pool,
        "holdout": [],
        "route_replay": route_log,
    }
    (spawn_dir / f"{level}_spawn_pool.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def make_train_env(rank: int, args: argparse.Namespace, env_path: str):
    return SingleAgentGodotEnv(
        env_path=env_path,
        level=args.level,
        port=args.base_port + rank,
        seed=args.seed + rank,
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        max_decisions=args.max_decisions,
        show_window=False,
        log_path=str(args.run_dir / "godot" / f"train_{rank}.log"),
        child_env=PPO.isolated_home(args.run_dir, f"train_{rank}"),
        force_start_jump=args.force_start_jump,
        reward_profile="time",
        obs_encoding=args.obs_encoding,
        start_pool=args.train_pool,
        start_seed=args.seed + rank,
    )


def train(args: argparse.Namespace, env_path: str) -> Path:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "dqn").mkdir(exist_ok=True)
    (args.run_dir / "godot").mkdir(exist_ok=True)
    if args.start_pool is not None:
        pool_dir = args.run_dir / "pool"
        pool_dir.mkdir(exist_ok=True)
        shutil.copy2(args.start_pool, pool_dir / "start_pool.json")
        (pool_dir / "run_args.json").write_text(
            json.dumps(
                {
                    "start_pool": str(args.start_pool),
                    "level": args.level,
                    "obs_encoding": args.obs_encoding,
                    "net_arch": list(args.net_arch),
                    "n_envs": args.n_envs,
                    "timesteps": args.timesteps,
                    "max_decisions": args.max_decisions,
                    "speedup": args.speedup,
                    "seed": args.seed,
                    "force_start_jump": args.force_start_jump,
                    "reward_profile": "time",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    env_fns = [
        partial(make_train_env, rank, args, env_path)
        for rank in range(args.n_envs)
    ]
    if args.n_envs == 1:
        venv = DummyVecEnv(env_fns)
    else:
        # spawn is the portable macOS start method; only the env factory is
        # pickled, not the Godot socket.
        venv = SubprocVecEnv(env_fns, start_method="spawn")
    venv = VecMonitor(
        venv,
        filename=str(args.run_dir / "dqn" / "monitor.csv"),
        info_keywords=(
            "level",
            "finished",
            "timed_out",
            "run_time",
            "start_source",
            "start_kind",
            "start_cell",
            "prefix_len",
        ),
    )
    callbacks = [
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_every // args.n_envs),
            save_path=str(args.run_dir / "dqn" / "checkpoints"),
            name_prefix="dqn",
        )
    ]
    if args.eval_every > 0 and args.patience > 0:
        callbacks.append(PatienceCallback(args, env_path))
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
        policy_kwargs={"net_arch": list(args.net_arch)},
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
        steps = 0
        info = infos[0] if infos else {}
        finished = False
        run_time = float("inf")
        dones = np.asarray([False])
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
            finished = bool(dones[0] and info.get("finished", False))
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


class PatienceCallback(BaseCallback):
    """Stop on a canonical-time plateau and keep the best checkpoint.

    The training reward is not the stopping signal: DQN can improve its return
    while the deterministic policy stays slow. This evaluates canonical 1-1,
    saves ``dqn/best_model.zip`` on improvement, and stops after
    ``--patience`` evaluations without a meaningful improvement.
    """

    def __init__(self, args: argparse.Namespace, env_path: str):
        super().__init__()
        self.args = args
        self.env_path = env_path
        self.env: GodotEnv | None = None
        self.last_eval = 0
        self.best_time = float("inf")
        self.best_step = -1
        self.no_improve = 0
        self.best_path = args.run_dir / "dqn" / "best_model.zip"
        self.log_path = args.run_dir / "dqn" / "patience.csv"
        self.log_file = open(self.log_path, "w", newline="", encoding="utf-8")
        self.log_writer = csv.DictWriter(
            self.log_file,
            fieldnames=["step", "finished", "time", "best_time", "no_improve"],
        )
        self.log_writer.writeheader()

    def _ensure_env(self) -> GodotEnv:
        if self.env is None:
            self.env = GodotEnv(
                env_path=self.env_path,
                port=self.args.eval_port,
                show_window=False,
                seed=self.args.seed,
                action_repeat=self.args.action_repeat,
                speedup=self.args.speedup,
                nb_agents=1,
                level=self.args.level,
                convert_action_space=True,
                log_path=str(self.args.run_dir / "godot" / "eval_patience.log"),
                env=PPO.isolated_home(self.args.run_dir, "eval_patience"),
                reward_profile="time",
                obs_encoding=self.args.obs_encoding,
            )
        return self.env

    def _run_canonical(self) -> dict:
        env = self._ensure_env()
        obs, infos = env.reset()
        steps = 0
        info = infos[0] if infos else {}
        finished = False
        run_time = float("inf")
        if self.args.force_start_jump:
            obs, rewards, dones, _timeouts, infos = env.step(
                np.asarray([[1]], dtype=np.int64)
            )
            steps = 1
            info = infos[0] if infos else {}
            finished = bool(
                dones[0] and info.get("finished", False)
            )
        else:
            rewards = np.asarray([0.0])
            dones = np.asarray([False])
        while not bool(dones[0]) and steps < self.args.max_decisions:
            model_obs = obs["obs"] if isinstance(obs, dict) else obs
            action, _ = self.model.predict(
                np.asarray(model_obs, dtype=np.float32).reshape(1, -1),
                deterministic=True,
            )
            obs, rewards, dones, _timeouts, infos = env.step(
                np.asarray([[int(action[0])]], dtype=np.int64)
            )
            steps += 1
            info = infos[0] if infos else {}
            finished = bool(
                dones[0] and info.get("finished", False)
            )
            if finished:
                run_time = float(info.get("run_time", steps * 4.0 / 60.0))
        if finished and not np.isfinite(run_time):
            run_time = steps * 4.0 / 60.0
        return {
            "finished": finished,
            "time": run_time if finished else None,
            "steps": steps,
        }

    def _on_step(self) -> bool:
        if self.args.eval_every <= 0 or self.args.patience <= 0:
            return True
        if self.num_timesteps - self.last_eval < self.args.eval_every:
            return True

        self.last_eval = self.num_timesteps
        result = self._run_canonical()
        metric = (
            float(result["time"])
            if result["finished"] and result["time"] is not None
            else float("inf")
        )
        improved = metric < self.best_time * (1.0 - self.args.min_improvement)
        if improved:
            self.best_time = metric
            self.best_step = self.num_timesteps
            self.no_improve = 0
            self.model.save(str(self.best_path))
            print(
                f"[patience] best canonical={metric:.3f}s at {self.num_timesteps}",
                flush=True,
            )
        else:
            self.no_improve += 1
            current = "inf" if not np.isfinite(metric) else f"{metric:.3f}s"
            print(
                f"[patience] no improvement {self.no_improve}/"
                f"{self.args.patience} at {self.num_timesteps} "
                f"(canonical={current})",
                flush=True,
            )
        self.log_writer.writerow(
            {
                "step": self.num_timesteps,
                "finished": int(bool(result["finished"])),
                "time": "" if result["time"] is None else f"{result['time']:.6f}",
                "best_time": "" if not np.isfinite(self.best_time) else f"{self.best_time:.6f}",
                "no_improve": self.no_improve,
            }
        )
        self.log_file.flush()
        if self.no_improve >= self.args.patience:
            print(
                f"[patience] stopping; best canonical={self.best_time:.3f}s "
                f"at {self.best_step}",
                flush=True,
            )
            return False
        return True

    def _on_training_end(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
        self.log_file.flush()
        self.log_file.close()


def evaluate(args: argparse.Namespace, env_path: str, model_path: Path) -> dict:
    """Evaluate canonical + strata starts with one persistent batched env."""
    if args.eval_starts is None:
        raise SystemExit("--eval-starts is required for eval")
    model = DQN.load(model_path, device="cpu")
    starts_path = args.eval_starts
    raw_starts = json.loads(starts_path.read_text(encoding="utf-8"))
    if not isinstance(raw_starts, list) or not raw_starts:
        raise SystemExit(f"{starts_path} has no starts")
    starts = [
        PPO._flatten_reset_state(item)
        for item in raw_starts
        if isinstance(item, dict)
    ]
    if not starts:
        raise SystemExit(f"{starts_path} has no usable starts")
    count = len(starts)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = args.run_dir / "eval"
    if args.eval_tag:
        eval_dir = eval_dir / args.eval_tag
    eval_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "godot").mkdir(exist_ok=True)

    env = GodotEnv(
        env_path=env_path,
        port=args.eval_port,
        show_window=False,
        seed=args.seed,
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        nb_agents=count,
        level=args.level,
        convert_action_space=True,
        log_path=str(args.run_dir / "godot" / "eval.log"),
        env=PPO.isolated_home(args.run_dir, "eval"),
        reward_profile="time",
        obs_encoding=args.obs_encoding,
    )
    done = np.zeros(count, dtype=bool)
    finished = np.zeros(count, dtype=bool)
    finish_times = np.full(count, np.inf, dtype=np.float64)
    steps = np.zeros(count, dtype=np.int32)
    actual_spawns: list[list[int] | None] = [None] * count
    try:
        spawns = []
        for index, start in enumerate(starts):
            spawn = {
                "agent": index,
                "cell": list(start["cell"]),
                "facing": int(start.get("facing", 1)),
            }
            for key in RESET_STATE_KEYS:
                if key in start:
                    spawn[key] = start[key]
            spawns.append(spawn)
        obs, _infos = env.reset(spawns=spawns)
        if args.force_start_jump:
            obs, rewards, dones, _timeouts, infos = env.step(
                np.ones((count, 1), dtype=np.int64)
            )
            for index in range(count):
                info = infos[index] if index < len(infos) else {}
                spawn = info.get("spawn_cell", starts[index]["cell"])
                if isinstance(spawn, (list, tuple)) and len(spawn) >= 2:
                    actual_spawns[index] = [int(spawn[0]), int(spawn[1])]
                steps[index] = 1
                if bool(dones[index]):
                    done[index] = True
                    finished[index] = bool(
                        info.get("finished", float(rewards[index]) > 0.0)
                    )
                    if finished[index]:
                        finish_times[index] = float(
                            info.get("run_time", np.inf)
                        )

        for step in range(args.max_decisions):
            if done.all():
                break
            batch = np.stack(
                [
                    np.asarray(obs[index]["obs"], dtype=np.float32)
                    for index in range(count)
                ]
            )
            actions, _ = model.predict(batch, deterministic=True)
            actions = np.asarray(actions, dtype=np.int64).reshape(count, 1)
            actions[done] = 0
            obs, _rewards, dones, _timeouts, infos = env.step(actions)
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
                    finished[index] = bool(
                        info.get("finished", float(_rewards[index]) > 0.0)
                    )
                    if finished[index]:
                        finish_times[index] = float(info.get("run_time", np.inf))
    finally:
        env.close()

    rows = []
    for index, start in enumerate(starts):
        rows.append(
            {
                **start,
                "actual_spawn": actual_spawns[index],
                "spawn_ok": actual_spawns[index] == list(start["cell"]),
                "finished": bool(finished[index]),
                "run_time": (
                    float(finish_times[index])
                    if finished[index] and np.isfinite(finish_times[index])
                    else None
                ),
                "steps": int(steps[index]),
            }
        )
    (eval_dir / "results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    summary: dict[str, dict] = {}
    strata = sorted({str(row.get("stratum", "unknown")) for row in rows})
    for stratum in strata:
        subset = [row for row in rows if str(row.get("stratum", "unknown")) == stratum]
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
            "median": sorted(times)[len(times) // 2] if times else None,
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
    (eval_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--start-pool",
        type=Path,
        default=None,
        help="Clean-start start pool JSON. Required for train/eval; generate "
        "with `41 ... extract` from NEAT checkpoint snapshots.",
    )
    parser.add_argument(
        "--eval-starts",
        type=Path,
        default=None,
        help="JSON eval-start list; required for eval.",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--eval-tag", default="")
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
    parser.add_argument("--finishers", type=Path, default=DEFAULT_FINISHERS)
    parser.add_argument("--level-analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--level", default="1-1")
    parser.add_argument("--min-time-to-exit", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--base-port", type=int, default=13000)
    parser.add_argument("--eval-port", type=int, default=13100)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument(
        "--obs-encoding",
        default="raw",
        choices=("legacy", "multi_hot", "multi_hot_sensor", "raw"),
        help="Observation encoding. raw is the clean-start target; 794 is "
        "legacy baseline only.",
    )
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs=2,
        default=(512, 512),
        metavar=("H1", "H2"),
    )
    parser.add_argument(
        "--force-start-jump",
        dest="force_start_jump",
        action="store_true",
        help="Compatibility: press jump once during reset for old checkpoints "
        "trained before training reset force-started the run.",
    )
    parser.add_argument(
        "--no-force-start-jump",
        dest="force_start_jump",
        action="store_false",
        help="Leave the first start action policy-owned (default).",
    )
    parser.set_defaults(force_start_jump=False)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--learning-starts", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=25000)
    parser.add_argument("--eval-every", type=int, default=25000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-improvement", type=float, default=0.01)

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("extract")
    sub.add_parser("train")
    sub.add_parser("eval")
    sub.add_parser("all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_path = PPO.resolve_env_path()
    if args.command == "extract":
        result = extract(
            args.routes,
            args.finishers,
            args.level_analysis,
            args.run_dir,
            args.min_time_to_exit,
            args.seed,
            args.speedup,
            level=args.level,
        )
        print(
            f"spawns: train={result['train_spawns']} holdout={result['holdout_spawns']} "
            f"routes={result['train_routes']}+{result['holdout_routes']}",
            flush=True,
        )
        pool = result["train"]
        args.train_pool = pool
        (args.run_dir / "spawns" / f"{args.level}_start_pool.json").write_text(
            json.dumps(pool, indent=2), encoding="utf-8"
        )
    else:
        if args.start_pool is None:
            raise SystemExit("--start-pool is required for train/eval/all")
        if not args.start_pool.exists():
            raise SystemExit(
                f"missing start pool {args.start_pool}; generate one with "
                "`scripts/reinforcement/02_dqn.py extract` from NEAT checkpoint "
                "snapshots"
            )
        args.train_pool = PPO.load_start_pool(args.start_pool)
        kinds: dict[str, int] = {}
        for item in args.train_pool:
            kind = str(item.get("kind", "unknown"))
            kinds[kind] = kinds.get(kind, 0) + 1
        print(
            f"start pool: {len(args.train_pool)} starts from "
            f"{args.start_pool} {kinds}",
            flush=True,
        )

    if args.command in ("train", "all"):
        model_path = train(args, env_path)
        print(f"model: {model_path}", flush=True)

    if args.command in ("eval", "all"):
        if args.model is not None:
            model_path = args.model
        else:
            best_path = args.run_dir / "dqn" / "best_model.zip"
            model_path = (
                best_path
                if best_path.exists()
                else args.run_dir / "dqn" / "dqn_final.zip"
            )
        if not model_path.exists():
            raise SystemExit(f"missing model {model_path}; train first")
        summary = evaluate(args, env_path, model_path)
        print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
