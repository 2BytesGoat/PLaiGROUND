"""Extract clean-start state pools from NEAT checkpoint snapshots.

This is data preparation, not DQN training. The one-pass route archive stores
a ``checkpoint_file`` per finisher, so the extractor can build the pool without
replaying the game. If an archive has no checkpoint data, the fallback replays
the fastest clean routes and captures grounded states.

The pool keeps the canonical 0.70 / state 0.30 weighting and writes both the
spawn payload and the flat start list used by training.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from plaiground.envs.godot import GodotEnv
from plaiground.envs.runtime import REPO, isolated_home, resolve_env_path
from plaiground.observations import spec as obs_spec


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cell_value(
    grid: list[list[float]],
    cell: list[int] | tuple[int, int],
) -> float:
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
        held.append(
            (
                int(entry.get("stack_index", -1)),
                obs_spec.POWERUP_KIND_NAMES.index(kind_name),
            )
        )
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
    return _stack_matches_world_state(
        checkpoint.get("powerup_stack", []), world_state
    )


def _resolve_checkpoint_path(
    routes_path: Path,
    checkpoint_file: str,
) -> Path | None:
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


def _entry_from_record(
    record: dict,
    source: str,
) -> dict:
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
        "source": source,
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
    return entry


def _payload(
    *,
    level: str,
    routes_path: Path,
    min_time_to_exit: float,
    train_routes: int,
    entries: list[dict],
    route_log: list[dict],
    extra: dict | None = None,
) -> dict:
    canonical_weight = 0.70 if entries else 1.0
    route_weight = 0.30 / len(entries) if entries else 0.0
    train_pool = [
        {
            "canonical": True,
            "kind": "canonical",
            "weight": canonical_weight,
        }
    ]
    for entry in entries:
        entry["weight"] = route_weight
        train_pool.append(entry)
    payload = {
        "level": level,
        "source": str(routes_path),
        "min_time_to_exit": min_time_to_exit,
        "train_routes": train_routes,
        "holdout_routes": 0,
        "train_spawns": len(entries),
        "holdout_spawns": 0,
        "canonical_weight": canonical_weight,
        "route_weight": route_weight,
        "train": train_pool,
        "holdout": [],
        "route_replay": route_log,
    }
    if extra:
        payload.update(extra)
    return payload


def _write_start_pool(run_dir: Path, level: str, payload: dict) -> None:
    spawn_dir = run_dir / "spawns"
    spawn_dir.mkdir(parents=True, exist_ok=True)
    (spawn_dir / f"{level}_spawn_pool.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (spawn_dir / f"{level}_start_pool.json").write_text(
        json.dumps(payload["train"], indent=2), encoding="utf-8"
    )


def extract_checkpoint_routes(
    clean_routes: list[dict],
    routes_path: Path,
    analysis: dict,
    run_dir: Path,
    level: str,
    min_time_to_exit: float,
) -> dict | None:
    """Build a pool from snapshots captured by the one-pass route archive."""
    if not isinstance(clean_routes, list):
        raise ValueError("clean_routes must be a list")
    if not isinstance(analysis.get("time_to_exit"), list):
        raise ValueError("level analysis has no time_to_exit grid")
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
            route_log.append(
                {
                    "route_signature": route.get("route_signature"),
                    "route_time": route.get("run_time"),
                    "checkpoints": 0,
                    "error": "missing checkpoint file",
                }
            )
            continue
        try:
            checkpoints = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
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
            try:
                run_time = float(checkpoint.get("run_time", 0.0))
            except (TypeError, ValueError):
                continue
            if (
                cell_value(analysis["time_to_exit"], cell)
                < min_time_to_exit
            ):
                continue
            stack = checkpoint.get("powerup_stack", [-1, -1, -1])
            if not isinstance(stack, (list, tuple)):
                stack = [-1, -1, -1]
            stack_tuple = tuple(int(value) for value in stack)
            facing = 1 if int(checkpoint.get("facing", 1)) >= 0 else -1
            record = {
                "cell": [int(cell[0]), int(cell[1])],
                "facing": facing,
                "position": checkpoint.get("position"),
                "velocity": checkpoint.get("velocity", [0.0, 0.0]),
                "state_name": str(checkpoint.get("state_name", "Move")),
                "powerup_stack": list(stack_tuple),
                "run_time": run_time,
                "best_run_time": checkpoint.get(
                    "best_run_time", float("inf")
                ),
                "run_active": bool(checkpoint.get("run_active", True)),
                "started_walking": bool(
                    checkpoint.get("started_walking", True)
                ),
                "world_state": checkpoint.get("world_state"),
                "route_signature": route.get("route_signature"),
                "route_time": float(route["run_time"]),
            }
            key = (
                record["cell"][0],
                record["cell"][1],
                facing,
                stack_tuple,
            )
            if key not in records or run_time < float(
                records[key]["run_time"]
            ):
                records[key] = record
                accepted += 1
        route_log.append(
            {
                "route_signature": route.get("route_signature"),
                "route_time": route.get("run_time"),
                "checkpoints": accepted,
            }
        )

    entries = [
        _entry_from_record(record, "neat_checkpoint")
        for record in records.values()
    ]
    if not entries:
        return None
    payload = _payload(
        level=level,
        routes_path=routes_path,
        min_time_to_exit=min_time_to_exit,
        train_routes=len(best_by_signature),
        entries=entries,
        route_log=route_log,
    )
    _write_start_pool(run_dir, level, payload)
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
        shutil.copy2(
            finishers_path, routes_dir / f"{level}_finishers.jsonl"
        )

    clean_routes = load_json(routes_path)
    if not isinstance(clean_routes, list):
        raise ValueError(
            f"{routes_path} must contain a list of clean routes; "
            "use the clean/clean_routes.json archive, not the route registry"
        )
    analysis = load_json(analysis_path)
    if not isinstance(analysis.get("time_to_exit"), list):
        raise ValueError(f"{analysis_path} has no time_to_exit grid")
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

    train_routes = sorted(
        clean_routes, key=lambda item: float(item["run_time"])
    )
    env_path = resolve_env_path()
    GodotEnv.DEFAULT_TIMEOUT = 180
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
        env=isolated_home(run_dir, "route_replay"),
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
                        )
                        >= min_time_to_exit
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
                            if (
                                key not in records
                                or record["route_time"]
                                < records[key]["route_time"]
                            ):
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

    entries = [
        _entry_from_record(record, "neat_route")
        for record in records.values()
    ]
    payload = _payload(
        level=level,
        routes_path=routes_path,
        min_time_to_exit=min_time_to_exit,
        train_routes=len(train_routes),
        entries=entries,
        route_log=route_log,
        extra={"seed": seed},
    )
    _write_start_pool(run_dir, level, payload)
    return payload
