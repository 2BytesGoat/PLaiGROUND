"""Build PPO clean-start pools from filtered NEAT routes.

The route filter decides which routes belong to train and holdout. This module
expands only the selected routes into checkpoint starts, applies the clean-start
validity gates, and keeps the split intact:

* canonical is included only in the training pool;
* train and holdout state keys are deduplicated separately;
* the same state key may appear in both pools because route clusters can share
  map segments; the summary records that overlap explicitly.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .route_filter import load_routes


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _cell_value(grid: list[list[float]], cell: list[int] | tuple[int, int]) -> float:
    x, y = int(cell[0]), int(cell[1])
    if 0 <= y < len(grid) and 0 <= x < len(grid[y]):
        return float(grid[y][x])
    return -1.0


def _resolve_checkpoint_path(route_path: Path, checkpoint_file: str) -> Path | None:
    candidate = Path(checkpoint_file)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    for base in (
        route_path.parent,
        route_path.parent.parent,
        route_path.parent.parent / "archive",
        Path.cwd(),
    ):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return None


def _valid_checkpoint(
    checkpoint: dict[str, Any],
    analysis: dict[str, Any],
    min_time_to_exit: float,
) -> bool:
    if not isinstance(checkpoint, dict):
        return False
    cell = checkpoint.get("cell")
    if not (isinstance(cell, (list, tuple)) and len(cell) == 2):
        return False
    if not bool(checkpoint.get("on_floor", False)):
        return False
    if str(checkpoint.get("state_name", "Move")) not in ("Idle", "Move"):
        return False
    run_time = _finite(checkpoint.get("run_time"))
    if run_time is None or run_time <= 0.02:
        return False
    time_to_exit = analysis.get("time_to_exit")
    if not isinstance(time_to_exit, list):
        raise ValueError("level analysis has no time_to_exit grid")
    if _cell_value(time_to_exit, cell) < float(min_time_to_exit):
        return False
    stack = checkpoint.get("powerup_stack", [-1, -1, -1])
    if not isinstance(stack, (list, tuple)) or len(stack) > 3:
        return False
    if any(int(value) >= 0 for value in stack) and not isinstance(
        checkpoint.get("world_state"), dict
    ):
        return False
    position = checkpoint.get("position")
    if position is not None and (
        not isinstance(position, (list, tuple)) or len(position) != 2
    ):
        return False
    velocity = checkpoint.get("velocity", [0.0, 0.0])
    if not isinstance(velocity, (list, tuple)) or len(velocity) != 2:
        return False
    return True


def _state_key(checkpoint: dict[str, Any]) -> tuple[int, int, int, tuple[int, ...]]:
    cell = checkpoint["cell"]
    facing = 1 if int(checkpoint.get("facing", 1)) >= 0 else -1
    stack = tuple(
        int(value) for value in checkpoint.get("powerup_stack", [-1, -1, -1])
    )
    return (int(cell[0]), int(cell[1]), facing, stack)


def _state_entry(
    checkpoint: dict[str, Any],
    route: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "cell": [int(checkpoint["cell"][0]), int(checkpoint["cell"][1])],
        "facing": 1 if int(checkpoint.get("facing", 1)) >= 0 else -1,
        "state_name": str(checkpoint.get("state_name", "Move")),
        "powerup_stack": [
            int(value)
            for value in checkpoint.get("powerup_stack", [-1, -1, -1])
        ],
        "run_time": float(checkpoint["run_time"]),
        "run_active": bool(checkpoint.get("run_active", True)),
        "started_walking": bool(checkpoint.get("started_walking", True)),
        "source": "neat_checkpoint",
        "source_route": str(route.get("route_signature", "")),
        "source_time": _finite(route.get("run_time")),
        "run_time_at_state": float(checkpoint["run_time"]),
        "split": split,
        "cluster_id": route.get("cluster_id"),
        "kind": "state",
    }
    best_run_time = _finite(checkpoint.get("best_run_time"))
    if best_run_time is not None:
        entry["best_run_time"] = best_run_time
    position = checkpoint.get("position")
    if position is not None:
        entry["position"] = [float(position[0]), float(position[1])]
    velocity = checkpoint.get("velocity", [0.0, 0.0])
    entry["velocity"] = [float(velocity[0]), float(velocity[1])]
    world_state = checkpoint.get("world_state")
    if isinstance(world_state, dict):
        entry["world_state"] = world_state
    return entry


def build_split_pool(
    routes: Iterable[dict[str, Any]],
    *,
    route_path: Path,
    analysis: dict[str, Any],
    split: str,
    min_time_to_exit: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand selected routes into one deduplicated state list."""
    best: dict[tuple[int, int, int, tuple[int, ...]], dict[str, Any]] = {}
    stats = Counter()
    route_log: list[dict[str, Any]] = []
    for route in routes:
        stats["routes"] += 1
        checkpoint_file = route.get("checkpoint_file")
        if not checkpoint_file:
            stats["routes_without_checkpoint"] += 1
            route_log.append(
                {
                    "route_signature": route.get("route_signature"),
                    "split": split,
                    "accepted": 0,
                    "error": "missing checkpoint_file",
                }
            )
            continue
        path = _resolve_checkpoint_path(route_path, str(checkpoint_file))
        if path is None:
            stats["missing_checkpoint_files"] += 1
            route_log.append(
                {
                    "route_signature": route.get("route_signature"),
                    "split": split,
                    "accepted": 0,
                    "error": "checkpoint file not found",
                }
            )
            continue
        try:
            checkpoints = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            stats["unreadable_checkpoint_files"] += 1
            route_log.append(
                {
                    "route_signature": route.get("route_signature"),
                    "split": split,
                    "accepted": 0,
                    "error": str(exc),
                }
            )
            continue
        accepted = 0
        for checkpoint in checkpoints if isinstance(checkpoints, list) else []:
            if not _valid_checkpoint(checkpoint, analysis, min_time_to_exit):
                stats["rejected_checkpoints"] += 1
                continue
            key = _state_key(checkpoint)
            run_time = float(checkpoint["run_time"])
            previous = best.get(key)
            if previous is None or run_time < float(previous["run_time"]):
                best[key] = _state_entry(checkpoint, route, split)
                accepted += 1
        stats["accepted_checkpoints"] += accepted
        route_log.append(
            {
                "route_signature": route.get("route_signature"),
                "split": split,
                "accepted": accepted,
                "checkpoint_file": str(checkpoint_file),
            }
        )
    states = sorted(
        best.values(),
        key=lambda item: (
            item["run_time"],
            item["cell"][0],
            item["cell"][1],
            item["facing"],
        ),
    )
    summary = {
        "split": split,
        "routes": stats["routes"],
        "accepted_states": len(states),
        "accepted_checkpoints": stats["accepted_checkpoints"],
        "rejected_checkpoints": stats["rejected_checkpoints"],
        "routes_without_checkpoint": stats["routes_without_checkpoint"],
        "missing_checkpoint_files": stats["missing_checkpoint_files"],
        "unreadable_checkpoint_files": stats["unreadable_checkpoint_files"],
        "route_log": route_log,
    }
    return states, summary


def build_start_pools(
    *,
    filtered_dir: Path,
    archive_dir: Path,
    analysis_path: Path,
    output_dir: Path,
    level: str = "1-1",
    min_time_to_exit: float = 0.5,
    canonical_weight: float = 0.70,
    state_weight: float = 0.30,
) -> dict[str, Any]:
    if not 0.0 <= canonical_weight <= 1.0:
        raise ValueError("canonical_weight must be between 0 and 1")
    if not 0.0 <= state_weight <= 1.0:
        raise ValueError("state_weight must be between 0 and 1")
    if abs(canonical_weight + state_weight - 1.0) > 1e-9:
        raise ValueError("canonical_weight + state_weight must equal 1")

    train_path = filtered_dir / "train_clean_routes.json"
    holdout_path = filtered_dir / "holdout_clean_routes.json"
    fallback_path = filtered_dir / "filtered_clean_routes.json"
    if not train_path.exists() or not holdout_path.exists():
        if not fallback_path.exists():
            raise FileNotFoundError(
                f"{filtered_dir} needs train/holdout route files or "
                "filtered_clean_routes.json"
            )
        all_routes = load_routes(fallback_path)
        train_routes = [route for route in all_routes if route.get("split") != "holdout"]
        holdout_routes = [route for route in all_routes if route.get("split") == "holdout"]
    else:
        train_routes = load_routes(train_path)
        holdout_routes = load_routes(holdout_path)

    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    route_path = archive_dir / "routes.json"
    if not route_path.exists():
        route_path = archive_dir

    train_states, train_summary = build_split_pool(
        train_routes,
        route_path=route_path,
        analysis=analysis,
        split="train",
        min_time_to_exit=min_time_to_exit,
    )
    holdout_states, holdout_summary = build_split_pool(
        holdout_routes,
        route_path=route_path,
        analysis=analysis,
        split="holdout",
        min_time_to_exit=min_time_to_exit,
    )

    train_keys = {
        (item["cell"][0], item["cell"][1], item["facing"], tuple(item["powerup_stack"]))
        for item in train_states
    }
    holdout_keys = {
        (item["cell"][0], item["cell"][1], item["facing"], tuple(item["powerup_stack"]))
        for item in holdout_states
    }
    if train_states:
        state_weight_each = state_weight / len(train_states)
    else:
        canonical_weight = 1.0
        state_weight_each = 0.0
    train_pool: list[dict[str, Any]] = [
        {
            "canonical": True,
            "kind": "canonical",
            "weight": float(canonical_weight),
        }
    ]
    for state in train_states:
        entry = dict(state)
        entry["weight"] = float(state_weight_each)
        train_pool.append(entry)
    holdout_pool: list[dict[str, Any]] = []
    for state in holdout_states:
        entry = dict(state)
        entry["weight"] = 1.0 / len(holdout_states) if holdout_states else 0.0
        holdout_pool.append(entry)

    output_dir.mkdir(parents=True, exist_ok=True)
    overlap = train_keys & holdout_keys
    summary = {
        "level": level,
        "filtered_dir": str(filtered_dir),
        "archive_dir": str(archive_dir),
        "analysis": str(analysis_path),
        "min_time_to_exit": min_time_to_exit,
        "train_routes": len(train_routes),
        "holdout_routes": len(holdout_routes),
        "train_states": len(train_states),
        "holdout_states": len(holdout_states),
        "shared_state_keys": len(overlap),
        "canonical_weight": float(canonical_weight),
        "state_weight_total": float(state_weight),
        "state_weight_each": float(state_weight_each),
        "train": train_summary,
        "holdout": holdout_summary,
    }
    train_payload = {
        "level": level,
        "source": str(archive_dir),
        "summary": summary,
        "starts": train_pool,
    }
    holdout_payload = {
        "level": level,
        "source": str(archive_dir),
        "summary": summary,
        "starts": holdout_pool,
    }
    (output_dir / "train_start_pool.json").write_text(
        json.dumps(train_payload, indent=2), encoding="utf-8"
    )
    (output_dir / "holdout_start_pool.json").write_text(
        json.dumps(holdout_payload, indent=2), encoding="utf-8"
    )
    (output_dir / "split.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return {
        "summary": summary,
        "train_pool": train_pool,
        "holdout_pool": holdout_pool,
    }
