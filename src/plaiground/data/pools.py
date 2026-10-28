"""Load and normalize PPO/DQN start pools."""

from __future__ import annotations

import json
from pathlib import Path

from plaiground.envs.single_agent import RESET_STATE_KEYS


def flatten_reset_state(entry: dict) -> dict:
    """Return a clean-start entry with legacy nested ``state`` flattened."""
    item = dict(entry)
    legacy = item.pop("state", None)
    if isinstance(legacy, dict):
        for key in RESET_STATE_KEYS:
            if key in legacy and key not in item:
                item[key] = legacy[key]
    if "powerup_stack" in item:
        stack = item["powerup_stack"]
        if not isinstance(stack, (list, tuple)) or len(stack) > 3:
            raise ValueError(f"powerup_stack must be a list[0..3], got {stack!r}")
        item["powerup_stack"] = [int(value) for value in stack]
    if "run_time" in item:
        item["run_time"] = float(item["run_time"])
        if item["run_time"] < 0.0:
            raise ValueError(f"run_time must be non-negative: {item['run_time']!r}")
    if "best_run_time" in item:
        item["best_run_time"] = float(item["best_run_time"])
    if "world_state" in item and not isinstance(item["world_state"], dict):
        raise ValueError(f"world_state must be an object: {item['world_state']!r}")
    return item


def load_start_pool(path: Path) -> list[dict]:
    """Load optional canonical/cell/NEAT-prefix start specifications."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("starts") if isinstance(payload, dict) else payload
    payload_level = (
        str(payload.get("level"))
        if isinstance(payload, dict) and payload.get("level") is not None
        else None
    )
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} has no non-empty start list")
    pool: list[dict] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path} start[{index}] must be an object")
        item = flatten_reset_state(entry)
        if payload_level is not None:
            item.setdefault("level", payload_level)
        item["weight"] = max(0.0, float(item.get("weight", 1.0)))
        prefix = item.get("prefix_actions")
        if prefix is not None:
            if not isinstance(prefix, list):
                raise ValueError(f"{path} start[{index}].prefix_actions must be a list")
            item["prefix_actions"] = [int(action) for action in prefix]
        elif not item.get("canonical"):
            cell = item.get("cell")
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                raise ValueError(f"{path} start[{index}] needs cell or prefix_actions")
            item["cell"] = [int(cell[0]), int(cell[1])]
            item["facing"] = 1 if int(item.get("facing", 1)) >= 0 else -1
            if "position" in item:
                position = item["position"]
                if not isinstance(position, (list, tuple)) or len(position) != 2:
                    raise ValueError(f"{path} start[{index}].position must be [x, y]")
            if "velocity" in item:
                velocity = item["velocity"]
                if not isinstance(velocity, (list, tuple)) or len(velocity) != 2:
                    raise ValueError(f"{path} start[{index}].velocity must be [x, y]")
        if item["weight"] <= 0.0:
            continue
        pool.append(item)
    if not pool:
        raise ValueError(f"{path} has no positive-weight starts")
    return pool
