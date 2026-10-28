#!/usr/bin/env python3
"""Smoke-check the raw local-state observation from Godot.

Runs one world with ``obs_encoding=raw``, steps random actions, decodes the
flat vector with ``obs_spec.decode_raw``, and prints shapes/ranges plus a few
sanity assertions. This is a protocol check, not training.

Usage:
    .venv/bin/python tools/check_observation.py --level 1-1 --steps 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plaiground.algorithms.reinforcement import ppo as PPO
from plaiground.envs.godot import GodotEnv
from plaiground.observations import spec as obs_spec

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="1-1")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--port", type=int, default=13200)
    parser.add_argument("--out", type=Path, default=REPO / "runs/raw_obs_check")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env_path = PPO.resolve_env_path()
    env = GodotEnv(
        env_path=env_path,
        port=args.port,
        show_window=False,
        seed=args.seed,
        action_repeat=4,
        speedup=args.speedup,
        nb_agents=1,
        level=args.level,
        convert_action_space=True,
        log_path=str(args.out / "godot.log"),
        env=PPO.isolated_home(args.out, "raw_check"),
        reward_profile="time",
        obs_encoding="raw",
    )
    expected = obs_spec.expected_obs_size("raw")
    rng = np.random.default_rng(args.seed)
    records = []
    tile_names = env.static_info.get("tile_names", [])
    try:
        obs, infos = env.reset()
        for step in range(args.steps):
            values = np.asarray(obs[0]["obs"], dtype=np.float32)
            decoded = obs_spec.decode_raw(values)
            info = infos[0] if infos else {}
            cell = info.get("cell")
            center_id = int(decoded["tile_ids"][3, 3])
            center_name = (
                tile_names[center_id]
                if 0 <= center_id < len(tile_names)
                else None
            )
            record = {
                "step": step,
                "size": int(values.size),
                "center_tile_id": center_id,
                "center_tile_name": center_name,
                "tile_ids_shape": list(decoded["tile_ids"].shape),
                "tile_states_shape": list(decoded["tile_states"].shape),
                "collisions_shape": list(decoded["collisions"].shape),
                "flow_shape": list(decoded["flow"].shape),
                "player": decoded["player"],
                "info_cell": cell,
                "info_gold_ratio": info.get("gold_ratio"),
                "info_flow_cost": info.get("flow_cost"),
                "info_flow_in_bounds": info.get("flow_in_bounds"),
                "info_wasted_air_frames": info.get("wasted_air_frames"),
                "tile_state_nonzero": {
                    name: int(
                        np.count_nonzero(
                            decoded["tile_states"][:, index] >= 0
                            if name == "powerup_kind"
                            else decoded["tile_states"][:, index] != 0
                        )
                    )
                    for index, name in enumerate(obs_spec.RAW_TILE_STATE_FIELDS)
                },
                "collision_hits": int(np.count_nonzero(decoded["collisions"][:, 1])),
                "collision_types": sorted(
                    {int(v) for v in decoded["collisions"][:, 2]}
                ),
                "flow": [round(float(v), 4) for v in decoded["flow"]],
                "flow_min": float(decoded["flow"].min()),
                "flow_max": float(decoded["flow"].max()),
            }
            if cell is not None:
                record["cell_matches"] = (
                    int(decoded["player"]["cell_x"]) == int(cell[0])
                    and int(decoded["player"]["cell_y"]) == int(cell[1])
                )
            records.append(record)
            action = int(rng.integers(0, 2))
            obs, _rewards, dones, _timeouts, infos = env.step(
                np.asarray([[action]], dtype=np.int64)
            )
            if bool(dones[0]):
                obs, infos = env.reset()
    finally:
        env.close()

    sizes = sorted({row["size"] for row in records})
    cell_matches = [row["cell_matches"] for row in records if "cell_matches" in row]
    summary = {
        "level": args.level,
        "expected_size": expected,
        "observed_sizes": sizes,
        "steps": len(records),
        "cell_matches": sum(bool(v) for v in cell_matches),
        "cell_checks": len(cell_matches),
        "first": records[0],
        "last": records[-1],
        "tile_state_nonzero_last": records[-1]["tile_state_nonzero"],
        "collision_hits_last": records[-1]["collision_hits"],
        "collision_types_last": records[-1]["collision_types"],
        "flow_last": [round(float(v), 4) for v in records[-1].get("flow", [])],
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if sizes != [expected]:
        raise SystemExit(f"raw size mismatch: expected {expected}, got {sizes}")
    if cell_matches and not all(cell_matches):
        raise SystemExit("raw player cell does not match info cell")
    if any(row["info_gold_ratio"] is None for row in records):
        raise SystemExit("info is missing gold_ratio")
    if any(
        abs(
            float(row["player"]["gold_ratio"])
            - float(row["info_gold_ratio"])
        ) > 1e-5
        for row in records
    ):
        raise SystemExit("raw gold_ratio does not match info gold_ratio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
