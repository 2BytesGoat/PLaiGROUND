#!/usr/bin/env python3
"""Build a small expert buffer for the starts in a clean-start pool.

The DDQN expert replay buffer is sampled uniformly over all transitions. With
many routes, the few transitions that correspond to a particular mid-route
start are diluted. This tool extracts the exact expert transition at each
state start (same source route and decision step) and repeats it so a
fine-tune can learn the start action directly.

The source expert data must be the ``expert_routes.npz`` produced by
``collect_expert_routes.py`` (it stores route signature, decision step, and
per-transition success).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from gymnasium import spaces

from plaiground.algorithms.reinforcement.replay import RareReplayBuffer
from plaiground.experiments.runner import load_run, resolve_path


def _match_starts(
    starts: list[dict],
    data,
    *,
    action_repeat: int,
    suffix: bool = False,
) -> tuple[list[int], list[dict]]:
    signatures = np.asarray(data["route_signature"]).astype(str)
    steps = np.asarray(data["step"]).astype(np.int64)
    success = np.asarray(
        data.get("route_success", np.ones(len(signatures), dtype=bool)),
        dtype=bool,
    )
    selected: list[int] = []
    selected_keys: set[tuple[str, int]] = set()
    log: list[dict] = []
    for index, start in enumerate(starts):
        if start.get("canonical"):
            continue
        signature = str(start.get("source_route", ""))
        run_time = start.get(
            "run_time_at_state", start.get("run_time")
        )
        try:
            step = int(
                round(float(run_time) * 60.0 / float(action_repeat))
            )
        except (TypeError, ValueError):
            step = -1
        matches = np.flatnonzero(
            (signatures == signature)
            & (steps == step)
            & success
        )
        if not matches.size:
            log.append(
                {
                    "index": index,
                    "cell": start.get("cell"),
                    "facing": start.get("facing"),
                    "run_time": run_time,
                    "step": step,
                    "route_signature": signature,
                    "matched": False,
                }
            )
            continue
        if suffix:
            route_rows = np.flatnonzero(
                (signatures == signature)
                & (steps >= step)
                & success
            )
        else:
            route_rows = matches[:1]
        added = 0
        for row in route_rows:
            key = (str(signatures[row]), int(steps[row]))
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(int(row))
            added += 1
        log.append(
            {
                "index": index,
                "cell": start.get("cell"),
                "facing": start.get("facing"),
                "run_time": run_time,
                "step": step,
                "route_signature": signature,
                "matched": True,
                "transitions_added": added,
            }
        )
    return selected, log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expert-data", type=Path, default=None)
    parser.add_argument("--start-pool", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Repeat each matched transition in the buffer.",
    )
    parser.add_argument(
        "--suffix",
        action="store_true",
        help=(
            "Include every successful transition from the matched decision "
            "step to the end of the source route, not just the start action."
        ),
    )
    args = parser.parse_args()

    run = load_run(args.config)
    expert_path = resolve_path(
        args.expert_data or run.data.get("expert_data")
    )
    start_path = resolve_path(
        args.start_pool or run.data.get("train_pool")
    )
    if not expert_path.exists():
        raise SystemExit(f"missing expert data: {expert_path}")
    if not start_path.exists():
        raise SystemExit(f"missing start pool: {start_path}")
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")

    data = np.load(expert_path)
    starts = json.loads(start_path.read_text(encoding="utf-8"))["starts"]
    selected, log = _match_starts(
        starts,
        data,
        action_repeat=run.action_repeat,
        suffix=args.suffix,
    )
    if not selected:
        raise SystemExit("no start transitions matched the expert data")

    obs = np.asarray(data["obs"][selected], dtype=np.float32)
    next_obs = np.asarray(data["next_obs"][selected], dtype=np.float32)
    actions = np.asarray(data["actions"][selected], dtype=np.int64)
    rewards = np.asarray(data["rewards"][selected], dtype=np.float32)
    dones = np.asarray(data["dones"][selected], dtype=np.float32)
    if obs.shape[1] != 957:
        raise SystemExit(f"expected 957 expert observations, got {obs.shape}")

    count = len(selected) * int(args.repeat)
    buffer = RareReplayBuffer(
        buffer_size=count,
        observation_space=spaces.Box(
            low=-np.inf, high=np.inf, shape=(957,), dtype=np.float32
        ),
        action_space=spaces.Discrete(2),
        device="cpu",
        n_envs=1,
    )
    positions = np.repeat(np.arange(len(selected)), int(args.repeat))
    buffer.observations[:, 0] = obs[positions]
    buffer.next_observations[:, 0] = next_obs[positions]
    buffer.actions[:, 0, 0] = actions[positions]
    buffer.rewards[:, 0] = rewards[positions]
    buffer.dones[:, 0] = dones[positions]
    buffer.timeouts[:, 0] = 0.0
    buffer.bucket_ids[:, 0] = 0
    buffer.bucket_names = {"expert_starts": 0}
    buffer.priorities[:, 0] = 1.0
    buffer.returns[:, 0] = 0.0
    buffer.q_values[:, 0] = np.nan
    buffer.profit_gaps[:, 0] = 0.0
    buffer.full = True
    buffer.pos = 0
    buffer._probability_cache = None

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "config": str(run.config_path),
        "expert_data": str(expert_path),
        "start_pool": str(start_path),
        "output": str(output_path),
        "repeat": int(args.repeat),
        "suffix": bool(args.suffix),
        "starts": len(starts),
        "matched": len(selected),
        "missing": sum(1 for item in log if not item["matched"]),
        "transitions": count,
        "actions_zero": int(np.sum(actions == 0)),
        "actions_one": int(np.sum(actions == 1)),
        "route_log": log,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"start expert buffer={output_path} "
        f"matched={len(selected)}/{len(starts)} "
        f"transitions={count} metadata={metadata_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
