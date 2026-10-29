#!/usr/bin/env python3
"""Build a fixed 1-4 expert replay buffer from successful NEAT routes.

The buffer is sampled separately during DDQN training for the expert TD loss
and the action-margin loss. It is not merged into the protected old-level
anchor, so the old-level rarity weighting stays intact.
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


def _success_mask(expert_path: Path, data) -> np.ndarray:
    if "route_success" in data:
        return np.asarray(data["route_success"], dtype=bool)
    metadata_path = expert_path.with_suffix(".json")
    if not metadata_path.exists():
        return np.ones(int(data["actions"].shape[0]), dtype=bool)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    successes = {
        str(item.get("route_signature"))
        for item in metadata.get("route_log", [])
        if item.get("finished")
    }
    signatures = np.asarray(data["route_signature"]).astype(str)
    return np.asarray(
        [str(signature) in successes for signature in signatures],
        dtype=bool,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expert-data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run = load_run(args.config)
    run.ensure_dirs()
    expert_value = (
        args.expert_data
        or run.data.get("expert_data")
    )
    if expert_value is None:
        raise SystemExit("missing [data] expert_data")
    expert_path = resolve_path(expert_value)
    if not expert_path.exists():
        raise SystemExit(f"missing expert data: {expert_path}")

    output_value = (
        args.output
        or run.config.get("run", {}).get("expert_buffer_from")
    )
    if output_value is None:
        raise SystemExit("missing [run] expert_buffer_from")
    output_path = resolve_path(output_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(expert_path)
    mask = _success_mask(expert_path, data)
    obs = np.asarray(data["obs"][mask], dtype=np.float32)
    next_obs = np.asarray(data["next_obs"][mask], dtype=np.float32)
    actions = np.asarray(data["actions"][mask], dtype=np.int64)
    rewards = np.asarray(data["rewards"][mask], dtype=np.float32)
    dones = np.asarray(data["dones"][mask], dtype=np.float32)
    raw = np.asarray(data["obs_raw"][mask], dtype=np.float32)
    count = int(obs.shape[0])
    if count == 0:
        raise SystemExit("no successful expert transitions")
    if obs.shape[1] != 957:
        raise SystemExit(
            f"expert observations are {obs.shape[1]}, expected 957"
        )

    observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(957,), dtype=np.float32
    )
    action_space = spaces.Discrete(2)
    buffer = RareReplayBuffer(
        buffer_size=count,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        n_envs=1,
    )
    buffer.observations[:, 0] = obs
    buffer.next_observations[:, 0] = next_obs
    buffer.actions[:, 0, 0] = actions
    buffer.rewards[:, 0] = rewards
    buffer.dones[:, 0] = dones
    buffer.timeouts[:, 0] = 0.0
    buffer.bucket_ids[:, 0] = 0
    buffer.bucket_names = {"expert_1_4": 0}
    buffer.priorities[:, 0] = 1.0
    buffer.returns[:, 0] = 0.0
    buffer.q_values[:, 0] = np.nan
    buffer.profit_gaps[:, 0] = 0.0
    buffer.full = True
    buffer.pos = 0
    buffer._probability_cache = None

    with output_path.open("wb") as handle:
        pickle.dump(buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    flags = raw[:, -3:]
    metadata = {
        "config": str(run.config_path),
        "expert_data": str(expert_path),
        "output": str(output_path),
        "transitions": count,
        "actions_zero": int(np.sum(actions == 0)),
        "actions_one": int(np.sum(actions == 1)),
        "flag_means": {
            "on_ceiling": float(flags[:, 0].mean()),
            "jump_held": float(flags[:, 1].mean()),
            "stick_active": float(flags[:, 2].mean()),
        },
        "rewards": {
            "min": float(rewards.min()),
            "max": float(rewards.max()),
            "mean": float(rewards.mean()),
            "positive_finish": int(np.sum(rewards > 400.0)),
        },
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"expert buffer={output_path} transitions={count} "
        f"zero={metadata['actions_zero']} one={metadata['actions_one']} "
        f"finish={metadata['rewards']['positive_finish']} "
        f"metadata={metadata_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
