"""Train filtered BC from successful trajectories logged during PPO.

`10_ppo_pilot.py --trajectory-dir ...` writes one .npz per successful fast
episode. This script groups those episodes by start, keeps the fastest fraction
of each start, weights episodes equally, and behavior-clones the policy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from stable_baselines3 import PPO


def load_distill_module():
    spec = importlib.util.spec_from_file_location(
        "self_distill", Path(__file__).with_name("38_self_distill.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--level", default=None)
    parser.add_argument("--keep-quantile", type=float, default=0.25)
    parser.add_argument("--min-keep", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def start_key(payload) -> tuple:
    cell = payload.get("start_cell", np.asarray([], dtype=np.int64))
    cell = tuple(int(v) for v in np.asarray(cell).reshape(-1))
    return (
        str(payload.get("level", "")),
        str(payload.get("start_kind", "")),
        cell,
        int(payload.get("start_facing", 0)),
        int(payload.get("prefix_len", 0)),
    )


def main() -> None:
    args = parse_args()
    args.trajectory_dir = args.trajectory_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.trajectory_dir.rglob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError("no episode_*.npz trajectories found")

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for path in paths:
        payload = np.load(path)
        level = str(payload.get("level", ""))
        if args.level and level != args.level:
            continue
        grouped[start_key(payload)].append(
            {
                "path": path,
                "observations": payload["observations"],
                "actions": payload["actions"],
                "run_time": float(payload.get("run_time", np.inf)),
            }
        )

    selected: list[dict] = []
    stats: dict[str, dict] = {}
    for key, episodes in sorted(grouped.items(), key=lambda item: str(item[0])):
        episodes.sort(key=lambda item: item["run_time"])
        keep = max(
            args.min_keep, int(np.ceil(len(episodes) * args.keep_quantile))
        )
        keep = min(keep, len(episodes))
        selected.extend(episodes[:keep])
        stats[str(key)] = {
            "episodes": len(episodes),
            "kept": keep,
            "cutoff_s": float(episodes[keep - 1]["run_time"]),
            "best_s": float(episodes[0]["run_time"]),
        }
        print(
            f"[filter] {key} episodes={len(episodes)} keep={keep} "
            f"best={episodes[0]['run_time']:.3f}s "
            f"cutoff={episodes[keep - 1]['run_time']:.3f}s",
            flush=True,
        )

    observations = []
    actions = []
    weights = []
    for episode in selected:
        weight = 1.0 / max(1, len(episode["observations"]))
        observations.append(episode["observations"])
        actions.append(episode["actions"])
        weights.extend([weight] * len(episode["observations"]))
    obs = np.concatenate(observations, axis=0).astype(np.float32)
    acts = np.concatenate(actions, axis=0).astype(np.int64)
    sample_weights = np.asarray(weights, dtype=np.float32)
    print(
        f"[merge] episodes={len(selected)} observations={len(obs)}",
        flush=True,
    )

    distill = load_distill_module()
    model = PPO.load(str(args.checkpoint), device="cpu")
    train_args = SimpleNamespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    bc_stats = distill.train_bc(model, obs, acts, sample_weights, train_args)
    model.save(str(args.output_dir / "distilled"))
    summary = {
        "checkpoint": str(args.checkpoint),
        "trajectories": len(paths),
        "selected_episodes": len(selected),
        "observations": int(len(obs)),
        "collect": stats,
        "bc": bc_stats,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output_dir / 'distilled.zip'}", flush=True)


if __name__ == "__main__":
    main()
