"""Merge sharded self-distillation datasets and run one BC pass."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.shard_dir = args.shard_dir.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.shard_dir.glob("shard_*/successful_rollouts.npz"))
    if not paths:
        raise FileNotFoundError("no shard_*/successful_rollouts.npz found")
    observations = []
    actions = []
    weights = []
    for path in paths:
        data = np.load(path)
        observations.append(data["observations"])
        actions.append(data["actions"])
        weights.append(data["sample_weights"])
        print(
            f"[merge] {path.parent.name} obs={len(data['observations'])}",
            flush=True,
        )
    obs = np.concatenate(observations, axis=0)
    acts = np.concatenate(actions, axis=0)
    sample_weights = np.concatenate(weights, axis=0)
    print(f"[merge] total obs={len(obs)}", flush=True)

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
        "shards": len(paths),
        "observations": int(len(obs)),
        "actions": int(len(acts)),
        "bc": bc_stats,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output_dir / 'distilled.zip'}", flush=True)


if __name__ == "__main__":
    main()
