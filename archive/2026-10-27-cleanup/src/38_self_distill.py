"""Collect successful stochastic rollouts and behavior-clone the policy.

Why this exists: a PPO policy can finish a level while sampling actions but
fail when evaluated with the deterministic argmax action. This script collects
successful sampled trajectories, keeps the fastest fraction of each start,
weights every kept episode equally, and fine-tunes the action distribution with
cross-entropy. It is filtered self-distillation / behavior cloning, not an
external teacher.

The game is still driven through the normal training bridge; no game logic or
shipped WATCH AI runtime is changed. 794-value sensor observations are kept.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


def load_pilot_module():
    spec = importlib.util.spec_from_file_location(
        "ppo_pilot", Path(__file__).with_name("10_ppo_pilot.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--level", default="1-3")
    parser.add_argument(
        "--starts",
        type=Path,
        default=None,
        help="Optional start-pool/strata JSON. Default: canonical start only.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Successful rollouts to collect before filtering (single start).",
    )
    parser.add_argument(
        "--episodes-per-start",
        type=int,
        default=0,
        help="When >0, collect this many successes for every assigned start.",
    )
    parser.add_argument("--max-attempts", type=int, default=250)
    parser.add_argument(
        "--max-attempts-per-start",
        type=int,
        default=500,
        help="Per-start attempt cap when --episodes-per-start is set.",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=1,
        help="Split the start list into this many shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard to collect.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Write the filtered shard dataset without running BC.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=6.0,
        help="Collect successful runs at or below this finish time.",
    )
    parser.add_argument(
        "--keep-quantile",
        type=float,
        default=0.25,
        help="Fraction of each start's successful runs to keep, fastest first.",
    )
    parser.add_argument(
        "--min-keep",
        type=int,
        default=1,
        help="Keep at least this many runs per start.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=12010)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def make_start_pool(module, path: Path | None) -> list[dict]:
    if path is None:
        return [{"kind": "canonical", "canonical": True, "weight": 1.0}]
    starts = module.load_start_pool(path.expanduser().resolve())
    for item in starts:
        if "kind" not in item and item.get("stratum"):
            item["kind"] = str(item["stratum"])
    return starts


def _start_key(info: dict) -> tuple:
    cell = info.get("start_cell") or ()
    if isinstance(cell, list):
        cell = tuple(cell)
    return (
        str(info.get("level", "")),
        str(info.get("start_kind") or info.get("start_source") or "unknown"),
        cell,
        int(info.get("start_facing") or 0),
        int(info.get("prefix_len") or 0),
    )


def _spec_key(level: str, spec: dict) -> tuple:
    cell = spec.get("cell") or ()
    if isinstance(cell, list):
        cell = tuple(cell)
    kind = (
        "prefix"
        if spec.get("prefix_actions")
        else str(spec.get("kind") or ("canonical" if spec.get("canonical") else "cell"))
    )
    return (
        level,
        kind,
        cell,
        int(spec.get("facing") or 0),
        len(spec.get("prefix_actions") or []),
    )


def collect(
    module,
    model: PPO,
    args: argparse.Namespace,
    starts: list[dict],
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    env = module.SingleAgentGodotEnv(
        env_path=module.resolve_env_path(),
        level=args.level,
        port=args.port,
        seed=args.seed,
        action_repeat=args.action_repeat,
        speedup=args.speedup,
        max_decisions=args.max_decisions,
        show_window=False,
        log_path=str(output_dir / "godot" / "collect.log"),
        child_env=module.isolated_home(output_dir, "collect"),
        force_start_jump=True,
        reward_profile="time",
        obs_encoding="multi_hot_sensor",
        start_pool=starts,
        start_seed=args.seed,
    )
    episodes: list[dict] = []
    attempts = 0
    counts: dict[tuple, int] = defaultdict(int)

    def run_episode() -> tuple[list[tuple[np.ndarray, int]], bool, float, dict]:
        obs, _info = env.reset()
        trajectory: list[tuple[np.ndarray, int]] = []
        done = False
        final_info: dict = {}
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            action_int = int(np.asarray(action).reshape(-1)[0])
            next_obs, _reward, done, _truncated, final_info = env.step(
                action_int
            )
            trajectory.append(
                (np.asarray(obs, dtype=np.float32).copy(), action_int)
            )
            obs = next_obs
        finished = bool(final_info.get("finished", False))
        run_time = final_info.get("run_time", float("inf"))
        try:
            run_time = float(run_time)
        except (TypeError, ValueError):
            run_time = float("inf")
        return trajectory, finished, run_time, final_info

    try:
        if args.episodes_per_start > 0:
            # Work through assigned starts one at a time. This prevents easy
            # starts from being oversampled while a rare start is still trying
            # to reach its target.
            for spec in starts:
                if attempts >= args.max_attempts:
                    break
                spec_key = _spec_key(args.level, spec)
                env.start_pool = [dict(spec)]
                env.start_weights = np.asarray([1.0], dtype=np.float64)
                start_attempts = 0
                while (
                    counts[spec_key] < args.episodes_per_start
                    and start_attempts < args.max_attempts_per_start
                    and attempts < args.max_attempts
                ):
                    attempts += 1
                    start_attempts += 1
                    trajectory, finished, run_time, final_info = run_episode()
                    if (
                        finished
                        and np.isfinite(run_time)
                        and run_time <= args.max_time
                    ):
                        key = _start_key(final_info)
                        counts[key] += 1
                        episodes.append(
                            {
                                "key": key,
                                "time": run_time,
                                "trajectory": trajectory,
                            }
                        )
                        print(
                            f"[collect] {counts[key]}/"
                            f"{args.episodes_per_start} key={key} "
                            f"attempt={attempts} time={run_time:.3f}s "
                            f"steps={len(trajectory)}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[collect] reject key={spec_key} "
                            f"attempt={attempts} finished={finished} "
                            f"time={run_time}",
                            flush=True,
                        )
                if counts[spec_key] < args.episodes_per_start:
                    print(
                        f"[collect] target missed key={spec_key} "
                        f"{counts[spec_key]}/{args.episodes_per_start} "
                        f"attempts={start_attempts}",
                        flush=True,
                    )
        else:
            while len(episodes) < args.episodes and attempts < args.max_attempts:
                attempts += 1
                trajectory, finished, run_time, final_info = run_episode()
                if (
                    finished
                    and np.isfinite(run_time)
                    and run_time <= args.max_time
                ):
                    key = _start_key(final_info)
                    counts[key] += 1
                    episodes.append(
                        {
                            "key": key,
                            "time": run_time,
                            "trajectory": trajectory,
                        }
                    )
                    print(
                        f"[collect] {len(episodes)}/{args.episodes} "
                        f"key={key} attempt={attempts} time={run_time:.3f}s "
                        f"steps={len(trajectory)}",
                        flush=True,
                    )
                else:
                    print(
                        f"[collect] reject attempt={attempts} "
                        f"finished={finished} time={run_time}",
                        flush=True,
                    )
    finally:
        env.close()

    if not episodes:
        raise RuntimeError("no successful trajectories collected")

    by_start: dict[tuple, list[dict]] = defaultdict(list)
    for episode in episodes:
        by_start[episode["key"]].append(episode)

    selected: list[dict] = []
    strata_stats: dict[str, dict] = {}
    for key, group in sorted(by_start.items(), key=lambda item: str(item[0])):
        group.sort(key=lambda episode: episode["time"])
        keep = max(args.min_keep, int(np.ceil(len(group) * args.keep_quantile)))
        keep = min(keep, len(group))
        selected.extend(group[:keep])
        strata_stats[str(key)] = {
            "collected": len(group),
            "kept": keep,
            "cutoff_s": float(group[keep - 1]["time"]),
            "best_s": float(group[0]["time"]),
            "worst_s": float(group[-1]["time"]),
        }
        print(
            f"[filter] {key} collected={len(group)} keep={keep} "
            f"best={group[0]['time']:.3f}s cutoff={group[keep - 1]['time']:.3f}s",
            flush=True,
        )

    observations: list[np.ndarray] = []
    actions: list[int] = []
    weights: list[float] = []
    kept_times: list[float] = []
    for episode in selected:
        trajectory = episode["trajectory"]
        # Equal total weight per episode, not per timestep. A slow trajectory
        # has more decisions and would otherwise dominate the BC loss.
        weight = 1.0 / max(1, len(trajectory))
        kept_times.append(float(episode["time"]))
        for observation, action_int in trajectory:
            observations.append(observation)
            actions.append(action_int)
            weights.append(weight)

    weight_array = np.asarray(weights, dtype=np.float32)
    stats = {
        "attempts": attempts,
        "successes": len(episodes),
        "kept_episodes": len(selected),
        "kept_observations": len(observations),
        "keep_quantile": args.keep_quantile,
        "mean_time": float(np.mean(kept_times)),
        "median_time": float(np.median(kept_times)),
        "min_time": float(np.min(kept_times)),
        "max_time": float(np.max(kept_times)),
        "strata": strata_stats,
    }
    return (
        np.stack(observations).astype(np.float32),
        np.asarray(actions, dtype=np.int64),
        weight_array,
        stats,
    )


def train_bc(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    sample_weights: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    device = next(model.policy.parameters()).device
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
    weight_tensor = torch.as_tensor(
        sample_weights, dtype=torch.float32, device=device
    )
    weight_tensor = weight_tensor / weight_tensor.mean()
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.learning_rate)
    count = len(observations)
    indices = np.arange(count)
    rng = np.random.default_rng(args.seed)
    last_loss = float("nan")
    last_accuracy = float("nan")
    model.policy.train()
    for epoch in range(args.epochs):
        rng.shuffle(indices)
        losses: list[float] = []
        correct = 0
        weighted = 0.0
        for start in range(0, count, args.batch_size):
            batch = indices[start : start + args.batch_size]
            batch_obs = obs_tensor[batch]
            batch_actions = action_tensor[batch]
            batch_weights = weight_tensor[batch]
            distribution = model.policy.get_distribution(batch_obs)
            per_sample = -distribution.log_prob(batch_actions)
            loss = (per_sample * batch_weights).sum() / batch_weights.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            predicted = distribution.distribution.probs.argmax(dim=1)
            correct += int(
                ((predicted == batch_actions) * batch_weights).sum().item()
            )
            weighted += float(batch_weights.sum().item())
        last_loss = float(np.mean(losses))
        last_accuracy = correct / max(1e-9, weighted)
        print(
            f"[bc] epoch={epoch + 1}/{args.epochs} "
            f"loss={last_loss:.4f} weighted_action_acc={last_accuracy:.3f}",
            flush=True,
        )
    return {"loss": last_loss, "weighted_action_accuracy": last_accuracy}


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    (args.output_dir / "godot").mkdir(parents=True, exist_ok=True)
    module = load_pilot_module()
    starts = make_start_pool(module, args.starts)
    if args.shards < 1:
        raise ValueError("--shards must be >= 1")
    if not 0 <= args.shard_index < args.shards:
        raise ValueError("--shard-index must be in [0, shards)")
    if args.shards > 1:
        starts = [
            spec
            for index, spec in enumerate(starts)
            if index % args.shards == args.shard_index
        ]
    if not starts:
        raise ValueError("shard has no starts")
    model = PPO.load(str(args.checkpoint), device="cpu")
    print(
        f"loaded {args.checkpoint} for level={args.level} "
        f"starts={len(starts)} shard={args.shard_index}/{args.shards} "
        f"per_start={args.episodes_per_start or args.episodes} "
        f"keep={args.keep_quantile:.0%} max_time={args.max_time}",
        flush=True,
    )
    started = time.time()
    observations, actions, sample_weights, collect_stats = collect(
        module, model, args, starts, args.output_dir
    )
    np.savez_compressed(
        args.output_dir / "successful_rollouts.npz",
        observations=observations,
        actions=actions,
        sample_weights=sample_weights,
    )
    print(
        f"[collect] stats {json.dumps(collect_stats, sort_keys=True)}",
        flush=True,
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "level": args.level,
        "starts": len(starts),
        "shard_index": args.shard_index,
        "shards": args.shards,
        "observations": int(len(observations)),
        "actions": int(len(actions)),
        "collect": collect_stats,
        "elapsed_s": round(time.time() - started, 1),
    }
    if args.collect_only:
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(
            f"wrote {args.output_dir / 'successful_rollouts.npz'}",
            flush=True,
        )
        return
    bc_stats = train_bc(
        model, observations, actions, sample_weights, args
    )
    model.save(str(args.output_dir / "distilled"))
    summary["bc"] = bc_stats
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output_dir / 'distilled.zip'}", flush=True)


if __name__ == "__main__":
    main()
