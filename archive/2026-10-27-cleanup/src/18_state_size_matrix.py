"""Run the controlled 1-11 state x width PPO matrix.

This is the follow-up to the multi-hot teacher pilot. It keeps the reward,
cadence, PPO budget, evaluation protocol, and seed set fixed while varying only:

    observation state: legacy (58) vs multi_hot (754)
    network width:     64x64 vs 256x256

It deliberately starts from random PPO weights, not a behavior clone, so the
result answers whether the current PPO recipe can train 1-11 from each state.

Usage:
    .venv/bin/python src/18_state_size_matrix.py
    .venv/bin/python src/18_state_size_matrix.py --seeds 42,7 --timesteps 300000
    .venv/bin/python src/18_state_size_matrix.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


PILOT = Path(__file__).with_name("10_ppo_pilot.py")
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "runs" / "ab_1-11"


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def run_one(
    state: str,
    width: int,
    seed: int,
    index: int,
    args: argparse.Namespace,
    child_env: dict[str, str],
) -> dict:
    name = f"{state}_w{width}_s{seed}"
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    summary_path = run_dir / "eval" / "summary.csv"
    if (
        args.skip_existing
        and (run_dir / "ppo_pilot.zip").exists()
        and summary_path.exists()
    ):
        summary = read_summary(summary_path)
        print(
            f"[{name}] existing best={summary.get('best_time', '?')} "
            f"median={summary.get('median_time', '?')}",
            flush=True,
        )
        return {
            "state": state,
            "width": width,
            "seed": seed,
            "status": "existing",
            "model": str(run_dir / "ppo_pilot.zip"),
            "best_time": summary.get("best_time", ""),
            "median_time": summary.get("median_time", ""),
            "finishes": summary.get("finishes", ""),
            "deaths": summary.get("deaths", ""),
            "timeouts": summary.get("timeouts", ""),
            "efficiency_mean": summary.get("efficiency_mean", ""),
            "flow_mean": summary.get("flow_mean", ""),
            "log": str(log_path),
        }

    # patience * eval_every exceeds the budget, so every arm gets the full
    # 300k timesteps. This is intentional: early stopping would make the arms
    # incomparable.
    patience = max(1, (args.timesteps + args.eval_every - 1) // args.eval_every + 1)
    command = [
        sys.executable,
        str(PILOT),
        "--level",
        "1-11",
        "--n-envs",
        str(args.n_envs),
        "--timesteps",
        str(args.timesteps),
        "--max-decisions",
        "150",
        "--action-repeat",
        "4",
        "--speedup",
        str(args.speedup),
        "--reward-profile",
        "time",
        "--obs-encoding",
        state,
        "--net-arch",
        str(width),
        str(width),
        "--eval-agents",
        "100",
        "--eval-every",
        str(args.eval_every),
        "--patience",
        str(patience),
        "--min-finish-rate",
        "0.90",
        "--seed",
        str(seed),
        "--base-port",
        str(args.base_port + index * 20),
        "--eval-port",
        str(args.eval_port + index * 20),
        "--output-dir",
        str(run_dir),
    ]
    if args.dry_run:
        print(" ".join(command), flush=True)
        return {
            "state": state,
            "width": width,
            "seed": seed,
            "status": "dry-run",
            "model": "",
            "log": str(log_path),
        }

    started = time.time()
    print(f"[{name}] starting", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=child_env,
        )
    elapsed = time.time() - started
    summary = read_summary(summary_path)
    model = run_dir / "ppo_pilot.zip"
    status = "ok" if completed.returncode == 0 and model.exists() else "failed"
    print(
        f"[{name}] {status} in {elapsed:.1f}s "
        f"best={summary.get('best_time', '?')} "
        f"median={summary.get('median_time', '?')} "
        f"finishes={summary.get('finishes', '?')}",
        flush=True,
    )
    return {
        "state": state,
        "width": width,
        "seed": seed,
        "status": status,
        "model": str(model) if model.exists() else "",
        "best_time": summary.get("best_time", ""),
        "median_time": summary.get("median_time", ""),
        "finishes": summary.get("finishes", ""),
        "deaths": summary.get("deaths", ""),
        "timeouts": summary.get("timeouts", ""),
        "efficiency_mean": summary.get("efficiency_mean", ""),
        "flow_mean": summary.get("flow_mean", ""),
        "wall_s": f"{elapsed:.1f}",
        "log": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", default="legacy,multi_hot")
    parser.add_argument("--widths", default="64,256")
    parser.add_argument("--seeds", default="42,7")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--base-port", type=int, default=11008)
    parser.add_argument("--eval-port", type=int, default=12000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    states = parse_csv(args.states)
    widths = [int(value) for value in parse_csv(args.widths)]
    seeds = [int(value) for value in parse_csv(args.seeds)]
    unknown_states = [state for state in states if state not in ("legacy", "multi_hot")]
    if unknown_states:
        raise ValueError(f"Unknown state(s): {unknown_states}")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()

    combos = [
        (state, width, seed)
        for state in states
        for width in widths
        for seed in seeds
    ]
    rows = []
    started = time.time()
    for index, (state, width, seed) in enumerate(combos):
        rows.append(run_one(state, width, seed, index, args, child_env))

    summary_path = args.output_dir / "matrix_summary.csv"
    fields = [
        "state",
        "width",
        "seed",
        "status",
        "model",
        "best_time",
        "median_time",
        "finishes",
        "deaths",
        "timeouts",
        "efficiency_mean",
        "flow_mean",
        "wall_s",
        "log",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    print(f"[matrix] summary: {summary_path}", flush=True)
    print(f"[matrix] wall={time.time() - started:.1f}s", flush=True)
    return 0 if all(
        row["status"] in ("ok", "existing", "dry-run") for row in rows
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
