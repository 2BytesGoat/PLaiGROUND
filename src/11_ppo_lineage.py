#!/usr/bin/env python3
"""Carry one PPO policy through the campaign in order.

This is the PPO counterpart to 09_neat_lineage.py: one model, one stage at a
time, 1-1 -> 1-16. Each stage zero-shot evaluates the current policy, trains
the same weights on the new level, snapshots the result, and moves on.

Training is deliberately sequential. Parallel level environments are not used
here because they change the experiment: they mix level gradients at every
update instead of measuring transfer from one level to the next.

Usage:
    python src/11_ppo_lineage.py --levels all --output-dir runs/ppo_lineage
    python src/11_ppo_lineage.py --start-level 1-3 \
        --initial-model runs/ppo_1-3/ppo_pilot.zip \
        --output-dir runs/ppo_lineage_from_1-3

Each stage calls 10_ppo_pilot.py as a subprocess. The stage model is copied to
snapshots/<level>/; the stage's eval/summary.csv contains the 100-agent
deterministic result.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

LEVELS = [f"1-{i}" for i in range(1, 17)]
SCRIPT = Path(__file__).with_name("10_ppo_pilot.py")
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "runs" / "ppo_lineage"


def parse_levels(value: str) -> list[str]:
    if not value or value.lower() == "all":
        return list(LEVELS)
    levels = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [level for level in levels if level not in LEVELS]
    if unknown:
        raise ValueError(f"Unknown level(s): {', '.join(unknown)}")
    return levels


def read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def run_command(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            cmd,
            cwd=SCRIPT.parent.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}); see {log_path}"
        )


def stage_args(args: argparse.Namespace, level: str, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--level",
        level,
        "--n-envs",
        str(args.n_envs),
        "--max-decisions",
        str(args.max_decisions),
        "--action-repeat",
        str(args.action_repeat),
        "--speedup",
        str(args.speedup),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--net-arch",
        *[str(v) for v in args.net_arch],
        "--save-freq",
        str(args.save_freq),
        "--seed",
        str(args.seed),
        "--base-port",
        str(args.base_port),
        "--eval-port",
        str(args.eval_port),
        "--eval-agents",
        str(args.eval_agents),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--verbose",
        str(args.verbose),
    ]
    if args.show:
        cmd.append("--show")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="all")
    parser.add_argument("--start-level", default="1-1")
    parser.add_argument(
        "--initial-model",
        type=Path,
        help="Model state before --start-level (required when not starting at 1-1).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--chunk-timesteps", type=int, default=100_000)
    parser.add_argument("--max-timesteps", type=int, default=200_000)
    parser.add_argument("--min-finish-rate", type=float, default=0.30)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--eval-agents", type=int, default=100)
    parser.add_argument("--base-port", type=int, default=11008)
    parser.add_argument("--eval-port", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Continue to later levels even when the finish-rate gate fails.",
    )
    parser.add_argument(
        "--final-eval",
        action="store_true",
        help="Run a final 100-agent evaluation on every level.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    levels = parse_levels(args.levels)
    if args.start_level not in levels:
        raise ValueError(f"--start-level {args.start_level} not in --levels")
    levels = levels[levels.index(args.start_level):]
    if args.start_level != "1-1" and args.initial_model is None:
        raise ValueError("--initial-model is required when not starting at 1-1")
    if args.initial_model is not None and not args.initial_model.exists():
        raise FileNotFoundError(args.initial_model)

    args.output_dir = args.output_dir.resolve()
    if args.force:
        shutil.rmtree(args.output_dir, ignore_errors=True)
    stages_dir = args.output_dir / "stages"
    zero_dir = args.output_dir / "zero-shot"
    snapshots_dir = args.output_dir / "snapshots"
    for directory in (stages_dir, zero_dir, snapshots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "summary.csv"
    summary_rows: list[dict] = []
    current_model = args.initial_model

    def write_summary() -> None:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "level",
                    "zero_shot_finishes",
                    "zero_shot_best",
                    "trained_timesteps",
                    "post_finishes",
                    "post_best",
                    "gate",
                    "snapshot",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    for index, level in enumerate(levels, start=1):
        print(f"\n=== stage {index}/{len(levels)}: {level} ===", flush=True)

        zero = {}
        if current_model is not None:
            zero_stage = zero_dir / level
            zero_cmd = stage_args(args, level, zero_stage)
            zero_cmd += ["--eval-only", "--resume", str(current_model)]
            run_command(zero_cmd, zero_stage / "eval.log")
            zero = read_summary(zero_stage / "eval" / "summary.csv")
            print(
                f"[{level}] zero-shot finishes="
                f"{zero.get('finishes', '?')}/{args.eval_agents} "
                f"best={zero.get('best_time') or 'none'}",
                flush=True,
            )

        stage_dir = stages_dir / level
        trained = 0
        post = {}
        while True:
            train_cmd = stage_args(args, level, stage_dir)
            train_cmd += ["--timesteps", str(args.chunk_timesteps)]
            if current_model is not None:
                train_cmd += [
                    "--resume",
                    str(current_model),
                    "--additional-timesteps",
                ]
            run_command(train_cmd, stage_dir / "train.log")
            trained += args.chunk_timesteps
            post = read_summary(stage_dir / "eval" / "summary.csv")
            rate = int(post.get("finishes", 0)) / args.eval_agents
            print(
                f"[{level}] trained={trained} finishes="
                f"{post.get('finishes', '?')}/{args.eval_agents} "
                f"best={post.get('best_time') or 'none'}",
                flush=True,
            )
            if rate >= args.min_finish_rate or trained >= args.max_timesteps:
                break
            current_model = stage_dir / "ppo_pilot.zip"

        current_model = stage_dir / "ppo_pilot.zip"
        if not current_model.exists():
            raise FileNotFoundError(current_model)

        snapshot_dir = snapshots_dir / level
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_dir / "ppo_pilot.zip"
        shutil.copy2(current_model, snapshot)
        print(f"[{level}] snapshot: {snapshot}", flush=True)

        rate = int(post.get("finishes", 0)) / args.eval_agents
        passed = rate >= args.min_finish_rate
        summary_rows.append(
            {
                "level": level,
                "zero_shot_finishes": zero.get("finishes", ""),
                "zero_shot_best": zero.get("best_time", ""),
                "trained_timesteps": trained,
                "post_finishes": post.get("finishes", ""),
                "post_best": post.get("best_time", ""),
                "gate": "pass" if passed else "fail",
                "snapshot": str(snapshot),
            }
        )
        write_summary()

        if not passed and not args.continue_on_fail:
            print(
                f"[{level}] gate failed; stopping. "
                "Use --continue-on-fail to keep training later levels.",
                flush=True,
            )
            break

    write_summary()
    print(f"\nsummary: {summary_path}", flush=True)

    if args.final_eval:
        final_dir = args.output_dir / "final-eval"
        final_cmd = [
            sys.executable,
            str(SCRIPT),
            "--levels",
            "all",
            "--eval-only",
            "--resume",
            str(current_model),
            "--max-decisions",
            str(args.max_decisions),
            "--action-repeat",
            str(args.action_repeat),
            "--speedup",
            str(args.speedup),
            "--eval-agents",
            str(args.eval_agents),
            "--eval-port",
            str(args.eval_port),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(final_dir),
            "--device",
            args.device,
        ]
        if args.show:
            final_cmd.append("--show")
        run_command(final_cmd, final_dir / "eval.log")


if __name__ == "__main__":
    main()
