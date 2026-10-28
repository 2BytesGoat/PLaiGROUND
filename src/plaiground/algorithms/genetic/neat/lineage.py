#!/usr/bin/env python3
"""Carry one NEAT population through the campaign in order.

This is deliberately not an all-levels independent sweep. Stage 1 trains a
fresh population on 1-1. The stage endpoint is copied to ``snapshots/`` and the
same population is resumed on 1-2, then 1-3, and so on. Before each new level,
the current population is evaluated on that level without evolution, giving a
zero-shot transfer probe.

Per-level training stops early when the best fitness has plateaued and enough
agents finish (see ``--min-generations`` / ``--patience`` /
``--min-finish-rate``). Checkpoints are written every generation so the stage
endpoint can always be copied exactly.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

LEVELS = [f"1-{i}" for i in range(1, 17)]
REPO = Path(__file__).resolve().parents[5]
DEFAULT_CONFIG = Path(__file__).with_name("neat_1_1_config.txt")
SCRIPT = REPO / "scripts" / "genetic" / "03_neat_lineage.py"
DEFAULT_OUTPUT = REPO / "runs" / "neat_lineage"


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(path: Path, population: int) -> dict:
    rows = read_history(path)
    if not rows:
        return {
            "best_fitness": "",
            "best_time": "",
            "finishes": "",
            "generations": 0,
            "first_finish_generation": "",
            "finish_rate": "",
        }
    finish_times = [
        float(row["best_time"]) for row in rows if row.get("best_time")
    ]
    best_time = min(finish_times) if finish_times else math.inf
    last = rows[-1]
    finishes = int(float(last["finishes"]))
    first_finish = next(
        (int(row["generation"]) for row in rows if int(float(row["finishes"])) > 0),
        "",
    )
    return {
        "best_fitness": f"{max(float(row['best_fitness']) for row in rows):.2f}",
        "best_time": f"{best_time:.3f}" if math.isfinite(best_time) else "",
        "finishes": finishes,
        "generations": len(rows),
        "first_finish_generation": first_finish,
        "finish_rate": f"{finishes / population:.3f}",
    }


def checkpoint_generation(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[-1])


def latest_checkpoint(directory: Path) -> Path | None:
    checkpoints = sorted(directory.glob("neat-checkpoint-*"), key=checkpoint_generation)
    return checkpoints[-1] if checkpoints else None


BENIGN_LOG_PREFIXES = (
    "Godot Engine",
    "SW Init",
    "SW ready",
    "[Telemetry]",
    "Waiting for one second",
    "trying to connect",
    "performing handshake",
    "handshake complete",
    "resetting all agents",
    "received close",
    "Loaded environment variables",
    "No game binary has been provided",
    "waiting for remote",
    "connection established",
    "close message",
    "action space:",
    "observation space:",
    "action space [",
    "observation space [",
    "WARNING: ObjectDB instances leaked",
    "ERROR: There is no animation with name 'Celebrate'",
    "GDScript backtrace",
    "at: cleanup (",
    "at: clear (core/io/resource",
    "[0] update (res://src/ghost",
    "[1] _process (res://src/training/main_training",
)


def is_benign_log_line(line: str) -> bool:
    stripped = line.lstrip()
    if stripped.startswith(BENIGN_LOG_PREFIXES):
        return True
    if stripped.startswith("ERROR:") and "resources still in use at exit" in line:
        return True
    return False


def run_command(
    cmd: list[str],
    log_path: Path,
    label: str,
    clean: bool = True,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] {' '.join(str(part) for part in cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if not clean or not is_benign_log_line(line):
                print(line, end="", flush=True)
        return proc.wait()


def run_stage(level: str, initial_checkpoint: Path | None, args, stage_dir: Path) -> int:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--level",
        level,
        "--config",
        str(args.config),
        "--output-dir",
        str(stage_dir),
        "--generations",
        str(args.max_generations),
        "--min-generations",
        str(args.min_generations),
        "--patience",
        str(args.patience),
        "--min-finish-rate",
        str(args.min_finish_rate),
        "--min-finish-streak",
        str(args.min_finish_streak),
        "--max-steps",
        str(args.max_steps),
        "--action-repeat",
        str(args.action_repeat),
        "--speedup",
        str(args.speedup),
        "--checkpoint-interval",
        "1",
        "--port",
        str(args.port),
    ]
    if initial_checkpoint is not None:
        cmd.extend(["--resume", str(initial_checkpoint)])
    else:
        cmd.extend(["--population", str(args.population), "--seed", str(args.seed)])
    return run_command(
        cmd,
        stage_dir / "train.log",
        f"stage {level}",
        clean=args.clean_logs,
    )


def run_eval(
    level: str,
    checkpoint: Path,
    args,
    eval_dir: Path,
    label: str,
) -> int:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--evaluate",
        "--level",
        level,
        "--resume",
        str(checkpoint),
        "--max-steps",
        str(args.max_steps),
        "--action-repeat",
        str(args.action_repeat),
        "--speedup",
        str(args.eval_speedup),
        "--port",
        str(args.eval_port),
        "--output-dir",
        str(eval_dir),
    ]
    if args.show:
        cmd.append("--show")
    return run_command(
        cmd,
        eval_dir / "eval.log",
        label,
        clean=args.clean_logs,
    )


def parse_levels(value: str) -> list[str]:
    if not value or value.lower() == "all":
        return LEVELS
    levels = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [level for level in levels if level not in LEVELS]
    if unknown:
        raise ValueError(f"Unknown level(s): {', '.join(unknown)}")
    return levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="all", help="'all' or comma-separated ids")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--start-level", default="1-1")
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Population checkpoint used before --start-level.",
    )
    parser.add_argument("--max-generations", type=int, default=30)
    parser.add_argument("--min-generations", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-finish-rate", type=float, default=0.30)
    parser.add_argument("--min-finish-streak", type=int, default=3)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--eval-speedup", type=float, default=4.0)
    parser.add_argument("--port", type=int, default=11008)
    parser.add_argument("--eval-port", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the 100-agent zero-shot/stage/final evaluations in the window.",
    )
    parser.add_argument(
        "--clean-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter known-benign Godot startup/shutdown noise from stdout.",
    )
    parser.add_argument(
        "--skip-pre-eval",
        action="store_true",
        help="Skip zero-shot evaluation before each new level.",
    )
    parser.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="Skip the final all-level evaluation matrix.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing lineage output directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = parse_levels(args.levels)
    if args.start_level not in levels:
        raise ValueError(
            f"--start-level {args.start_level} is not in --levels {args.levels}"
        )
    levels = levels[levels.index(args.start_level):]
    if args.start_level != "1-1" and args.initial_checkpoint is None:
        raise ValueError(
            "--initial-checkpoint is required when --start-level is not 1-1"
        )
    if args.initial_checkpoint is not None and not args.initial_checkpoint.exists():
        raise FileNotFoundError(args.initial_checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = args.output_dir / "snapshots"
    stages_dir = args.output_dir / "stages"
    evals_dir = args.output_dir / "evals"
    final_dir = args.output_dir / "final-evals"
    for directory in (snapshots_dir, stages_dir, evals_dir, final_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.force:
        for directory in (snapshots_dir, stages_dir, evals_dir, final_dir):
            shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)

    current_checkpoint: Path | None = args.initial_checkpoint
    summary_rows: list[dict] = []
    final_checkpoint: Path | None = None

    for index, level in enumerate(levels, start=1):
        stage_dir = stages_dir / level
        stage_dir.mkdir(parents=True, exist_ok=True)

        pre = {"best_time": "", "finishes": ""}
        if current_checkpoint is not None and not args.skip_pre_eval:
            pre_dir = evals_dir / f"pre-{level}"
            if pre_dir.exists() and args.force:
                shutil.rmtree(pre_dir)
            pre_dir.mkdir(parents=True, exist_ok=True)
            code = run_eval(
                level,
                current_checkpoint,
                args,
                pre_dir,
                f"zero-shot {level}",
            )
            if code != 0:
                print(f"[{level}] zero-shot evaluation failed", flush=True)
            pre = summarize(pre_dir / "history.csv", args.population)
            print(
                f"[{level}] zero-shot before training: "
                f"finishes={pre['finishes']}/{args.population} "
                f"best={pre['best_time'] or 'none'}",
                flush=True,
            )

        print(
            f"\n=== stage {index}/{len(levels)}: {level} ===",
            flush=True,
        )
        code = run_stage(level, current_checkpoint, args, stage_dir)
        if code != 0:
            print(f"[{level}] stage training failed; stopping", flush=True)
            break

        endpoint = latest_checkpoint(stage_dir)
        if endpoint is None:
            print(f"[{level}] no checkpoint was written; stopping", flush=True)
            break

        train = summarize(stage_dir / "history.csv", args.population)
        snapshot_dir = snapshots_dir / level
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        shutil.copytree(stage_dir, snapshot_dir)
        snapshot_checkpoint = latest_checkpoint(snapshot_dir)
        if snapshot_checkpoint is None:
            print(f"[{level}] snapshot has no checkpoint; stopping", flush=True)
            break
        current_checkpoint = snapshot_checkpoint
        final_checkpoint = snapshot_checkpoint

        post_dir = evals_dir / f"post-{level}"
        if post_dir.exists() and args.force:
            shutil.rmtree(post_dir)
        post_dir.mkdir(parents=True, exist_ok=True)
        code = run_eval(
            level,
            current_checkpoint,
            args,
            post_dir,
            f"stage showcase {level}",
        )
        if code != 0:
            print(f"[{level}] stage showcase failed", flush=True)
        post = summarize(post_dir / "history.csv", args.population)

        summary_rows.append(
            {
                "stage": index,
                "level": level,
                "pre_best_time": pre["best_time"],
                "pre_finishes": pre["finishes"],
                "training_generations": train["generations"],
                "training_best_time": train["best_time"],
                "training_finishes": train["finishes"],
                "post_best_time": post["best_time"],
                "post_finishes": post["finishes"],
                "post_finish_rate": post.get("finish_rate", ""),
                "first_finish_generation": train.get("first_finish_generation", ""),
            }
        )
        print(
            f"[{level}] after training: finishes={post['finishes']}/{args.population} "
            f"best={post['best_time'] or 'none'}",
            flush=True,
        )

    if summary_rows:
        summary_path = args.output_dir / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nwrote {summary_path}", flush=True)

    if final_checkpoint is not None and not args.skip_final_eval:
        final_rows: list[dict] = []
        for level in levels:
            final_eval_dir = final_dir / level
            if final_eval_dir.exists() and args.force:
                shutil.rmtree(final_eval_dir)
            final_eval_dir.mkdir(parents=True, exist_ok=True)
            code = run_eval(
                level,
                final_checkpoint,
                args,
                final_eval_dir,
                f"final eval {level}",
            )
            if code != 0:
                print(f"[{level}] final evaluation failed", flush=True)
            result = summarize(final_eval_dir / "history.csv", args.population)
            final_rows.append(
                {
                    "level": level,
                    "best_time": result["best_time"],
                    "finishes": result["finishes"],
                    "finish_rate": result.get("finish_rate", ""),
                }
            )
        final_summary = args.output_dir / "final_summary.csv"
        with final_summary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(final_rows[0].keys()))
            writer.writeheader()
            writer.writerows(final_rows)
        print(f"wrote {final_summary}", flush=True)
        print("final evaluation matrix complete", flush=True)

    plot_path = args.output_dir / "lineage_learning.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        stages = [row["stage"] for row in summary_rows]
        pre_times = [
            float(row["pre_best_time"]) if row["pre_best_time"] else math.nan
            for row in summary_rows
        ]
        post_times = [
            float(row["post_best_time"]) if row["post_best_time"] else math.nan
            for row in summary_rows
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(stages, pre_times, marker="o", label="zero-shot before stage")
        ax.plot(stages, post_times, marker="o", label="after stage")
        ax.set_xticks(stages)
        ax.set_xticklabels([row["level"] for row in summary_rows], rotation=45)
        ax.set_ylabel("best finish time (s)")
        ax.set_title("Single-lineage NEAT curriculum")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160, bbox_inches="tight")
        print(f"wrote {plot_path}", flush=True)
    except Exception as exc:  # pragma: no cover - optional reporting
        print(f"lineage plot skipped: {exc}", flush=True)


if __name__ == "__main__":
    main()
