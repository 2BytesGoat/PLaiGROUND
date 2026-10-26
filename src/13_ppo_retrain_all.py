"""Retrain every campaign teacher with the time profile and eval patience.

This is a driver around ``10_ppo_pilot.py``; it does not implement a new
optimizer. Each level starts from the current post-fix teacher and runs PPO
with the training-only ``--reward-profile=time``. The pilot stops when the
deterministic median time stops improving, or at the hard additional-step cap.

Usage:
    .venv/bin/python src/13_ppo_retrain_all.py --levels all
    .venv/bin/python src/13_ppo_retrain_all.py --levels 1-11 --timesteps 50000
    .venv/bin/python src/13_ppo_retrain_all.py --levels all --dry-run

Outputs:
    runs/teacher_time/<level>/                  PPO run, checkpoints, eval
    runs/teacher_time/<level>/ppo_pilot.zip     selected model
    runs/teacher_time/summary.csv               per-level status + eval row
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


LEVELS = [f"1-{i}" for i in range(1, 17)]
# Keep this in sync with 12_ppo_distill.py.
TEACHER_ROOTS = (
    Path("runs/teacher_champions"),
    Path("runs/teacher_repair/final"),
    Path("runs/teacher_repair_post"),
    Path("runs/ppo_lineage/snapshots"),
    Path("runs/ppo_lineage_cont/snapshots"),
)
PILOT = Path(__file__).with_name("10_ppo_pilot.py")
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "runs" / "teacher_time"


def parse_levels(value: str) -> list[str]:
    if not value or value.lower() == "all":
        return list(LEVELS)
    levels = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [level for level in levels if level not in LEVELS]
    if unknown:
        raise ValueError(f"Unknown level(s): {', '.join(unknown)}")
    return levels


def teacher_path(level: str) -> Path | None:
    for root in TEACHER_ROOTS:
        candidate = root / level / "ppo_pilot.zip"
        if candidate.exists():
            return candidate
    return None


def read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def run_level(level: str, index: int, args, env_path: str) -> dict:
    level_dir = args.output_dir / level
    level_dir.mkdir(parents=True, exist_ok=True)
    log_path = level_dir / "train.log"
    summary_path = level_dir / "eval" / "summary.csv"
    if (
        args.skip_existing
        and (level_dir / "ppo_pilot.zip").exists()
        and summary_path.exists()
    ):
        summary = read_summary(summary_path)
        print(
            f"[{level}] existing best={summary.get('best_time', '?')} "
            f"median={summary.get('median_time', '?')}",
            flush=True,
        )
        return {
            "level": level,
            "status": "existing",
            "teacher": "",
            "model": str(level_dir / "ppo_pilot.zip"),
            "best_time": summary.get("best_time", ""),
            "median_time": summary.get("median_time", ""),
            "finishes": summary.get("finishes", ""),
            "deaths": summary.get("deaths", ""),
            "timeouts": summary.get("timeouts", ""),
            "efficiency_mean": summary.get("efficiency_mean", ""),
            "flow_mean": summary.get("flow_mean", ""),
            "log": str(log_path),
        }
    teacher = teacher_path(level)
    if teacher is None:
        return {
            "level": level,
            "status": "no-teacher",
            "teacher": "",
            "model": "",
            "log": str(log_path),
        }

    command = [
        sys.executable,
        str(PILOT),
        "--level",
        level,
        "--resume",
        str(teacher),
        "--timesteps",
        str(args.timesteps),
        "--n-envs",
        str(args.n_envs),
        "--max-decisions",
        str(args.max_decisions),
        "--action-repeat",
        str(args.action_repeat),
        "--speedup",
        str(args.speedup),
        "--eval-agents",
        str(args.eval_agents),
        "--eval-every",
        str(args.eval_every),
        "--patience",
        str(args.patience),
        "--min-improvement",
        str(args.min_improvement),
        "--min-finish-rate",
        str(args.min_finish_rate),
        "--reward-profile",
        "time",
        "--obs-encoding",
        str(args.obs_encoding),
        "--base-port",
        str(args.base_port + index * 20),
        "--eval-port",
        str(args.eval_port + index * 20),
        "--output-dir",
        str(level_dir),
        "--seed",
        str(args.seed),
    ]
    if args.show:
        command.append("--show")
    if args.skip_eval:
        command.append("--skip-eval")
    if args.dry_run:
        print(" ".join(command), flush=True)
        return {
            "level": level,
            "status": "dry-run",
            "teacher": str(teacher),
            "model": "",
            "log": str(log_path),
        }

    started = time.time()
    print(f"[{level}] starting from {teacher}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env_path,
        )
    elapsed = time.time() - started
    summary = read_summary(level_dir / "eval" / "summary.csv")
    model = level_dir / "ppo_pilot.zip"
    status = "ok" if completed.returncode == 0 and model.exists() else "failed"
    print(
        f"[{level}] {status} in {elapsed:.1f}s "
        f"best={summary.get('best_time', '?')} "
        f"median={summary.get('median_time', '?')} "
        f"finishes={summary.get('finishes', '?')}",
        flush=True,
    )
    return {
        "level": level,
        "status": status,
        "teacher": str(teacher),
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
    parser.add_argument("--levels", default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Additional PPO timesteps per teacher (resume offset is separate).",
    )
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--eval-agents", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--min-finish-rate", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-port", type=int, default=11008)
    parser.add_argument("--eval-port", type=int, default=12000)
    parser.add_argument(
        "--obs-encoding",
        default="legacy",
        choices=("legacy", "multi_hot"),
        help="Observation contract; teachers and student must match.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse levels that already have ppo_pilot.zip and eval/summary.csv.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.levels = parse_levels(args.levels)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The game process isolation is done by 10_ppo_pilot; the driver only needs
    # a clean inherited environment. Keep this hook explicit for future worker
    # execution without changing the current sequential behavior.
    import os

    env_path = os.environ.copy()

    rows = []
    started = time.time()
    for index, level in enumerate(args.levels):
        rows.append(run_level(level, index, args, env_path))

    summary_path = args.output_dir / "summary.csv"
    fields = [
        "level",
        "status",
        "teacher",
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
    print(f"[retrain] summary: {summary_path}", flush=True)
    print(f"[retrain] wall={time.time() - started:.1f}s", flush=True)
    return 0 if all(
        row["status"] in ("ok", "existing", "dry-run", "no-teacher")
        for row in rows
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
