"""Fine-tune per-level multi-hot PPO teachers from a behavior-cloned checkpoint.

The multi-hot observation is larger and a from-scratch PPO teacher can spend a
long time relearning movement/finish behavior. This driver starts each level
from a multi-hot StudentPolicy checkpoint (usually distilled from the legacy
teacher champions), then runs the normal time-profile PPO loop with the value
head freshly initialized.

Usage:
    .venv/bin/python src/17_multi_hot_teachers.py \
        --init-student runs/multi_hot_base/student.pt --levels all
    .venv/bin/python src/17_multi_hot_teachers.py \
        --init-student runs/multi_hot_base/student.pt --levels 1-11 --timesteps 100000

Outputs:
    runs/teacher_multi_hot/<level>/           PPO run, eval, selected model
    runs/teacher_multi_hot/summary.csv       per-level status + eval row
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


LEVELS = [f"1-{i}" for i in range(1, 17)]
PILOT = Path(__file__).with_name("10_ppo_pilot.py")
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "runs" / "teacher_multi_hot"


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


def run_level(level: str, index: int, args, env_path: dict[str, str]) -> dict:
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
            "init_student": str(args.init_student),
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

    command = [
        sys.executable,
        str(PILOT),
        "--level",
        level,
        "--init-student",
        str(args.init_student),
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
        "--learning-rate",
        str(args.learning_rate),
        "--ent-coef",
        str(args.ent_coef),
        "--reward-profile",
        "time",
        "--obs-encoding",
        "multi_hot",
        "--net-arch",
        *[str(width) for width in args.net_arch],
        "--base-port",
        str(args.base_port + index * 20),
        "--eval-port",
        str(args.eval_port + index * 20),
        "--output-dir",
        str(level_dir),
        "--seed",
        str(args.seed),
    ]
    if args.skip_eval:
        command.append("--skip-eval")
    if args.dry_run:
        print(" ".join(command), flush=True)
        return {
            "level": level,
            "status": "dry-run",
            "init_student": str(args.init_student),
            "model": "",
            "log": str(log_path),
        }

    started = time.time()
    print(f"[{level}] starting from {args.init_student}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env_path,
        )
    elapsed = time.time() - started
    summary = read_summary(summary_path)
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
        "init_student": str(args.init_student),
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
    parser.add_argument(
        "--init-student",
        type=Path,
        required=True,
        help="Multi-hot StudentPolicy checkpoint used to warm-start PPO.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--eval-agents", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--min-finish-rate", type=float, default=0.90)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-port", type=int, default=11008)
    parser.add_argument("--eval-port", type=int, default=12000)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.levels = parse_levels(args.levels)
    args.output_dir = args.output_dir.resolve()
    args.init_student = args.init_student.resolve()
    if not args.init_student.exists():
        raise FileNotFoundError(args.init_student)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env_path = os.environ.copy()

    rows = []
    started = time.time()
    for index, level in enumerate(args.levels):
        rows.append(run_level(level, index, args, env_path))

    summary_path = args.output_dir / "summary.csv"
    fields = [
        "level",
        "status",
        "init_student",
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
    print(f"[multi-hot teachers] summary: {summary_path}", flush=True)
    print(f"[multi-hot teachers] wall={time.time() - started:.1f}s", flush=True)
    return 0 if all(
        row["status"] in ("ok", "existing", "dry-run") for row in rows
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
