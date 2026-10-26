"""Multi-seed PPO restarts after a teacher plateaus.

Each seed starts from the current champion/teacher and runs the time profile
with evaluation patience. The best deterministic eval across seeds becomes the
level's candidate. This is multi-start optimization, not random-state training:
the level itself is deterministic, so a new seed changes PPO exploration, not
the spawn.

Usage:
    .venv/bin/python src/16_ppo_seed_restarts.py \
      --levels 1-7,1-10,1-11 --seeds 42,43,44 --timesteps 25000

    .venv/bin/python src/16_ppo_seed_restarts.py --levels all --seeds 42,43
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path


LEVELS = [f"1-{i}" for i in range(1, 17)]
TEACHER_ROOTS = (
    Path("runs/teacher_champions"),
    Path("runs/teacher_repair/final"),
    Path("runs/teacher_repair_post"),
    Path("runs/ppo_lineage/snapshots"),
    Path("runs/ppo_lineage_cont/snapshots"),
)
PILOT = Path(__file__).with_name("10_ppo_pilot.py")
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "runs" / "teacher_seeds"
DEFAULT_LEVELS = "1-7,1-10,1-11"


def parse_levels(value: str) -> list[str]:
    if not value or value.lower() == "all":
        return list(LEVELS)
    levels = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [level for level in levels if level not in LEVELS]
    if unknown:
        raise ValueError(f"Unknown level(s): {', '.join(unknown)}")
    return levels


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


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


def as_float(value, default=float("inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def eligible(summary: dict) -> bool:
    if not summary:
        return False
    agents = as_float(summary.get("agents"), 100.0)
    finishes = as_float(summary.get("finishes"), 0.0)
    if finishes / max(agents, 1.0) < 0.90:
        return False
    return as_float(summary.get("median_time")) != float("inf")


def metric(summary: dict) -> float:
    median = as_float(summary.get("median_time"))
    return median if median != float("inf") else as_float(summary.get("best_time"))


def run_seed(level: str, seed: int, index: int, args, teacher: Path) -> dict:
    level_dir = args.output_dir / level / f"seed-{seed}"
    level_dir.mkdir(parents=True, exist_ok=True)
    log_path = level_dir / "train.log"
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
        str(seed),
    ]
    if args.show:
        command.append("--show")
    if args.dry_run:
        print(" ".join(command), flush=True)
        return {
            "level": level,
            "seed": seed,
            "status": "dry-run",
            "median_time": "",
            "finishes": "",
        }

    print(f"[{level}] seed {seed} from {teacher}", flush=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    summary = read_summary(level_dir / "eval" / "summary.csv")
    status = "ok" if completed.returncode == 0 else "failed"
    print(
        f"[{level}] seed {seed} {status} in {time.time() - started:.1f}s "
        f"median={summary.get('median_time', '?')} "
        f"finishes={summary.get('finishes', '?')}",
        flush=True,
    )
    return {
        "level": level,
        "seed": seed,
        "status": status,
        "median_time": summary.get("median_time", ""),
        "finishes": summary.get("finishes", ""),
        "efficiency_mean": summary.get("efficiency_mean", ""),
        "flow_mean": summary.get("flow_mean", ""),
        "model": str(level_dir / "ppo_pilot.zip"),
        "log": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default=DEFAULT_LEVELS)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--timesteps", type=int, default=25_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--eval-agents", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--min-finish-rate", type=float, default=0.90)
    parser.add_argument("--seed-patience", type=int, default=2)
    parser.add_argument("--base-port", type=int, default=13000)
    parser.add_argument("--eval-port", type=int, default=14000)
    parser.add_argument(
        "--obs-encoding",
        default="legacy",
        choices=("legacy", "multi_hot"),
        help="Observation contract; teachers and student must match.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    levels = parse_levels(args.levels)
    seeds = parse_seeds(args.seeds)

    rows = []
    for level_index, level in enumerate(levels):
        teacher = teacher_path(level)
        if teacher is None:
            print(f"[{level}] no teacher", flush=True)
            continue
        best_model: Path | None = None
        best_summary: dict = {}
        best_metric = float("inf")
        no_improve = 0
        level_rows = []
        for seed_index, seed in enumerate(seeds):
            row = run_seed(level, seed, level_index * len(seeds) + seed_index, args, teacher)
            level_rows.append(row)
            summary = {
                "median_time": row.get("median_time", ""),
                "finishes": row.get("finishes", ""),
                "agents": args.eval_agents,
                "efficiency_mean": row.get("efficiency_mean", ""),
                "flow_mean": row.get("flow_mean", ""),
            }
            if not args.dry_run and eligible(summary) and metric(summary) < best_metric:
                best_metric = metric(summary)
                best_summary = summary
                best_model = Path(row["model"])
                no_improve = 0
            else:
                no_improve += 1
            if not args.dry_run and no_improve >= args.seed_patience:
                print(
                    f"[{level}] seed plateau after {no_improve} non-improving seeds",
                    flush=True,
                )
                break
        if best_model is not None and best_model.exists():
            target_dir = args.output_dir / level
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_model, target_dir / "ppo_pilot.zip")
            best_eval = best_model.parent / "eval" / "summary.csv"
            if best_eval.exists():
                target_eval = target_dir / "eval" / "summary.csv"
                target_eval.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(best_eval, target_eval)
            rows.append(
                {
                    "level": level,
                    "chosen_seed": best_model.parent.name,
                    "median_time": best_summary.get("median_time", ""),
                    "finishes": best_summary.get("finishes", ""),
                    "efficiency_mean": best_summary.get("efficiency_mean", ""),
                    "flow_mean": best_summary.get("flow_mean", ""),
                    "model": str(target_dir / "ppo_pilot.zip"),
                }
            )
        rows.extend(level_rows)

    if not args.dry_run:
        summary_path = args.output_dir / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "level",
                "chosen_seed",
                "median_time",
                "finishes",
                "efficiency_mean",
                "flow_mean",
                "model",
                "seed",
                "status",
                "log",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fields})
        print(f"[seed] summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
