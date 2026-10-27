"""Train the 512x512 multi-hot 8x8 teacher for every campaign level.

Runs one level at a time, sequentially, because each run already uses eight
Godot processes (8 agents/process). Existing runs with a best_model.zip are
skipped by default; 1-11 is already complete.

Example:
    .venv/bin/python src/20_train_all_levels_8x8.py
    .venv/bin/python src/20_train_all_levels_8x8.py --levels 1-1,1-2 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

PILOT = Path(__file__).with_name("10_ppo_pilot.py")
DEFAULT_LEVELS = [f"1-{i}" for i in range(1, 17)]
PATIENCE_BY_LEVEL = {
    **{f"1-{i}": 5 for i in range(1, 7)},
    **{f"1-{i}": 8 for i in range(7, 11)},
    **{f"1-{i}": 13 for i in range(11, 17)},
}
MAX_TIMESTEPS_BY_LEVEL = {
    **{f"1-{i}": 1_200_000 for i in range(1, 7)},
    **{f"1-{i}": 1_800_000 for i in range(7, 11)},
    **{f"1-{i}": 2_400_000 for i in range(11, 17)},
}
REFERENCE_BEST = {
    "1-1": 2.333333,
    "1-2": 3.016667,
    "1-3": 3.233333,
    "1-4": 4.250000,
    "1-5": 2.966667,
    "1-6": 5.050000,
    "1-7": 7.616667,
    "1-8": 3.216667,
    "1-9": 2.900000,
    "1-10": 4.466667,
    "1-11": 8.050000,
    "1-12": 5.966667,
    "1-13": 6.266667,
    "1-14": 6.816667,
    "1-15": 5.850000,
    "1-16": 6.133333,
}
REGRESSION_TOLERANCE = 1.05


def parse_levels(value: str) -> list[str]:
    levels = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [level for level in levels if level not in DEFAULT_LEVELS]
    if unknown:
        raise ValueError(f"Unknown level(s): {unknown}")
    return levels


def read_best_from_log(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"best \S+ median=([0-9.]+)s", text)
    if not matches:
        return ""
    return min(matches, key=float)


def read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def result_status(level: str, best_time: str) -> str:
    if not best_time:
        return "no_clear"
    reference = REFERENCE_BEST.get(level)
    if reference is None:
        return "ok"
    try:
        value = float(best_time)
    except ValueError:
        return "ok"
    if value > reference * REGRESSION_TOLERANCE:
        return "regression"
    return "ok"


def write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default=",".join(DEFAULT_LEVELS))
    parser.add_argument("--skip-levels", default="1-11")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override per-level cap (easy 1.2M, mid 1.8M, hard 2.4M).",
    )
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override the per-level default (easy 5, mid 8, hard 13).",
    )
    parser.add_argument("--eval-agents", type=int, default=100)
    parser.add_argument("--agents-per-process", type=int, default=8)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--obs-encoding", default="multi_hot")
    parser.add_argument("--reward-profile", default="time")
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-port", type=int, default=14608)
    parser.add_argument("--eval-port", type=int, default=15600)
    parser.add_argument("--output-root", type=Path, default=Path("runs/topology/all_levels_8x8_w512"))
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    levels = [level for level in parse_levels(args.levels) if level not in parse_levels(args.skip_levels)]
    args.output_root = args.output_root.resolve()
    rows: list[dict] = []
    started_all = time.time()

    for index, level in enumerate(levels):
        patience = (
            args.patience
            if args.patience is not None
            else PATIENCE_BY_LEVEL.get(level, 13)
        )
        timesteps = (
            args.timesteps
            if args.timesteps is not None
            else MAX_TIMESTEPS_BY_LEVEL.get(level, 2_400_000)
        )
        run_dir = args.output_root / level
        log_path = run_dir / "train.log"
        summary_path = run_dir / "eval" / "summary.csv"
        best_path = run_dir / "best_model.zip"
        if args.skip_existing and best_path.exists():
            best_time = read_best_from_log(log_path)
            status = result_status(level, best_time)
            print(
                f"[{level}] existing best checkpoint ({status}); skipping",
                flush=True,
            )
            rows.append(
                {
                    "level": level,
                    "status": status,
                    "best_time": best_time,
                    "reference_time": REFERENCE_BEST.get(level, ""),
                    "final_time": read_summary(summary_path).get("best_time", ""),
                    "finishes": read_summary(summary_path).get("finishes", ""),
                    "deaths": read_summary(summary_path).get("deaths", ""),
                    "timeouts": read_summary(summary_path).get("timeouts", ""),
                    "patience": patience,
                    "timesteps": timesteps,
                    "wall_s": "",
                    "output_dir": str(run_dir),
                }
            )
            write_summary(args.output_root / "all_levels_summary.csv", rows)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(PILOT),
            "--level", level,
            "--n-envs", str(args.processes),
            "--agents-per-process", str(args.agents_per_process),
            "--timesteps", str(timesteps),
            "--max-decisions", str(args.max_decisions),
            "--action-repeat", str(args.action_repeat),
            "--speedup", "8",
            "--reward-profile", args.reward_profile,
            "--obs-encoding", args.obs_encoding,
            "--net-arch", *[str(size) for size in args.net_arch],
            "--eval-agents", str(args.eval_agents),
            "--eval-every", str(args.eval_every),
            "--patience", str(patience),
            "--min-finish-rate", "0.90",
            "--save-freq", "200000",
            "--seed", str(args.seed),
            "--base-port", str(args.base_port + index * 20),
            "--eval-port", str(args.eval_port + index * 20),
            "--output-dir", str(run_dir),
        ]
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue

        started = time.time()
        print(
            f"[{level}] starting (patience={patience}, timesteps={timesteps})",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.time() - started
        summary = read_summary(summary_path)
        best_time = read_best_from_log(log_path)
        if completed.returncode != 0:
            status = "failed"
        elif best_path.exists():
            status = result_status(level, best_time)
        else:
            status = "no_clear"
        rows.append(
            {
                "level": level,
                "status": status,
                "best_time": best_time,
                "reference_time": REFERENCE_BEST.get(level, ""),
                "final_time": summary.get("best_time", ""),
                "finishes": summary.get("finishes", ""),
                "deaths": summary.get("deaths", ""),
                "timeouts": summary.get("timeouts", ""),
                "patience": patience,
                "timesteps": timesteps,
                "wall_s": f"{elapsed:.1f}",
                "output_dir": str(run_dir),
            }
        )
        write_summary(args.output_root / "all_levels_summary.csv", rows)
        print(
            f"[{level}] {status} in {elapsed:.0f}s best={best_time or '?'}",
            flush=True,
        )
        if status == "failed":
            print(f"[{level}] stopping after failure; see {log_path}", flush=True)
            return 1

    print(f"[all-levels] done in {time.time() - started_all:.0f}s", flush=True)
    print(f"[all-levels] summary: {args.output_root / 'all_levels_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
