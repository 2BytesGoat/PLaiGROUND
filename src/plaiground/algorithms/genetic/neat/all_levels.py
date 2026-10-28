#!/usr/bin/env python3
"""Train one NEAT population per campaign level, then show checkpoints.

This is a thin batch layer over ``scripts/genetic/01_neat.py``. Each level gets an
independent population and output directory. Training runs headless; after
training, one generation from each checkpoint is replayed with the game window
open so the population's behaviour can be inspected as it evolved.

Examples:
    python scripts/genetic/02_neat_all_levels.py
    python scripts/genetic/02_neat_all_levels.py --workers 4 --generations 20
    python scripts/genetic/02_neat_all_levels.py --levels 1-2,1-5,1-16 --no-show-checkpoints
"""

from __future__ import annotations

import argparse
import csv
import math
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LEVELS = [f"1-{i}" for i in range(1, 17)]
REPO = Path(__file__).resolve().parents[5]
DEFAULT_CONFIG = Path(__file__).with_name("neat_1_1_config.txt")
SCRIPT = REPO / "scripts" / "genetic" / "02_neat_all_levels.py"
DEFAULT_OUTPUT = REPO / "runs" / "neat_all"


def parse_levels(value: str) -> list[str]:
    if not value or value.lower() == "all":
        return LEVELS
    levels = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [level for level in levels if level not in LEVELS]
    if unknown:
        raise ValueError(
            f"Unknown level(s): {', '.join(unknown)}. Valid: {', '.join(LEVELS)}"
        )
    return levels


def checkpoint_generation(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[-1])


def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_level(level: str, level_dir: Path, population: int) -> dict:
    rows = read_history(level_dir / "history.csv")
    if not rows:
        return {
            "level": level,
            "status": "missing",
            "generations": 0,
            "first_finish_generation": "",
            "best_time": "",
            "best_fitness": "",
            "final_finishes": "",
            "final_finish_rate": "",
            "final_mean_fitness": "",
        }

    first_finish = next(
        (int(row["generation"]) for row in rows if int(float(row["finishes"])) > 0),
        "",
    )
    finish_times = [
        float(row["best_time"])
        for row in rows
        if row.get("best_time") not in ("", None)
    ]
    best_fitness = max(as_float(row["best_fitness"]) for row in rows)
    last = rows[-1]
    final_finishes = int(float(last["finishes"]))
    best_time = min(finish_times) if finish_times else math.inf

    return {
        "level": level,
        "status": "learned" if math.isfinite(best_time) else "no_finish",
        "generations": len(rows),
        "first_finish_generation": first_finish,
        "best_time": f"{best_time:.3f}" if math.isfinite(best_time) else "",
        "best_fitness": f"{best_fitness:.2f}",
        "final_finishes": final_finishes,
        "final_finish_rate": f"{final_finishes / population:.3f}",
        "final_mean_fitness": f"{as_float(last['mean_fitness']):.2f}",
    }


def run_training(
    level: str,
    args: argparse.Namespace,
    port: int,
    level_dir: Path,
) -> tuple[str, bool, float]:
    level_dir.mkdir(parents=True, exist_ok=True)
    log_path = level_dir / "train.log"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--level",
        level,
        "--config",
        str(args.config),
        "--output-dir",
        str(level_dir),
        "--generations",
        str(args.generations),
        "--population",
        str(args.population),
        "--max-steps",
        str(args.max_steps),
        "--action-repeat",
        "1",
        "--speedup",
        str(args.speedup),
        "--seed",
        str(args.seed),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--port",
        str(port),
    ]

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        print(f"[{level}] training on port {port} -> {log_path}", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        code = proc.wait()
    elapsed = time.time() - start
    return level, code == 0, elapsed


def run_showcase(
    level: str,
    checkpoint: Path,
    args: argparse.Namespace,
    level_dir: Path,
) -> bool:
    gen = checkpoint_generation(checkpoint)
    showcase_dir = level_dir / "showcase" / f"gen-{gen:03d}"
    showcase_dir.mkdir(parents=True, exist_ok=True)
    log_path = showcase_dir / "showcase.log"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--level",
        level,
        "--config",
        str(args.config),
        "--resume",
        str(checkpoint),
        "--generations",
        "1",
        "--max-steps",
        str(args.showcase_max_steps),
        "--action-repeat",
        "1",
        "--speedup",
        str(args.showcase_speedup),
        "--show",
        "--port",
        str(args.showcase_port),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--output-dir",
        str(showcase_dir),
    ]
    print(
        f"[{level}] showing checkpoint generation {gen} "
        f"(speedup {args.showcase_speedup}x)",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        code = proc.wait()
    return code == 0


def write_summary(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.csv"
    fields = [
        "level",
        "status",
        "generations",
        "first_finish_generation",
        "best_time",
        "best_fitness",
        "final_finishes",
        "final_finish_rate",
        "final_mean_fitness",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}", flush=True)


def write_curves(output_dir: Path, levels: list[str], population: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional reporting
        print(f"learning-curve plot skipped: {exc}", flush=True)
        return

    cols = 4
    rows = max(1, math.ceil(len(levels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.1 * rows), squeeze=False)
    for index, level in enumerate(levels):
        ax = axes[index // cols][index % cols]
        history = read_history(output_dir / level / "history.csv")
        generations = [int(row["generation"]) for row in history]
        finishes = [int(float(row["finishes"])) for row in history]
        ax.plot(generations, finishes, color="#2a9d8f", marker="o", markersize=3)
        ax.set_ylim(0, max(1, population))
        ax.set_title(level, loc="left", fontsize=10)
        ax.grid(alpha=0.2)
        if history:
            best = [float(row["best_time"]) for row in history if row["best_time"]]
            if best:
                ax.text(
                    0.98,
                    0.92,
                    f"best {min(best):.2f}s",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color="#264653",
                )
    for index in range(len(levels), rows * cols):
        axes[index // cols][index % cols].axis("off")
    fig.suptitle("NEAT learning progress — finishes per generation", fontsize=15)
    fig.tight_layout()
    path = output_dir / "learning_curves.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"wrote {path}", flush=True)


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
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--speedup", type=float, default=8.0)
    parser.add_argument("--showcase-speedup", type=float, default=4.0)
    parser.add_argument("--showcase-max-steps", type=int, default=600)
    parser.add_argument("--showcase-port", type=int, default=12000)
    parser.add_argument("--base-port", type=int, default=11008)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay each checkpoint population once in the game window.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only summarize/show existing level histories.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain levels that already have a full history.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = parse_levels(args.levels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.workers = max(1, args.workers)

    print(
        f"NEAT all-levels: {len(levels)} levels | population {args.population} | "
        f"{args.generations} generations | {args.workers} workers | "
        f"speedup {args.speedup}x",
        flush=True,
    )

    completed: list[str] = []
    if not args.skip_training:
        port_pool: queue.Queue[int] = queue.Queue()
        for slot in range(args.workers):
            port_pool.put(args.base_port + slot)

        def task(level: str) -> tuple[str, bool, float]:
            level_dir = args.output_dir / level
            history = read_history(level_dir / "history.csv")
            if (
                not args.force
                and len(history) >= args.generations
                and (level_dir / "best_genome.pkl").exists()
            ):
                print(f"[{level}] complete history found; skipping training", flush=True)
                return level, True, 0.0
            port = port_pool.get()
            try:
                return run_training(level, args, port, level_dir)
            finally:
                port_pool.put(port)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(task, level): level for level in levels}
            for future in as_completed(futures):
                level, ok, elapsed = future.result()
                completed.append(level)
                print(
                    f"[{level}] training {'finished' if ok else 'FAILED'} "
                    f"in {elapsed:.1f}s",
                    flush=True,
                )

    rows = [
        summarize_level(level, args.output_dir / level, args.population)
        for level in levels
    ]
    write_summary(rows, args.output_dir)

    print("\nlevel   status     first   best     final")
    for row in rows:
        print(
            f"{row['level']:<7} {row['status']:<9} "
            f"{str(row['first_finish_generation']):<7} "
            f"{str(row['best_time'] or '-'):<8} "
            f"{row['final_finishes']}/{args.population}"
        )

    if args.show_checkpoints:
        for level in levels:
            checkpoints = sorted(
                (args.output_dir / level).glob("neat-checkpoint-*"),
                key=checkpoint_generation,
            )
            for checkpoint in checkpoints:
                if checkpoint_generation(checkpoint) % args.checkpoint_interval != 0:
                    continue
                run_showcase(level, checkpoint, args, args.output_dir / level)

    write_curves(args.output_dir, levels, args.population)


if __name__ == "__main__":
    main()
