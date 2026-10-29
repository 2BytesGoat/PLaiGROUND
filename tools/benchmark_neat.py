#!/usr/bin/env python3
"""Benchmark NEAT throughput across parallelism and speedup settings.

This is a short, reusable sweep tool. It runs a few generations per combo with
route recording disabled, records wall time/CPU/memory, and writes JSON +
Markdown results. It is intended for choosing defaults on a new machine, not for
producing training data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "0021_neat_explore_1_3" / "neat_config.txt"


def _hardware() -> dict[str, Any]:
    disk = shutil.disk_usage(ROOT)
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "free_disk_gb": round(disk.free / 1e9, 2),
    }


def _tree_rss(pid: int) -> int:
    try:
        process = psutil.Process(pid)
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total
    except psutil.Error:
        return 0


def _history_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    last = rows[-1]
    times = [
        float(row["generation_s"])
        for row in rows
        if row.get("generation_s")
    ]
    return {
        "generations": len(rows),
        "generation_s_mean": statistics.mean(times) if times else None,
        "generation_s_median": statistics.median(times) if times else None,
        "finishes_last": int(float(last.get("finishes", 0) or 0)),
        "best_time_last": last.get("best_time") or None,
        "best_fitness_last": last.get("best_fitness") or None,
    }


def _run_combo(
    *,
    level: str,
    config: Path,
    population: int,
    generations: int,
    parallel: int,
    speedup: float,
    max_steps: int,
    action_repeat: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    log_path = output_dir / "run.log"
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "genetic" / "01_neat.py"),
        "--level",
        level,
        "--config",
        str(config),
        "--population",
        str(population),
        "--generations",
        str(generations),
        "--parallel",
        str(parallel),
        "--action-repeat",
        str(action_repeat),
        "--speedup",
        str(speedup),
        "--max-steps",
        str(max_steps),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--archive-dir",
        str(output_dir / "archive"),
    ]
    start = time.perf_counter()
    peak_rss = 0
    cpu_samples: list[float] = []
    errors: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = psutil.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            return {
                "parallel": parallel,
                "speedup": speedup,
                "error": str(exc),
                "wall_s": None,
            }
        while proc.poll() is None:
            peak_rss = max(peak_rss, _tree_rss(proc.pid))
            cpu_samples.append(psutil.cpu_percent(interval=0.5))
        returncode = int(proc.returncode or 0)
    wall_s = time.perf_counter() - start
    if returncode:
        errors.append(f"exit code {returncode}")
    stats = _history_stats(output_dir / "history.csv")
    if not stats:
        errors.append("no history.csv")
    return {
        "parallel": parallel,
        "speedup": speedup,
        "population": population,
        "generations": generations,
        "wall_s": round(wall_s, 3),
        "peak_rss_gb": round(peak_rss / 1e9, 3),
        "mean_cpu_percent": round(
            statistics.mean(cpu_samples) if cpu_samples else 0.0, 2
        ),
        "returncode": returncode,
        "errors": errors,
        **stats,
    }


def _markdown(hardware: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        "# NEAT parallelism benchmark",
        "",
        "## Hardware",
        "",
        "```text",
    ]
    for key, value in hardware.items():
        lines.append(f"{key}: {value}")
    lines += [
        "```",
        "",
        "Short sweeps have no route recording; use them for throughput only.",
        "",
        "## Results",
        "",
        "| Parallel | Speedup | Wall s | Gen s | Agents/s | Peak RAM GB | CPU % | Finishes | Errors |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        gen_s = row.get("generation_s_mean")
        agents_s = (
            row["population"] * row["generations"] / row["wall_s"]
            if row.get("wall_s")
            else None
        )
        lines.append(
            "| {parallel} | {speedup} | {wall} | {gen} | {agents} | {ram} | {cpu} | {fin} | {err} |".format(
                parallel=row.get("parallel", "?"),
                speedup=row.get("speedup", "?"),
                wall=row.get("wall_s", "?"),
                gen=round(gen_s, 2) if gen_s else "?",
                agents=round(agents_s, 1) if agents_s else "?",
                ram=row.get("peak_rss_gb", "?"),
                cpu=row.get("mean_cpu_percent", "?"),
                fin=row.get("finishes_last", "?"),
                err=", ".join(row.get("errors", [])) or "-",
            )
        )
    lines += ["", "## Recommendation", ""]
    valid = [
        row
        for row in results
        if row.get("wall_s")
        and not row.get("errors")
        and row.get("generation_s_mean")
    ]
    if valid:
        best = min(valid, key=lambda row: float(row["generation_s_mean"]))
        lines += [
            f"Fastest stable combo: **parallel={best['parallel']}, "
            f"speedup={best['speedup']}** "
            f"(generation mean {best['generation_s_mean']:.2f} s, "
            f"wall {best['wall_s']:.1f} s, "
            f"peak RAM {best['peak_rss_gb']:.2f} GB).",
            "",
            "Re-run this benchmark on a new machine before changing defaults.",
        ]
    else:
        lines.append("No valid combo completed.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="1-3")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--population", type=int, default=384)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument(
        "--parallel",
        type=int,
        nargs="+",
        default=[4, 6, 8, 12],
        help="Process counts; values must divide the population.",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        nargs="+",
        default=[8.0, 12.0],
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    host = platform.node().replace(" ", "-") or "host"
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = (
        args.output_dir
        if args.output_dir is not None
        else ROOT / "runs" / "benchmarks" / f"{stamp}_{host}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    hardware = _hardware()
    (out_dir / "hardware.json").write_text(
        json.dumps(hardware, indent=2), encoding="utf-8"
    )

    combos = [
        (parallel, speedup)
        for parallel in args.parallel
        for speedup in args.speedup
        if args.population % parallel == 0
    ]
    invalid = [
        (parallel, speedup)
        for parallel in args.parallel
        for speedup in args.speedup
        if args.population % parallel != 0
    ]
    print(
        f"hardware: {hardware['cpu_count']} CPUs, "
        f"{hardware['total_ram_gb']} GB RAM, {hardware['platform']}",
        flush=True,
    )
    print(
        f"combos: {len(combos)} valid, {len(invalid)} skipped "
        f"(population {args.population} must be divisible by parallel)",
        flush=True,
    )
    for parallel, speedup in invalid:
        print(f"skip parallel={parallel} speedup={speedup}: not divisible", flush=True)
    if args.dry_run:
        for parallel, speedup in combos:
            print(f"would run parallel={parallel} speedup={speedup}", flush=True)
        return 0

    results: list[dict[str, Any]] = []
    for index, (parallel, speedup) in enumerate(combos, start=1):
        name = f"p{parallel}_s{speedup:g}"
        combo_dir = out_dir / name
        combo_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{index}/{len(combos)}] parallel={parallel} speedup={speedup} ...",
            flush=True,
        )
        result = _run_combo(
            level=args.level,
            config=args.config,
            population=args.population,
            generations=args.generations,
            parallel=parallel,
            speedup=speedup,
            max_steps=args.max_steps,
            action_repeat=args.action_repeat,
            seed=args.seed,
            output_dir=combo_dir,
        )
        results.append(result)
        (out_dir / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print(
            f"  wall={result.get('wall_s')}s "
            f"gen_mean={result.get('generation_s_mean')} "
            f"ram={result.get('peak_rss_gb')}GB "
            f"errors={result.get('errors')}",
            flush=True,
        )

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (out_dir / "results.md").write_text(
        _markdown(hardware, results), encoding="utf-8"
    )
    print(f"wrote {out_dir / 'results.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
