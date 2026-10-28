#!/usr/bin/env python3
"""Extract clean-start state pools from NEAT checkpoint snapshots.

The experiment manifest supplies the archive, level analysis, output directory,
and level. Use the overrides for a one-off extraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plaiground.data.route_starts import extract
from plaiground.experiments.runner import load_run, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--routes", type=Path, default=None)
    parser.add_argument("--finishers", type=Path, default=None)
    parser.add_argument("--level-analysis", type=Path, default=None)
    parser.add_argument("--min-time-to-exit", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _required_path(value: str | Path | None, name: str) -> Path:
    if value is None:
        raise SystemExit(f"missing {name}")
    path = resolve_path(value)
    if not path.exists():
        raise SystemExit(f"missing {name}: {path}")
    return path


def main() -> int:
    args = parse_args()
    run = load_run(args.config, output_dir=args.output_dir)
    run.ensure_dirs()
    data = run.data

    routes_path = _required_path(
        args.routes or data.get("archive"),
        "[data] archive",
    )
    analysis_path = _required_path(
        args.level_analysis or data.get("level_analysis"),
        "[data] level_analysis",
    )
    finishers_value = args.finishers or data.get("finishers")
    finishers_path = (
        resolve_path(finishers_value)
        if finishers_value is not None
        else run.output_dir / "archive" / "finishers.jsonl"
    )
    min_time_to_exit = (
        float(args.min_time_to_exit)
        if args.min_time_to_exit is not None
        else float(data.get("min_time_to_exit", 0.5))
    )

    payload = extract(
        routes_path,
        finishers_path,
        analysis_path,
        run.output_dir,
        min_time_to_exit,
        run.seed,
        run.speedup,
        level=run.level,
    )
    start_pool = run.output_dir / "spawns" / f"{run.level}_start_pool.json"
    print(
        json.dumps(
            {
                "level": run.level,
                "train_routes": payload["train_routes"],
                "train_spawns": payload["train_spawns"],
                "start_pool": str(start_pool),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
