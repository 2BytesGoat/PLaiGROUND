#!/usr/bin/env python3
"""Report training episodes by start source, with a canonical-only mode.

The aggregate ``episodes.csv`` mixes canonical, medium/hard cell, focus, and
NEAT-prefix starts. Comparing the best time from one start family with the
median of the whole mix is misleading. This tool keeps those populations
separate and makes canonical-only speed reporting the default view when
``--canonical-only`` is used.

Usage:
    python src/36_episode_stats.py runs/.../episodes.csv --level 1-3
    python src/36_episode_stats.py runs/.../episodes.csv --canonical-only
    python src/36_episode_stats.py runs/.../episodes.csv --group-by start_cell
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def as_bool(value: str | None) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def start_kind(row: dict) -> str:
    kind = str(row.get("start_kind") or "").strip()
    if kind:
        return kind
    # Older episode files did not have start_kind. Keep them visible rather
    # than silently folding them into canonical statistics.
    source = str(row.get("start_source") or "").strip()
    return source or "unknown"


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarise(rows: list[dict]) -> dict:
    finished_rows = [
        row for row in rows if as_bool(row.get("finished")) and row.get("run_time")
    ]
    times = [float(row["run_time"]) for row in finished_rows]
    steps = [
        int(float(row["episode_length"]))
        for row in finished_rows
        if row.get("episode_length")
    ]
    return {
        "episodes": len(rows),
        "finished": len(finished_rows),
        "timeouts": sum(as_bool(row.get("timed_out")) for row in rows),
        "finish_rate": len(finished_rows) / len(rows) if rows else None,
        "best": min(times) if times else None,
        "median": statistics.median(times) if times else None,
        "p90": percentile(times, 0.90),
        "mean": statistics.mean(times) if times else None,
        "median_steps": statistics.median(steps) if steps else None,
    }


def format_value(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("--level", default=None)
    parser.add_argument(
        "--group-by",
        choices=("start_kind", "start_source", "start_cell", "level"),
        default="start_kind",
    )
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="Use only rows whose start_kind is canonical; group by level.",
    )
    parser.add_argument("--min-episodes", type=int, default=1)
    args = parser.parse_args()

    rows = load_rows(args.episodes)
    if args.level:
        rows = [row for row in rows if row.get("level") == args.level]
    if args.canonical_only:
        rows = [row for row in rows if start_kind(row) == "canonical"]

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if args.canonical_only:
            key = row.get("level") or "?"
        elif args.group_by == "start_kind":
            key = start_kind(row)
        elif args.group_by == "start_source":
            key = row.get("start_source") or "unknown"
        elif args.group_by == "start_cell":
            cell = row.get("start_cell") or row.get("spawn_cell") or "?"
            facing = row.get("start_facing") or "?"
            key = f"{cell} facing {facing}"
        else:
            key = row.get("level") or "?"
        groups[str(key)].append(row)

    print(f"episodes: {args.episodes}")
    if args.level:
        print(f"level filter: {args.level}")
    if args.canonical_only:
        print("filter: canonical starts only")
    print()
    print("| group | episodes | finished | timeouts | best | median | p90 | mean |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(groups):
        group = groups[key]
        if len(group) < args.min_episodes:
            continue
        summary = summarise(group)
        print(
            f"| {key} | {summary['episodes']} | {summary['finished']} | "
            f"{summary['timeouts']} | {format_value(summary['best'])} | "
            f"{format_value(summary['median'])} | "
            f"{format_value(summary['p90'])} | "
            f"{format_value(summary['mean'])} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
